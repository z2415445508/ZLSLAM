"""
光流一致性检查模块

实现前后向一致性检查、时序一致性检查和遮挡检测
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List


class ConsistencyChecker:
    """光流一致性检查器"""
    
    def __init__(self, config: dict):
        """
        初始化一致性检查器
        
        Args:
            config: 配置字典，包含以下参数:
                - fb_threshold: 前后向一致性阈值（像素）
                - temporal_window: 时序一致性窗口大小
                - occlusion_threshold: 遮挡检测阈值
        """
        self.fb_threshold = config.get('fb_threshold', 1.0)
        self.temporal_window = config.get('temporal_window', 3)
        self.occlusion_threshold = config.get('occlusion_threshold', 1.5)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward_backward_check(
        self, 
        flow_fwd: torch.Tensor, 
        flow_bwd: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前后向一致性检查
        
        基本原理：对于静态场景，前向光流和后向光流应该互为逆运算
        即: flow_fwd(p) + flow_bwd(p + flow_fwd(p)) ≈ 0
        
        Args:
            flow_fwd: 前向光流 [B, 2, H, W] 或 [H, W, 2]
            flow_bwd: 后向光流 [B, 2, H, W] 或 [H, W, 2]
            
        Returns:
            consistency_mask: 一致性掩码 [B, H, W] 或 [H, W]，True表示一致
            fb_error: 前后向误差 [B, H, W] 或 [H, W]
        """
        # 统一格式为 [B, 2, H, W]
        if flow_fwd.dim() == 3:  # [H, W, 2]
            flow_fwd = flow_fwd.permute(2, 0, 1).unsqueeze(0)
            flow_bwd = flow_bwd.permute(2, 0, 1).unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        B, _, H, W = flow_fwd.shape
        
        # 创建网格坐标
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=flow_fwd.device, dtype=flow_fwd.dtype),
            torch.arange(W, device=flow_fwd.device, dtype=flow_fwd.dtype),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # [1, 2, H, W]
        
        # 计算前向光流的目标位置
        target_pos = grid + flow_fwd  # [B, 2, H, W]
        
        # 归一化坐标到[-1, 1]用于grid_sample
        target_pos_norm = target_pos.clone()
        target_pos_norm[:, 0] = 2.0 * target_pos[:, 0] / (W - 1) - 1.0
        target_pos_norm[:, 1] = 2.0 * target_pos[:, 1] / (H - 1) - 1.0
        target_pos_norm = target_pos_norm.permute(0, 2, 3, 1)  # [B, H, W, 2]
        
        # 在目标位置采样后向光流
        flow_bwd_warped = F.grid_sample(
            flow_bwd, 
            target_pos_norm, 
            mode='bilinear', 
            padding_mode='zeros',
            align_corners=True
        )  # [B, 2, H, W]
        
        # 计算前后向一致性误差
        fb_diff = flow_fwd + flow_bwd_warped
        fb_error = torch.norm(fb_diff, p=2, dim=1)  # [B, H, W]
        
        # 生成一致性掩码
        consistency_mask = fb_error < self.fb_threshold
        
        if squeeze_output:
            consistency_mask = consistency_mask.squeeze(0)
            fb_error = fb_error.squeeze(0)
            
        return consistency_mask, fb_error
    
    def compute_occlusion_mask(
        self, 
        flow_fwd: torch.Tensor, 
        flow_bwd: torch.Tensor
    ) -> torch.Tensor:
        """
        计算遮挡掩码
        
        遮挡区域的特征：
        1. 前后向一致性误差大
        2. 光流幅值较大
        
        Args:
            flow_fwd: 前向光流 [B, 2, H, W] 或 [H, W, 2]
            flow_bwd: 后向光流 [B, 2, H, W] 或 [H, W, 2]
            
        Returns:
            occlusion_mask: 遮挡掩码 [B, H, W] 或 [H, W]，True表示遮挡
        """
        # 计算前后向一致性
        consistency_mask, fb_error = self.forward_backward_check(flow_fwd, flow_bwd)
        
        # 统一格式
        if flow_fwd.dim() == 3:
            flow_fwd = flow_fwd.permute(2, 0, 1).unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # 计算光流幅值
        flow_magnitude = torch.norm(flow_fwd, p=2, dim=1)  # [B, H, W]
        
        # 遮挡判定：一致性差且光流幅值大
        occlusion_mask = torch.logical_and(
            ~consistency_mask,
            flow_magnitude > self.occlusion_threshold
        )
        
        if squeeze_output:
            occlusion_mask = occlusion_mask.squeeze(0)
            
        return occlusion_mask
    
    def temporal_consistency_check(
        self, 
        flow_history: List[torch.Tensor],
        weights: Optional[List[float]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        时序一致性检查
        
        通过分析连续帧的光流变化，检测时序上不一致的区域
        
        Args:
            flow_history: 光流历史列表，每个元素为 [H, W, 2] 或 [B, 2, H, W]
            weights: 时序权重，越近的帧权重越大
            
        Returns:
            temporal_consistency: 时序一致性分数 [H, W]
            temporal_mask: 时序一致性掩码 [H, W]
        """
        if len(flow_history) < 2:
            raise ValueError("需要至少2帧光流进行时序一致性检查")
        
        # 统一格式
        flows = []
        for flow in flow_history:
            if flow.dim() == 3:  # [H, W, 2]
                flow = flow.permute(2, 0, 1).unsqueeze(0)
            flows.append(flow)
        
        # 默认权重：越近的帧权重越大
        if weights is None:
            weights = [1.0 / (i + 1) for i in range(len(flows))]
            weights = [w / sum(weights) for w in weights]
        
        # 计算加权平均光流
        weighted_flow = sum(w * f for w, f in zip(weights, flows))
        
        # 计算每帧与平均光流的偏差
        deviations = []
        for flow in flows:
            diff = flow - weighted_flow
            deviation = torch.norm(diff, p=2, dim=1)  # [B, H, W]
            deviations.append(deviation)
        
        # 计算时序一致性分数（偏差的标准差）
        deviations_stack = torch.stack(deviations, dim=0)  # [T, B, H, W]
        temporal_variance = torch.var(deviations_stack, dim=0)  # [B, H, W]
        temporal_consistency = 1.0 / (1.0 + temporal_variance)  # 归一化到[0, 1]
        
        # 生成时序一致性掩码
        temporal_mask = temporal_consistency > 0.5
        
        return temporal_consistency.squeeze(0), temporal_mask.squeeze(0)
    
    def compute_confidence_map(
        self,
        fb_error: torch.Tensor,
        flow_magnitude: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算光流置信度图
        
        置信度基于：
        1. 前后向一致性误差（越小越好）
        2. 光流幅值（适中最好）
        
        Args:
            fb_error: 前后向误差 [H, W]
            flow_magnitude: 光流幅值 [H, W]（可选）
            
        Returns:
            confidence_map: 置信度图 [H, W]，范围[0, 1]
        """
        # 基于前后向误差的置信度
        fb_confidence = torch.exp(-fb_error / self.fb_threshold)
        
        if flow_magnitude is not None:
            # 基于光流幅值的置信度（使用高斯函数，峰值在中等幅值）
            optimal_magnitude = 5.0  # 最优光流幅值
            magnitude_confidence = torch.exp(
                -((flow_magnitude - optimal_magnitude) ** 2) / (2 * optimal_magnitude ** 2)
            )
            
            # 综合置信度
            confidence_map = fb_confidence * magnitude_confidence
        else:
            confidence_map = fb_confidence
        
        return confidence_map
    
    def adaptive_threshold(
        self,
        fb_error: torch.Tensor,
        percentile: float = 75.0
    ) -> float:
        """
        自适应阈值计算
        
        根据当前帧的误差分布动态调整阈值
        
        Args:
            fb_error: 前后向误差 [H, W]
            percentile: 百分位数
            
        Returns:
            threshold: 自适应阈值
        """
        # 计算指定百分位数的误差值
        threshold = torch.quantile(fb_error.flatten(), percentile / 100.0)
        
        # 限制阈值范围
        threshold = torch.clamp(threshold, min=0.5, max=5.0)
        
        return threshold.item()
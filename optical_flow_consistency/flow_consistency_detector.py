"""
光流一致性检测器主模块

整合一致性检查、几何验证和动态分割，提供统一的检测接口
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .consistency_checker import ConsistencyChecker
from .geometric_validator import GeometricValidator
from .dynamic_segmentation import DynamicSegmentation


class FlowConsistencyDetector:
    """光流一致性动态检测器"""
    
    def __init__(self, config: dict, flow_model=None):
        """
        初始化光流一致性检测器
        
        Args:
            config: 配置字典
            flow_model: 光流估计模型 (RAFT/GMA)，如果为None则需要外部提供光流
        """
        self.config = config
        self.flow_model = flow_model
        
        # 初始化子模块
        self.consistency_checker = ConsistencyChecker(config)
        self.geometric_validator = GeometricValidator(config)
        self.segmentation = DynamicSegmentation(config)
        
        # 阈值参数
        self.fb_threshold = config.get('fb_threshold', 1.0)
        self.geo_threshold = config.get('geo_threshold', 2.0)
        self.epi_threshold = config.get('epi_threshold', 1.0)
        
        # 融合参数
        self.use_fb_check = config.get('use_fb_check', True)
        self.use_geo_check = config.get('use_geo_check', True)
        self.use_epi_check = config.get('use_epi_check', True)
        self.use_temporal = config.get('use_temporal', False)
        
        # 权重参数
        self.fb_weight = config.get('fb_weight', 0.4)
        self.geo_weight = config.get('geo_weight', 0.4)
        self.epi_weight = config.get('epi_weight', 0.2)
        
        # 时序缓存
        self.flow_history = []
        self.mask_history = []
        self.max_history = config.get('temporal_window', 3)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def detect_dynamic_regions(
        self,
        frame_t: torch.Tensor,
        frame_t1: torch.Tensor,
        pose_t: torch.Tensor,
        pose_t1: torch.Tensor,
        depth_t: torch.Tensor,
        K: torch.Tensor,
        flow_fwd: Optional[torch.Tensor] = None,
        flow_bwd: Optional[torch.Tensor] = None,
        semantic_mask: Optional[torch.Tensor] = None,
        return_details: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        检测动态区域
        
        Args:
            frame_t: 时刻t的图像 [3, H, W] 或 [H, W, 3]
            frame_t1: 时刻t+1的图像 [3, H, W] 或 [H, W, 3]
            pose_t: 时刻t的相机位姿 [4, 4]
            pose_t1: 时刻t+1的相机位姿 [4, 4]
            depth_t: 时刻t的深度图 [H, W]
            K: 相机内参矩阵 [3, 3]
            flow_fwd: 前向光流 [H, W, 2]（可选，如果为None则计算）
            flow_bwd: 后向光流 [H, W, 2]（可选）
            semantic_mask: 语义分割掩码 [H, W]（可选）
            return_details: 是否返回详细信息
            
        Returns:
            results: 包含以下键的字典:
                - dynamic_mask: 动态区域掩码 [H, W]
                - confidence_map: 置信度图 [H, W]（如果return_details=True）
                - consistency_map: 一致性图 [H, W]（如果return_details=True）
                - region_labels: 区域标签 [H, W]（如果return_details=True）
        """
        # 确保输入在正确的设备上
        frame_t = frame_t.to(self.device)
        frame_t1 = frame_t1.to(self.device)
        depth_t = depth_t.to(self.device)
        
        # 计算或使用提供的光流
        if flow_fwd is None or flow_bwd is None:
            flow_fwd, flow_bwd = self.compute_flow_pair(frame_t, frame_t1)
        else:
            flow_fwd = flow_fwd.to(self.device)
            flow_bwd = flow_bwd.to(self.device)
        
        # 1. 前后向一致性检查
        fb_consistency_mask = None
        fb_error = None
        if self.use_fb_check:
            fb_consistency_mask, fb_error = self.consistency_checker.forward_backward_check(
                flow_fwd, flow_bwd
            )
        
        # 2. 几何一致性验证
        geo_consistency_score = None
        geo_mask = None
        if self.use_geo_check:
            # 计算预期光流
            flow_expected, valid_mask = self.geometric_validator.compute_expected_flow(
                pose_t, pose_t1, depth_t, K
            )
            
            # 计算几何一致性
            geo_consistency_score, geo_mask = self.geometric_validator.compute_geometric_consistency(
                flow_fwd, flow_expected, valid_mask
            )
        
        # 3. 极线约束验证（可选）
        epi_mask = None
        if self.use_epi_check:
            # 构建像素坐标
            H, W = flow_fwd.shape[:2]
            y_coords, x_coords = torch.meshgrid(
                torch.arange(H, device=self.device, dtype=torch.float32),
                torch.arange(W, device=self.device, dtype=torch.float32),
                indexing='ij'
            )
            pts1 = torch.stack([x_coords, y_coords], dim=0)  # [2, H, W]
            pts2 = pts1 + flow_fwd.permute(2, 0, 1)  # [2, H, W]
            
            # 计算相对位姿
            if isinstance(pose_t, torch.Tensor):
                pose_t_inv = torch.inverse(pose_t)
                T_21 = pose_t1 @ pose_t_inv
            else:
                pose_t_inv = np.linalg.inv(pose_t)
                T_21 = pose_t1 @ pose_t_inv
            
            # 验证极线约束
            epi_error, epi_mask = self.geometric_validator.validate_epipolar_constraint(
                pts1, pts2, T_21, K
            )
        
        # 4. 融合多个一致性检查结果
        consistency_map, confidence_map = self.compute_consistency_map(
            fb_consistency_mask, fb_error,
            geo_consistency_score, geo_mask,
            epi_mask
        )
        
        # 5. 时序一致性（如果启用）
        if self.use_temporal and len(self.flow_history) > 0:
            self.flow_history.append(flow_fwd)
            if len(self.flow_history) > self.max_history:
                self.flow_history.pop(0)
            
            temporal_consistency, temporal_mask = self.consistency_checker.temporal_consistency_check(
                self.flow_history
            )
            
            # 融合时序信息
            consistency_map = consistency_map * temporal_consistency
        
        # 6. 分割动态区域
        dynamic_mask, region_labels = self.segmentation.segment_dynamic_regions(
            consistency_map, confidence_map
        )
        
        # 7. 精细化掩码
        dynamic_mask = self.segmentation.refine_mask(
            dynamic_mask, frame_t
        )
        
        # 8. 融合语义信息（如果提供）
        if semantic_mask is not None:
            semantic_weight = self.config.get('semantic_weight', 0.3)
            dynamic_mask = self.segmentation.merge_with_semantic(
                dynamic_mask, semantic_mask, semantic_weight
            )
        
        # 9. 时序滤波（如果启用）
        if self.use_temporal:
            self.mask_history.append(dynamic_mask)
            if len(self.mask_history) > self.max_history:
                self.mask_history.pop(0)
            
            if len(self.mask_history) >= 2:
                dynamic_mask = self.segmentation.temporal_filtering(
                    self.mask_history
                )
        
        # 构建返回结果
        results = {
            'dynamic_mask': dynamic_mask,
        }
        
        if return_details:
            results.update({
                'confidence_map': confidence_map,
                'consistency_map': consistency_map,
                'region_labels': region_labels,
                'fb_error': fb_error,
                'geo_score': geo_consistency_score,
                'flow_fwd': flow_fwd,
                'flow_bwd': flow_bwd,
            })
        
        return results
    
    def compute_flow_pair(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算前向和后向光流
        
        Args:
            img1: 第一帧图像 [3, H, W] 或 [H, W, 3]
            img2: 第二帧图像 [3, H, W] 或 [H, W, 3]
            
        Returns:
            flow_fwd: 前向光流 [H, W, 2]
            flow_bwd: 后向光流 [H, W, 2]
        """
        if self.flow_model is None:
            raise ValueError("未提供光流模型，无法计算光流")
        
        # 统一格式为 [B, 3, H, W]
        if img1.dim() == 3:
            if img1.shape[0] == 3:  # [3, H, W]
                img1 = img1.unsqueeze(0)
                img2 = img2.unsqueeze(0)
            else:  # [H, W, 3]
                img1 = img1.permute(2, 0, 1).unsqueeze(0)
                img2 = img2.permute(2, 0, 1).unsqueeze(0)
        
        # 计算前向光流
        with torch.no_grad():
            flow_fwd_list = self.flow_model(img1, img2, iters=12, test_mode=True)
            if isinstance(flow_fwd_list, tuple):
                flow_fwd = flow_fwd_list[-1]
            else:
                flow_fwd = flow_fwd_list
        
        # 计算后向光流
        with torch.no_grad():
            flow_bwd_list = self.flow_model(img2, img1, iters=12, test_mode=True)
            if isinstance(flow_bwd_list, tuple):
                flow_bwd = flow_bwd_list[-1]
            else:
                flow_bwd = flow_bwd_list
        
        # 转换格式为 [H, W, 2]
        flow_fwd = flow_fwd.squeeze(0).permute(1, 2, 0)
        flow_bwd = flow_bwd.squeeze(0).permute(1, 2, 0)
        
        return flow_fwd, flow_bwd
    
    def compute_consistency_map(
        self,
        fb_mask: Optional[torch.Tensor],
        fb_error: Optional[torch.Tensor],
        geo_score: Optional[torch.Tensor],
        geo_mask: Optional[torch.Tensor],
        epi_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算综合一致性图
        
        融合多个一致性检查的结果
        
        Args:
            fb_mask: 前后向一致性掩码 [H, W]
            fb_error: 前后向误差 [H, W]
            geo_score: 几何一致性分数 [H, W]
            geo_mask: 几何一致性掩码 [H, W]
            epi_mask: 极线一致性掩码 [H, W]
            
        Returns:
            consistency_map: 综合一致性图 [H, W]，值越大越一致
            confidence_map: 置信度图 [H, W]
        """
        # 初始化
        consistency_scores = []
        weights = []
        
        # 前后向一致性
        if fb_mask is not None and fb_error is not None:
            fb_score = torch.exp(-fb_error / self.fb_threshold)
            consistency_scores.append(fb_score)
            weights.append(self.fb_weight)
        
        # 几何一致性
        if geo_score is not None:
            consistency_scores.append(geo_score)
            weights.append(self.geo_weight)
        
        # 极线一致性
        if epi_mask is not None:
            epi_score = epi_mask.float()
            consistency_scores.append(epi_score)
            weights.append(self.epi_weight)
        
        # 归一化权重
        if len(weights) > 0:
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
        
        # 加权融合
        if len(consistency_scores) > 0:
            consistency_map = sum(w * s for w, s in zip(weights, consistency_scores))
        else:
            # 如果没有任何一致性检查，返回全1
            H, W = fb_error.shape if fb_error is not None else (480, 640)
            consistency_map = torch.ones(H, W, device=self.device)
        
        # 计算置信度（基于一致性的方差）
        if len(consistency_scores) > 1:
            scores_stack = torch.stack(consistency_scores, dim=0)
            variance = torch.var(scores_stack, dim=0)
            confidence_map = 1.0 / (1.0 + variance)
        else:
            confidence_map = torch.ones_like(consistency_map)
        
        return consistency_map, confidence_map
    
    def reset_history(self):
        """重置时序历史"""
        self.flow_history = []
        self.mask_history = []
    
    def get_statistics(self, region_labels: torch.Tensor, flow: torch.Tensor) -> dict:
        """
        获取区域统计信息
        
        Args:
            region_labels: 区域标签 [H, W]
            flow: 光流场 [H, W, 2]
            
        Returns:
            statistics: 区域统计信息字典
        """
        return self.segmentation.compute_region_statistics(region_labels, flow)
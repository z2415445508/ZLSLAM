"""
几何验证模块

实现基于相机运动和深度的几何一致性验证
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flow_utils import compute_epipolar_distance


class GeometricValidator:
    """基于几何约束的验证器"""
    
    def __init__(self, config: dict):
        """
        初始化几何验证器
        
        Args:
            config: 配置字典，包含以下参数:
                - geo_threshold: 几何一致性阈值（像素）
                - epi_threshold: 极线距离阈值（像素）
                - min_depth: 最小有效深度
                - max_depth: 最大有效深度
        """
        self.geo_threshold = config.get('geo_threshold', 2.0)
        self.epi_threshold = config.get('epi_threshold', 1.0)
        self.min_depth = config.get('min_depth', 0.1)
        self.max_depth = config.get('max_depth', 10.0)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def compute_expected_flow(
        self,
        pose_t: torch.Tensor,
        pose_t1: torch.Tensor,
        depth: torch.Tensor,
        K: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据相机运动和深度计算预期光流
        
        基本原理：
        1. 将像素点反投影到3D空间
        2. 应用相机运动变换
        3. 投影回图像平面
        4. 计算像素位移
        
        Args:
            pose_t: 时刻t的相机位姿 [4, 4] (World to Camera)
            pose_t1: 时刻t+1的相机位姿 [4, 4] (World to Camera)
            depth: 深度图 [H, W]
            K: 相机内参矩阵 [3, 3]
            
        Returns:
            expected_flow: 预期光流 [H, W, 2]
            valid_mask: 有效像素掩码 [H, W]
        """
        H, W = depth.shape
        device = depth.device
        
        # 确保输入在正确的设备上
        if not isinstance(pose_t, torch.Tensor):
            pose_t = torch.tensor(pose_t, dtype=torch.float32, device=device)
        if not isinstance(pose_t1, torch.Tensor):
            pose_t1 = torch.tensor(pose_t1, dtype=torch.float32, device=device)
        if not isinstance(K, torch.Tensor):
            K = torch.tensor(K, dtype=torch.float32, device=device)
        
        # 创建像素坐标网格
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        # 深度有效性掩码
        valid_depth = torch.logical_and(
            depth > self.min_depth,
            depth < self.max_depth
        )
        
        # 构建齐次像素坐标 [3, H*W]
        ones = torch.ones_like(x_coords)
        pixel_coords = torch.stack([x_coords, y_coords, ones], dim=0)  # [3, H, W]
        pixel_coords = pixel_coords.reshape(3, -1)  # [3, H*W]
        
        # 反投影到相机坐标系
        K_inv = torch.inverse(K)
        cam_coords = K_inv @ pixel_coords  # [3, H*W]
        cam_coords = cam_coords * depth.reshape(1, -1)  # [3, H*W]
        
        # 转换到齐次坐标
        cam_coords_homo = torch.cat([
            cam_coords,
            torch.ones(1, H*W, device=device)
        ], dim=0)  # [4, H*W]
        
        # 计算相对位姿变换 T_t1_t = pose_t1 @ inv(pose_t)
        pose_t_inv = torch.inverse(pose_t)
        T_relative = pose_t1 @ pose_t_inv
        
        # 变换到t+1时刻的相机坐标系
        cam_coords_t1 = T_relative @ cam_coords_homo  # [4, H*W]
        cam_coords_t1 = cam_coords_t1[:3, :]  # [3, H*W]
        
        # 投影到t+1时刻的图像平面
        pixel_coords_t1 = K @ cam_coords_t1  # [3, H*W]
        
        # 归一化齐次坐标
        depth_t1 = pixel_coords_t1[2:3, :]  # [1, H*W]
        pixel_coords_t1 = pixel_coords_t1[:2, :] / (depth_t1 + 1e-8)  # [2, H*W]
        
        # 计算光流
        flow = pixel_coords_t1 - pixel_coords[:2, :]  # [2, H*W]
        flow = flow.reshape(2, H, W).permute(1, 2, 0)  # [H, W, 2]
        
        # 有效性掩码：深度有效 且 投影在图像内 且 深度为正
        in_bounds = torch.logical_and(
            torch.logical_and(
                pixel_coords_t1[0, :] >= 0,
                pixel_coords_t1[0, :] < W
            ),
            torch.logical_and(
                pixel_coords_t1[1, :] >= 0,
                pixel_coords_t1[1, :] < H
            )
        ).reshape(H, W)
        
        positive_depth = (depth_t1 > 0).reshape(H, W)
        
        valid_mask = torch.logical_and(
            torch.logical_and(valid_depth, in_bounds),
            positive_depth
        )
        
        return flow, valid_mask
    
    def validate_epipolar_constraint(
        self,
        pts1: torch.Tensor,
        pts2: torch.Tensor,
        T_21: torch.Tensor,
        K: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        验证极线约束
        
        对于静态点，其在两帧图像中的对应点应满足极线约束：
        p2^T * F * p1 = 0
        其中 F = K^(-T) * [t]_× * R * K^(-1)
        
        Args:
            pts1: 第一帧中的点 [N, 2] 或 [2, H, W]
            pts2: 第二帧中的点 [N, 2] 或 [2, H, W]
            T_21: 相对位姿变换 [4, 4]
            K: 相机内参 [3, 3]
            
        Returns:
            epipolar_error: 极线距离 [N] 或 [H, W]
            valid_mask: 有效掩码 [N] 或 [H, W]
        """
        # 判断输入格式
        if pts1.dim() == 3:  # [2, H, W]
            _, H, W = pts1.shape
            pts1_flat = pts1.reshape(2, -1).T  # [H*W, 2]
            pts2_flat = pts2.reshape(2, -1).T  # [H*W, 2]
            reshape_output = True
        else:  # [N, 2]
            pts1_flat = pts1
            pts2_flat = pts2
            reshape_output = False
        
        device = pts1.device
        
        # 确保张量在正确的设备上
        if not isinstance(T_21, torch.Tensor):
            T_21 = torch.tensor(T_21, dtype=torch.float32, device=device)
        if not isinstance(K, torch.Tensor):
            K = torch.tensor(K, dtype=torch.float32, device=device)
        
        # 转换为齐次坐标
        ones = torch.ones(pts1_flat.shape[0], 1, device=device)
        pts1_homo = torch.cat([pts1_flat, ones], dim=1).T  # [3, N]
        pts2_homo = torch.cat([pts2_flat, ones], dim=1).T  # [3, N]
        
        # 使用现有的compute_epipolar_distance函数
        # 转换为numpy进行计算（如果需要）
        if isinstance(T_21, torch.Tensor):
            T_21_np = T_21.cpu().numpy()
        else:
            T_21_np = T_21
            
        if isinstance(K, torch.Tensor):
            K_np = K.cpu().numpy()
        else:
            K_np = K
        
        pts1_np = pts1_homo.cpu().numpy()
        pts2_np = pts2_homo.cpu().numpy()
        
        # 计算极线距离
        epipolar_error = compute_epipolar_distance(T_21_np, K_np, pts1_np, pts2_np)
        epipolar_error = torch.from_numpy(epipolar_error).to(device)
        
        # 生成有效掩码
        valid_mask = epipolar_error < self.epi_threshold
        
        if reshape_output:
            epipolar_error = epipolar_error.reshape(H, W)
            valid_mask = valid_mask.reshape(H, W)
        
        return epipolar_error, valid_mask
    
    def compute_geometric_consistency(
        self,
        flow_obs: torch.Tensor,
        flow_exp: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算几何一致性
        
        比较观测光流和预期光流的差异
        
        Args:
            flow_obs: 观测光流 [H, W, 2]
            flow_exp: 预期光流 [H, W, 2]
            valid_mask: 有效像素掩码 [H, W]
            
        Returns:
            consistency_score: 一致性分数 [H, W]，范围[0, 1]
            geo_mask: 几何一致性掩码 [H, W]
        """
        # 计算光流差异
        flow_diff = flow_obs - flow_exp
        geo_error = torch.norm(flow_diff, p=2, dim=-1)  # [H, W]
        
        # 应用有效掩码
        if valid_mask is not None:
            geo_error = geo_error * valid_mask.float()
        
        # 计算一致性分数（使用指数衰减）
        consistency_score = torch.exp(-geo_error / self.geo_threshold)
        
        # 生成几何一致性掩码
        geo_mask = geo_error < self.geo_threshold
        
        if valid_mask is not None:
            geo_mask = torch.logical_and(geo_mask, valid_mask)
        
        return consistency_score, geo_mask
    
    def multi_scale_validation(
        self,
        flow_obs: torch.Tensor,
        pose_t: torch.Tensor,
        pose_t1: torch.Tensor,
        depth: torch.Tensor,
        K: torch.Tensor,
        scales: list = [1.0, 0.5, 0.25]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        多尺度几何验证
        
        在不同尺度上验证几何一致性，提高鲁棒性
        
        Args:
            flow_obs: 观测光流 [H, W, 2]
            pose_t, pose_t1: 相机位姿
            depth: 深度图 [H, W]
            K: 相机内参 [3, 3]
            scales: 尺度列表
            
        Returns:
            consistency_score: 综合一致性分数 [H, W]
            geo_mask: 综合几何掩码 [H, W]
        """
        H, W = flow_obs.shape[:2]
        device = flow_obs.device
        
        consistency_scores = []
        geo_masks = []
        
        for scale in scales:
            if scale != 1.0:
                # 缩放图像和深度
                new_H, new_W = int(H * scale), int(W * scale)
                
                flow_scaled = F.interpolate(
                    flow_obs.permute(2, 0, 1).unsqueeze(0),
                    size=(new_H, new_W),
                    mode='bilinear',
                    align_corners=True
                ).squeeze(0).permute(1, 2, 0) * scale
                
                depth_scaled = F.interpolate(
                    depth.unsqueeze(0).unsqueeze(0),
                    size=(new_H, new_W),
                    mode='nearest'
                ).squeeze()
                
                # 调整内参
                K_scaled = K.clone()
                K_scaled[0, 0] *= scale  # fx
                K_scaled[1, 1] *= scale  # fy
                K_scaled[0, 2] *= scale  # cx
                K_scaled[1, 2] *= scale  # cy
            else:
                flow_scaled = flow_obs
                depth_scaled = depth
                K_scaled = K
            
            # 计算预期光流
            flow_exp, valid_mask = self.compute_expected_flow(
                pose_t, pose_t1, depth_scaled, K_scaled
            )
            
            # 计算几何一致性
            score, mask = self.compute_geometric_consistency(
                flow_scaled, flow_exp, valid_mask
            )
            
            # 恢复到原始尺度
            if scale != 1.0:
                score = F.interpolate(
                    score.unsqueeze(0).unsqueeze(0),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=True
                ).squeeze()
                
                mask = F.interpolate(
                    mask.float().unsqueeze(0).unsqueeze(0),
                    size=(H, W),
                    mode='nearest'
                ).squeeze().bool()
            
            consistency_scores.append(score)
            geo_masks.append(mask)
        
        # 融合多尺度结果（取平均）
        consistency_score = torch.stack(consistency_scores).mean(dim=0)
        geo_mask = torch.stack([m.float() for m in geo_masks]).mean(dim=0) > 0.5
        
        return consistency_score, geo_mask
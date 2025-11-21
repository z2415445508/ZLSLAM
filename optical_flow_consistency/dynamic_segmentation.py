"""
动态区域分割模块

实现动态区域的分割、精细化和后处理
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Tuple, Optional
from scipy import ndimage


class DynamicSegmentation:
    """动态区域分割与精细化"""
    
    def __init__(self, config: dict):
        """
        初始化动态分割器
        
        Args:
            config: 配置字典，包含以下参数:
                - min_region_size: 最小动态区域大小（像素）
                - morph_kernel_size: 形态学操作核大小
                - edge_threshold: 边缘优化阈值
                - use_crf: 是否使用条件随机场优化
        """
        self.min_region_size = config.get('min_region_size', 100)
        self.morph_kernel_size = config.get('morph_kernel_size', 5)
        self.edge_threshold = config.get('edge_threshold', 0.1)
        self.use_crf = config.get('use_crf', False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def segment_dynamic_regions(
        self,
        consistency_map: torch.Tensor,
        confidence_map: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于一致性图分割动态区域
        
        Args:
            consistency_map: 一致性图 [H, W]，值越小越可能是动态
            confidence_map: 置信度图 [H, W]（可选）
            threshold: 分割阈值
            
        Returns:
            dynamic_mask: 动态区域掩码 [H, W]
            region_labels: 区域标签 [H, W]
        """
        # 生成初始动态掩码
        if confidence_map is not None:
            # 综合考虑一致性和置信度
            dynamic_score = (1 - consistency_map) * confidence_map
            dynamic_mask = dynamic_score > threshold
        else:
            dynamic_mask = consistency_map < threshold
        
        # 转换为numpy进行连通域分析
        if isinstance(dynamic_mask, torch.Tensor):
            dynamic_mask_np = dynamic_mask.cpu().numpy().astype(np.uint8)
        else:
            dynamic_mask_np = dynamic_mask.astype(np.uint8)
        
        # 连通域标记
        region_labels, num_regions = ndimage.label(dynamic_mask_np)
        
        # 过滤小区域
        for i in range(1, num_regions + 1):
            region_size = np.sum(region_labels == i)
            if region_size < self.min_region_size:
                region_labels[region_labels == i] = 0
        
        # 更新动态掩码
        dynamic_mask_filtered = region_labels > 0
        
        # 转换回torch tensor
        dynamic_mask = torch.from_numpy(dynamic_mask_filtered).to(self.device)
        region_labels = torch.from_numpy(region_labels).to(self.device)
        
        return dynamic_mask, region_labels
    
    def refine_mask(
        self,
        mask: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        iterations: int = 2
    ) -> torch.Tensor:
        """
        精细化掩码
        
        应用形态学操作和边缘优化
        
        Args:
            mask: 输入掩码 [H, W]
            image: 原始图像 [3, H, W] 或 [H, W, 3]（可选，用于边缘引导）
            iterations: 形态学操作迭代次数
            
        Returns:
            refined_mask: 精细化后的掩码 [H, W]
        """
        # 转换为numpy
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy().astype(np.uint8)
        else:
            mask_np = mask.astype(np.uint8)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size)
        )
        
        # 闭运算：先膨胀后腐蚀，填充小孔
        mask_closed = cv2.morphologyEx(
            mask_np,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=iterations
        )
        
        # 开运算：先腐蚀后膨胀，去除小噪点
        mask_refined = cv2.morphologyEx(
            mask_closed,
            cv2.MORPH_OPEN,
            kernel,
            iterations=iterations
        )
        
        # 如果提供了图像，进行边缘引导优化
        if image is not None:
            mask_refined = self._edge_guided_refinement(mask_refined, image)
        
        # 转换回torch tensor
        refined_mask = torch.from_numpy(mask_refined.astype(bool)).to(self.device)
        
        return refined_mask
    
    def _edge_guided_refinement(
        self,
        mask: np.ndarray,
        image: torch.Tensor
    ) -> np.ndarray:
        """
        边缘引导的掩码优化
        
        利用图像边缘信息优化掩码边界
        
        Args:
            mask: 掩码 [H, W]
            image: 图像 [3, H, W] 或 [H, W, 3]
            
        Returns:
            refined_mask: 优化后的掩码 [H, W]
        """
        # 转换图像格式
        if isinstance(image, torch.Tensor):
            if image.shape[0] == 3:  # [3, H, W]
                image_np = image.permute(1, 2, 0).cpu().numpy()
            else:  # [H, W, 3]
                image_np = image.cpu().numpy()
        else:
            image_np = image
        
        # 转换为灰度图
        if image_np.shape[-1] == 3:
            gray = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image_np * 255).astype(np.uint8)
        
        # 计算图像梯度
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_magnitude = gradient_magnitude / (gradient_magnitude.max() + 1e-8)
        
        # 找到掩码边界
        mask_dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        mask_eroded = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        boundary = mask_dilated.astype(int) - mask_eroded.astype(int)
        
        # 在边界区域，如果梯度强，保持掩码；如果梯度弱，可能需要调整
        edge_strength = gradient_magnitude * boundary
        
        # 根据边缘强度调整掩码
        # 这里使用简单策略：强边缘保持，弱边缘收缩
        refined_mask = mask.copy()
        weak_edge = (edge_strength > 0) & (edge_strength < self.edge_threshold)
        refined_mask[weak_edge] = 0
        
        return refined_mask
    
    def merge_with_semantic(
        self,
        flow_mask: torch.Tensor,
        semantic_mask: torch.Tensor,
        weight: float = 0.5
    ) -> torch.Tensor:
        """
        融合光流掩码和语义分割掩码
        
        Args:
            flow_mask: 基于光流的动态掩码 [H, W]
            semantic_mask: 语义分割的动态掩码 [H, W]
            weight: 语义掩码的权重
            
        Returns:
            merged_mask: 融合后的掩码 [H, W]
        """
        # 转换为float进行加权
        flow_mask_float = flow_mask.float()
        semantic_mask_float = semantic_mask.float()
        
        # 加权融合
        merged = (1 - weight) * flow_mask_float + weight * semantic_mask_float
        
        # 阈值化
        merged_mask = merged > 0.5
        
        return merged_mask
    
    def temporal_filtering(
        self,
        mask_history: list,
        weights: Optional[list] = None,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        时序滤波
        
        通过多帧掩码的时序一致性过滤误检
        
        Args:
            mask_history: 掩码历史列表 [T, H, W]
            weights: 时序权重（可选）
            threshold: 时序一致性阈值
            
        Returns:
            filtered_mask: 时序滤波后的掩码 [H, W]
        """
        if len(mask_history) == 0:
            raise ValueError("掩码历史为空")
        
        # 转换为float
        masks_float = [m.float() for m in mask_history]
        
        # 默认权重：越近的帧权重越大
        if weights is None:
            weights = [1.0 / (len(masks_float) - i) for i in range(len(masks_float))]
            weights = [w / sum(weights) for w in weights]
        
        # 加权平均
        weighted_mask = sum(w * m for w, m in zip(weights, masks_float))
        
        # 阈值化
        filtered_mask = weighted_mask > threshold
        
        return filtered_mask
    
    def extract_motion_boundaries(
        self,
        mask: torch.Tensor,
        dilation_size: int = 3
    ) -> torch.Tensor:
        """
        提取运动边界
        
        Args:
            mask: 动态掩码 [H, W]
            dilation_size: 膨胀核大小
            
        Returns:
            boundary_mask: 边界掩码 [H, W]
        """
        # 转换为numpy
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy().astype(np.uint8)
        else:
            mask_np = mask.astype(np.uint8)
        
        # 膨胀
        kernel = np.ones((dilation_size, dilation_size), np.uint8)
        mask_dilated = cv2.dilate(mask_np, kernel, iterations=1)
        
        # 边界 = 膨胀 - 原始
        boundary = mask_dilated - mask_np
        
        # 转换回torch
        boundary_mask = torch.from_numpy(boundary.astype(bool)).to(self.device)
        
        return boundary_mask
    
    def compute_region_statistics(
        self,
        region_labels: torch.Tensor,
        flow: Optional[torch.Tensor] = None
    ) -> dict:
        """
        计算区域统计信息
        
        Args:
            region_labels: 区域标签 [H, W]
            flow: 光流场 [H, W, 2]（可选）
            
        Returns:
            statistics: 包含各区域统计信息的字典
        """
        statistics = {}
        
        if isinstance(region_labels, torch.Tensor):
            region_labels_np = region_labels.cpu().numpy()
        else:
            region_labels_np = region_labels
        
        num_regions = int(region_labels_np.max())
        
        for i in range(1, num_regions + 1):
            region_mask = region_labels_np == i
            region_size = np.sum(region_mask)
            
            # 计算区域中心
            y_coords, x_coords = np.where(region_mask)
            center_y = np.mean(y_coords)
            center_x = np.mean(x_coords)
            
            stats = {
                'size': region_size,
                'center': (center_x, center_y),
                'bbox': (x_coords.min(), y_coords.min(), 
                        x_coords.max(), y_coords.max())
            }
            
            # 如果提供了光流，计算平均运动
            if flow is not None:
                if isinstance(flow, torch.Tensor):
                    flow_np = flow.cpu().numpy()
                else:
                    flow_np = flow
                
                region_flow = flow_np[region_mask]
                stats['mean_flow'] = np.mean(region_flow, axis=0)
                stats['flow_std'] = np.std(region_flow, axis=0)
            
            statistics[i] = stats
        
        return statistics
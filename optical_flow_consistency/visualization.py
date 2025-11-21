"""
可视化工具模块

提供光流一致性检测结果的可视化功能
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Optional, Tuple
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flow_utils import flow_to_image


class FlowConsistencyVisualizer:
    """光流一致性可视化器"""
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        初始化可视化器
        
        Args:
            save_dir: 保存目录（可选）
        """
        self.save_dir = save_dir
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
    
    def visualize_detection_results(
        self,
        image: torch.Tensor,
        dynamic_mask: torch.Tensor,
        consistency_map: Optional[torch.Tensor] = None,
        flow: Optional[torch.Tensor] = None,
        frame_idx: Optional[int] = None,
        save: bool = True
    ) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 原始图像 [3, H, W] 或 [H, W, 3]
            dynamic_mask: 动态掩码 [H, W]
            consistency_map: 一致性图 [H, W]（可选）
            flow: 光流场 [H, W, 2]（可选）
            frame_idx: 帧索引（可选）
            save: 是否保存
            
        Returns:
            vis_image: 可视化图像
        """
        # 转换图像格式
        if isinstance(image, torch.Tensor):
            if image.shape[0] == 3:  # [3, H, W]
                image_np = image.permute(1, 2, 0).cpu().numpy()
            else:  # [H, W, 3]
                image_np = image.cpu().numpy()
        else:
            image_np = image
        
        # 确保图像在[0, 1]范围
        if image_np.max() > 1.0:
            image_np = image_np / 255.0
        
        # 转换掩码
        if isinstance(dynamic_mask, torch.Tensor):
            mask_np = dynamic_mask.cpu().numpy()
        else:
            mask_np = dynamic_mask
        
        # 创建子图
        n_plots = 2 + (consistency_map is not None) + (flow is not None)
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
        
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # 1. 原始图像
        axes[plot_idx].imshow(image_np)
        axes[plot_idx].set_title('Original Image')
        axes[plot_idx].axis('off')
        plot_idx += 1
        
        # 2. 动态掩码叠加
        overlay = image_np.copy()
        mask_colored = np.zeros_like(overlay)
        mask_colored[mask_np] = [1.0, 0.0, 0.0]  # 红色表示动态
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        axes[plot_idx].imshow(overlay)
        axes[plot_idx].set_title('Dynamic Regions (Red)')
        axes[plot_idx].axis('off')
        plot_idx += 1
        
        # 3. 一致性图（如果提供）
        if consistency_map is not None:
            if isinstance(consistency_map, torch.Tensor):
                consistency_np = consistency_map.cpu().numpy()
            else:
                consistency_np = consistency_map
            
            im = axes[plot_idx].imshow(consistency_np, cmap='jet', vmin=0, vmax=1)
            axes[plot_idx].set_title('Consistency Map')
            axes[plot_idx].axis('off')
            plt.colorbar(im, ax=axes[plot_idx], fraction=0.046)
            plot_idx += 1
        
        # 4. 光流可视化（如果提供）
        if flow is not None:
            if isinstance(flow, torch.Tensor):
                flow_np = flow.cpu().numpy()
            else:
                flow_np = flow
            
            flow_color = flow_to_image(flow_np)
            axes[plot_idx].imshow(flow_color)
            axes[plot_idx].set_title('Optical Flow')
            axes[plot_idx].axis('off')
            plot_idx += 1
        
        plt.tight_layout()
        
        # 保存
        if save and self.save_dir is not None:
            if frame_idx is not None:
                save_path = os.path.join(self.save_dir, f'detection_{frame_idx:06d}.png')
            else:
                save_path = os.path.join(self.save_dir, 'detection_result.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # 转换为numpy数组
        fig.canvas.draw()
        vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return vis_image
    
    def visualize_consistency_details(
        self,
        image: torch.Tensor,
        fb_error: Optional[torch.Tensor] = None,
        geo_score: Optional[torch.Tensor] = None,
        epi_error: Optional[torch.Tensor] = None,
        frame_idx: Optional[int] = None,
        save: bool = True
    ) -> np.ndarray:
        """
        可视化一致性检查的详细结果
        
        Args:
            image: 原始图像
            fb_error: 前后向误差
            geo_score: 几何一致性分数
            epi_error: 极线误差
            frame_idx: 帧索引
            save: 是否保存
            
        Returns:
            vis_image: 可视化图像
        """
        # 转换图像
        if isinstance(image, torch.Tensor):
            if image.shape[0] == 3:
                image_np = image.permute(1, 2, 0).cpu().numpy()
            else:
                image_np = image.cpu().numpy()
        else:
            image_np = image
        
        if image_np.max() > 1.0:
            image_np = image_np / 255.0
        
        # 计算子图数量
        n_plots = 1 + sum([x is not None for x in [fb_error, geo_score, epi_error]])
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
        
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # 原始图像
        axes[plot_idx].imshow(image_np)
        axes[plot_idx].set_title('Original Image')
        axes[plot_idx].axis('off')
        plot_idx += 1
        
        # 前后向误差
        if fb_error is not None:
            if isinstance(fb_error, torch.Tensor):
                fb_error_np = fb_error.cpu().numpy()
            else:
                fb_error_np = fb_error
            
            im = axes[plot_idx].imshow(fb_error_np, cmap='hot', vmin=0, vmax=5)
            axes[plot_idx].set_title('Forward-Backward Error')
            axes[plot_idx].axis('off')
            plt.colorbar(im, ax=axes[plot_idx], fraction=0.046)
            plot_idx += 1
        
        # 几何一致性分数
        if geo_score is not None:
            if isinstance(geo_score, torch.Tensor):
                geo_score_np = geo_score.cpu().numpy()
            else:
                geo_score_np = geo_score
            
            im = axes[plot_idx].imshow(geo_score_np, cmap='jet', vmin=0, vmax=1)
            axes[plot_idx].set_title('Geometric Consistency Score')
            axes[plot_idx].axis('off')
            plt.colorbar(im, ax=axes[plot_idx], fraction=0.046)
            plot_idx += 1
        
        # 极线误差
        if epi_error is not None:
            if isinstance(epi_error, torch.Tensor):
                epi_error_np = epi_error.cpu().numpy()
            else:
                epi_error_np = epi_error
            
            im = axes[plot_idx].imshow(epi_error_np, cmap='hot', vmin=0, vmax=3)
            axes[plot_idx].set_title('Epipolar Error')
            axes[plot_idx].axis('off')
            plt.colorbar(im, ax=axes[plot_idx], fraction=0.046)
            plot_idx += 1
        
        plt.tight_layout()
        
        # 保存
        if save and self.save_dir is not None:
            if frame_idx is not None:
                save_path = os.path.join(self.save_dir, f'consistency_details_{frame_idx:06d}.png')
            else:
                save_path = os.path.join(self.save_dir, 'consistency_details.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # 转换为numpy数组
        fig.canvas.draw()
        vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return vis_image
    
    def visualize_region_statistics(
        self,
        image: torch.Tensor,
        region_labels: torch.Tensor,
        statistics: dict,
        frame_idx: Optional[int] = None,
        save: bool = True
    ) -> np.ndarray:
        """
        可视化区域统计信息
        
        Args:
            image: 原始图像
            region_labels: 区域标签
            statistics: 区域统计信息
            frame_idx: 帧索引
            save: 是否保存
            
        Returns:
            vis_image: 可视化图像
        """
        # 转换图像
        if isinstance(image, torch.Tensor):
            if image.shape[0] == 3:
                image_np = image.permute(1, 2, 0).cpu().numpy()
            else:
                image_np = image.cpu().numpy()
        else:
            image_np = image
        
        if image_np.max() > 1.0:
            image_np = image_np / 255.0
        
        # 转换标签
        if isinstance(region_labels, torch.Tensor):
            labels_np = region_labels.cpu().numpy()
        else:
            labels_np = region_labels
        
        # 创建彩色标签图
        num_regions = int(labels_np.max())
        colored_labels = np.zeros((*labels_np.shape, 3))
        
        # 为每个区域分配随机颜色
        np.random.seed(42)
        colors = np.random.rand(num_regions + 1, 3)
        colors[0] = [0, 0, 0]  # 背景为黑色
        
        for i in range(num_regions + 1):
            colored_labels[labels_np == i] = colors[i]
        
        # 创建可视化
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 原始图像叠加区域
        overlay = image_np * 0.5 + colored_labels * 0.5
        axes[0].imshow(overlay)
        axes[0].set_title('Detected Dynamic Regions')
        axes[0].axis('off')
        
        # 在图像上标注区域信息
        axes[1].imshow(image_np)
        axes[1].set_title('Region Statistics')
        axes[1].axis('off')
        
        for region_id, stats in statistics.items():
            center = stats['center']
            size = stats['size']
            
            # 绘制区域中心
            axes[1].plot(center[0], center[1], 'r*', markersize=15)
            
            # 标注区域信息
            text = f"R{region_id}\nSize: {size}"
            if 'mean_flow' in stats:
                flow = stats['mean_flow']
                text += f"\nFlow: ({flow[0]:.1f}, {flow[1]:.1f})"
            
            axes[1].text(center[0], center[1], text, 
                        color='yellow', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            # 绘制边界框
            if 'bbox' in stats:
                bbox = stats['bbox']
                rect = plt.Rectangle((bbox[0], bbox[1]), 
                                    bbox[2] - bbox[0], 
                                    bbox[3] - bbox[1],
                                    fill=False, edgecolor='red', linewidth=2)
                axes[1].add_patch(rect)
        
        plt.tight_layout()
        
        # 保存
        if save and self.save_dir is not None:
            if frame_idx is not None:
                save_path = os.path.join(self.save_dir, f'region_stats_{frame_idx:06d}.png')
            else:
                save_path = os.path.join(self.save_dir, 'region_statistics.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # 转换为numpy数组
        fig.canvas.draw()
        vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return vis_image
    
    def create_comparison_video(
        self,
        image_dir: str,
        mask_dir: str,
        output_path: str,
        fps: int = 10
    ):
        """
        创建对比视频
        
        Args:
            image_dir: 原始图像目录
            mask_dir: 掩码图像目录
            output_path: 输出视频路径
            fps: 帧率
        """
        import glob
        
        # 获取图像列表
        image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
        
        if len(image_files) == 0 or len(mask_files) == 0:
            print("未找到图像文件")
            return
        
        # 读取第一帧确定尺寸
        first_img = cv2.imread(image_files[0])
        h, w = first_img.shape[:2]
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w*2, h))
        
        # 处理每一帧
        for img_file, mask_file in zip(image_files, mask_files):
            img = cv2.imread(img_file)
            mask = cv2.imread(mask_file)
            
            # 水平拼接
            combined = np.hstack([img, mask])
            
            out.write(combined)
        
        out.release()
        print(f"视频已保存到: {output_path}")
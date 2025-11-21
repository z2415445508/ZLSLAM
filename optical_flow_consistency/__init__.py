"""
光流一致性动态区域检测模块

该模块实现基于光流一致性的动态区域精细化检测，用于提升4DGS-SLAM系统
在动态场景下的定位与建图性能。

主要组件:
- FlowConsistencyDetector: 主检测器
- ConsistencyChecker: 一致性检查
- GeometricValidator: 几何验证
- DynamicSegmentation: 动态区域分割
"""

from .flow_consistency_detector import FlowConsistencyDetector
from .consistency_checker import ConsistencyChecker
from .geometric_validator import GeometricValidator
from .dynamic_segmentation import DynamicSegmentation

__version__ = "1.0.0"
__author__ = "4DGS-SLAM Team"

__all__ = [
    'FlowConsistencyDetector',
    'ConsistencyChecker',
    'GeometricValidator',
    'DynamicSegmentation',
]
# 光流一致性动态区域检测 - 使用指南

## 概述

本模块实现了基于光流一致性的动态区域精细化检测，用于提升4DGS-SLAM系统在动态场景下的定位与建图性能。

## 功能特点

- ✅ **前后向光流一致性检查**: 检测光流的双向一致性
- ✅ **几何一致性验证**: 基于相机运动和深度的几何约束
- ✅ **极线约束验证**: 利用对极几何进行验证（可选）
- ✅ **时序一致性分析**: 多帧光流的时序分析
- ✅ **动态区域分割**: 精细化的动态区域提取
- ✅ **多尺度验证**: 提高检测鲁棒性（可选）

## 安装依赖

确保已安装以下依赖：

```bash
pip install torch torchvision
pip install opencv-python
pip install scipy
pip install matplotlib
pip install pyyaml
```

## 快速开始

### 1. 下载RAFT预训练模型

```bash
# 创建预训练模型目录
mkdir -p pretrained

# 下载RAFT模型（需要从官方仓库获取）
# 将模型放置在: pretrained/raft-things.pth
```

### 2. 配置启用

在配置文件中启用光流一致性检测（例如 `configs/rgbd/bonn/base_config.yaml`）：

```yaml
FlowConsistency:
  enabled: true                # 启用检测
  flow_model: "RAFT"           # 使用RAFT模型
  fb_threshold: 1.0            # 前后向一致性阈值（像素）
  geo_threshold: 2.0           # 几何一致性阈值（像素）
  use_fb_check: true           # 启用前后向检查
  use_geo_check: true          # 启用几何检查
  use_temporal: true           # 启用时序一致性
```

### 3. 运行SLAM系统

```bash
# 运行动态场景SLAM
python slam.py --config configs/rgbd/bonn/ballon.yaml --dynamic

# 评估模式
python slam.py --config configs/rgbd/bonn/ballon.yaml --dynamic --eval
```

## 配置参数详解

### 基本设置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | false | 是否启用光流一致性检测 |
| `flow_model` | str | "RAFT" | 光流模型类型（RAFT/GMA） |

### 阈值参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `fb_threshold` | float | 1.0 | 前后向一致性阈值（像素） |
| `geo_threshold` | float | 2.0 | 几何一致性阈值（像素） |
| `epi_threshold` | float | 1.0 | 极线距离阈值（像素） |
| `min_depth` | float | 0.1 | 最小有效深度（米） |
| `max_depth` | float | 10.0 | 最大有效深度（米） |

### 检查开关

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_fb_check` | bool | true | 使用前后向一致性检查 |
| `use_geo_check` | bool | true | 使用几何一致性检查 |
| `use_epi_check` | bool | false | 使用极线约束检查（计算量大） |
| `use_temporal` | bool | true | 使用时序一致性 |

### 融合权重

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `fb_weight` | float | 0.4 | 前后向一致性权重 |
| `geo_weight` | float | 0.4 | 几何一致性权重 |
| `epi_weight` | float | 0.2 | 极线一致性权重 |

### 分割参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `min_region_size` | int | 100 | 最小动态区域大小（像素） |
| `morph_kernel_size` | int | 5 | 形态学操作核大小 |
| `edge_threshold` | float | 0.1 | 边缘优化阈值 |

### 时序参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `temporal_window` | int | 3 | 时序一致性窗口大小 |

### 可视化设置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `save_visualization` | bool | false | 是否保存可视化结果 |
| `vis_interval` | int | 10 | 可视化间隔（帧） |

## 使用示例

### 示例1：基本使用

```python
from optical_flow_consistency import FlowConsistencyDetector
from RAFT.raft import RAFT
import torch

# 初始化光流模型
flow_model = RAFT(args)
flow_model.load_state_dict(torch.load('pretrained/raft-things.pth'))
flow_model = flow_model.cuda().eval()

# 初始化检测器
config = {
    'fb_threshold': 1.0,
    'geo_threshold': 2.0,
    'use_fb_check': True,
    'use_geo_check': True,
}
detector = FlowConsistencyDetector(config, flow_model)

# 检测动态区域
results = detector.detect_dynamic_regions(
    frame_t=image_t,
    frame_t1=image_t1,
    pose_t=pose_t,
    pose_t1=pose_t1,
    depth_t=depth_t,
    K=camera_intrinsics
)

dynamic_mask = results['dynamic_mask']
```

### 示例2：可视化结果

```python
from optical_flow_consistency.visualization import FlowConsistencyVisualizer

# 初始化可视化器
visualizer = FlowConsistencyVisualizer(save_dir='results/visualization')

# 可视化检测结果
visualizer.visualize_detection_results(
    image=image_t,
    dynamic_mask=results['dynamic_mask'],
    consistency_map=results['consistency_map'],
    flow=results['flow_fwd'],
    frame_idx=frame_idx,
    save=True
)
```

## 性能优化建议

### 1. 计算效率

- **关闭极线检查**: 如果性能要求高，设置 `use_epi_check: false`
- **减少时序窗口**: 降低 `temporal_window` 值（如设为2）
- **降低可视化频率**: 增大 `vis_interval` 值

### 2. 检测精度

- **调整阈值**: 根据场景特点调整 `fb_threshold` 和 `geo_threshold`
- **启用时序一致性**: 设置 `use_temporal: true` 减少误检
- **增大最小区域**: 提高 `min_region_size` 过滤小噪点

### 3. 场景适配

**室内场景**:
```yaml
fb_threshold: 0.8
geo_threshold: 1.5
min_region_size: 150
```

**室外场景**:
```yaml
fb_threshold: 1.5
geo_threshold: 3.0
min_region_size: 200
```

**快速运动**:
```yaml
fb_threshold: 2.0
geo_threshold: 3.5
use_temporal: true
temporal_window: 5
```

## 测试数据集

推荐使用以下数据集测试：

1. **TUM RGB-D Dynamic**: 包含多种动态场景
2. **Bonn RGB-D Dynamic**: 专门的动态物体数据集
3. **CoFusion Dataset**: 复杂动态场景

## 故障排除

### 问题1: 找不到RAFT模型

**错误**: `警告: 未找到RAFT模型 pretrained/raft-things.pth`

**解决方案**:
1. 从RAFT官方仓库下载预训练模型
2. 放置在 `pretrained/raft-things.pth`
3. 或修改配置指定模型路径

### 问题2: 内存不足

**错误**: `CUDA out of memory`

**解决方案**:
1. 关闭时序一致性: `use_temporal: false`
2. 关闭极线检查: `use_epi_check: false`
3. 减少时序窗口: `temporal_window: 2`

### 问题3: 检测效果不佳

**可能原因**:
- 阈值设置不合适
- 光流质量差
- 深度图噪声大

**解决方案**:
1. 根据场景调整阈值
2. 启用时序一致性
3. 增大最小区域大小

## 技术细节

### 算法原理

1. **前后向一致性**: 
   ```
   E_fb = ||F_fwd(p) + F_bwd(p + F_fwd(p))||
   ```

2. **几何一致性**:
   ```
   E_geo = ||F_obs(p) - F_exp(p)||
   其中 F_exp 基于相机运动和深度计算
   ```

3. **极线约束**:
   ```
   d_epi = |p2^T * F * p1| / sqrt((Fp1)_x^2 + (Fp1)_y^2)
   ```

### 模块架构

```
optical_flow_consistency/
├── __init__.py                      # 模块初始化
├── flow_consistency_detector.py    # 主检测器
├── consistency_checker.py          # 一致性检查
├── geometric_validator.py          # 几何验证
├── dynamic_segmentation.py         # 动态分割
└── visualization.py                # 可视化工具
```

## 引用

如果您在研究中使用了本模块，请引用：

```bibtex
@article{4dgs-slam-flow-consistency,
  title={Optical Flow Consistency for Dynamic Scene Detection in 4DGS-SLAM},
  author={Your Name},
  year={2025}
}
```

## 联系方式

如有问题或建议，请联系：
- Email: your.email@example.com
- GitHub Issues: [项目地址]

## 更新日志

### v1.0.0 (2025-11-21)
- ✅ 初始版本发布
- ✅ 实现前后向一致性检查
- ✅ 实现几何一致性验证
- ✅ 实现动态区域分割
- ✅ 集成到4DGS-SLAM系统
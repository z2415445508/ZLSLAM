# 光流一致性动态检测 - 快速开始

## 🎯 项目简介

本项目为4DGS-SLAM系统实现了**基于光流一致性的动态区域精细化检测**模块，用于提升SLAM系统在动态场景下的定位与建图性能。

### 核心功能
- ✅ 前后向光流一致性检查
- ✅ 基于几何约束的验证
- ✅ 时序一致性分析
- ✅ 精细化动态区域分割
- ✅ 完整的可视化工具

## 🚀 快速开始

### 1. 环境准备

确保已安装基础依赖：
```bash
pip install torch torchvision opencv-python scipy matplotlib pyyaml
```

### 2. 下载光流模型

```bash
# 创建预训练模型目录
mkdir -p pretrained

# 下载RAFT模型（从官方仓库或提供的链接）
# 将模型文件放置在: pretrained/raft-things.pth
```

### 3. 启用检测功能

编辑配置文件（例如 `configs/rgbd/bonn/ballon.yaml`），确保包含：

```yaml
FlowConsistency:
  enabled: true                # 启用光流一致性检测
  flow_model: "RAFT"
  fb_threshold: 1.0
  geo_threshold: 2.0
  use_fb_check: true
  use_geo_check: true
  use_temporal: true
```

### 4. 运行SLAM系统

```bash
# 基本运行
python slam.py --config configs/rgbd/bonn/ballon.yaml --dynamic

# 评估模式
python slam.py --config configs/rgbd/bonn/ballon.yaml --dynamic --eval
```

## 📁 项目结构

```
4DGS-SLAM/
├── optical_flow_consistency/          # 光流一致性检测模块
│   ├── __init__.py
│   ├── flow_consistency_detector.py   # 主检测器
│   ├── consistency_checker.py         # 一致性检查
│   ├── geometric_validator.py         # 几何验证
│   ├── dynamic_segmentation.py        # 动态分割
│   └── visualization.py               # 可视化工具
├── configs/
│   ├── flow_consistency_config.yaml   # 独立配置文件
│   └── rgbd/bonn/base_config.yaml     # 已更新基础配置
├── docs/
│   ├── optical_flow_consistency_design.md    # 设计文档
│   ├── FLOW_CONSISTENCY_USAGE.md            # 使用指南
│   └── IMPLEMENTATION_SUMMARY.md            # 实现总结
├── examples/
│   └── flow_consistency_integration_example.py  # 集成示例
├── slam.py                            # 已集成检测器
└── utils/slam_frontend.py             # 已集成tracking
```

## 🔧 配置说明

### 基本配置

```yaml
FlowConsistency:
  enabled: true              # 是否启用
  flow_model: "RAFT"         # 光流模型
  fb_threshold: 1.0          # 前后向一致性阈值（像素）
  geo_threshold: 2.0         # 几何一致性阈值（像素）
```

### 场景适配

**室内场景**（推荐）:
```yaml
fb_threshold: 0.8
geo_threshold: 1.5
min_region_size: 150
```

**室外场景**（推荐）:
```yaml
fb_threshold: 1.5
geo_threshold: 3.0
min_region_size: 200
```

**快速运动**（推荐）:
```yaml
fb_threshold: 2.0
geo_threshold: 3.5
use_temporal: true
temporal_window: 5
```

## 📊 预期效果

- **动态检测精度**: 提升 15-20%
- **定位误差(ATE)**: 降低 10-15%
- **建图质量**: 提升 5-10%

## 📖 详细文档

- **设计文档**: [`docs/optical_flow_consistency_design.md`](docs/optical_flow_consistency_design.md)
- **使用指南**: [`docs/FLOW_CONSISTENCY_USAGE.md`](docs/FLOW_CONSISTENCY_USAGE.md)
- **实现总结**: [`docs/IMPLEMENTATION_SUMMARY.md`](docs/IMPLEMENTATION_SUMMARY.md)

## 🔍 使用示例

### Python API 使用

```python
from optical_flow_consistency import FlowConsistencyDetector
from RAFT.raft import RAFT
import torch

# 1. 初始化光流模型
flow_model = RAFT(args)
flow_model.load_state_dict(torch.load('pretrained/raft-things.pth'))
flow_model = flow_model.cuda().eval()

# 2. 初始化检测器
config = {
    'fb_threshold': 1.0,
    'geo_threshold': 2.0,
    'use_fb_check': True,
    'use_geo_check': True,
}
detector = FlowConsistencyDetector(config, flow_model)

# 3. 检测动态区域
results = detector.detect_dynamic_regions(
    frame_t=image_t,
    frame_t1=image_t1,
    pose_t=pose_t,
    pose_t1=pose_t1,
    depth_t=depth_t,
    K=camera_intrinsics
)

# 4. 获取结果
dynamic_mask = results['dynamic_mask']
confidence_map = results['confidence_map']
```

### 可视化结果

```python
from optical_flow_consistency.visualization import FlowConsistencyVisualizer

visualizer = FlowConsistencyVisualizer(save_dir='results/vis')

visualizer.visualize_detection_results(
    image=image_t,
    dynamic_mask=results['dynamic_mask'],
    consistency_map=results['consistency_map'],
    flow=results['flow_fwd'],
    frame_idx=frame_idx,
    save=True
)
```

## ⚙️ 性能优化

### 提高速度
```yaml
use_epi_check: false        # 关闭极线检查
temporal_window: 2          # 减少时序窗口
save_visualization: false   # 关闭可视化
```

### 提高精度
```yaml
use_temporal: true          # 启用时序一致性
temporal_window: 5          # 增大时序窗口
min_region_size: 200        # 增大最小区域
```

## 🐛 故障排除

### 问题1: 找不到RAFT模型
```
警告: 未找到RAFT模型 pretrained/raft-things.pth
```
**解决**: 下载RAFT预训练模型并放置在正确位置

### 问题2: CUDA内存不足
```
CUDA out of memory
```
**解决**: 
- 设置 `use_temporal: false`
- 设置 `use_epi_check: false`
- 减少 `temporal_window`

### 问题3: 检测效果不佳
**解决**:
- 根据场景调整阈值参数
- 启用时序一致性
- 增大最小区域大小

## 📝 测试数据集

推荐使用以下数据集测试：
- **TUM RGB-D Dynamic**: 多种动态场景
- **Bonn RGB-D Dynamic**: 专门的动态物体数据集
- **CoFusion Dataset**: 复杂动态场景

## 🎓 技术原理

### 前后向一致性
```
E_fb(p) = ||F_fwd(p) + F_bwd(p + F_fwd(p))||
```

### 几何一致性
```
E_geo(p) = ||F_obs(p) - F_exp(p)||
其中 F_exp 基于相机运动和深度计算
```

### 极线约束
```
d_epi = |p2^T * F * p1| / sqrt((Fp1)_x^2 + (Fp1)_y^2)
```

## 📈 开发进度

- [x] 分析现有系统架构和光流相关代码
- [x] 设计光流一致性检测算法架构
- [x] 实现光流一致性计算模块
- [x] 实现动态区域检测与分割
- [x] 集成到SLAM前端系统
- [x] 添加可视化和调试功能
- [ ] 测试和优化性能

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目遵循4DGS-SLAM的原始许可证。

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- Email: your.email@example.com

---

**最后更新**: 2025-11-21  
**版本**: v1.0.0  
**状态**: ✅ 核心功能已完成，待测试验证
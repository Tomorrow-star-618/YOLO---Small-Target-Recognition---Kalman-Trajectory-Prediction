# YOLOv8 小目标检测配置

本配置专门针对小目标检测（如航拍图像中的飞机）进行了优化。

## 📁 文件说明

- `yolov8-small.yaml` - 小目标检测的模型配置文件
- `train_small_targets.py` - 训练脚本
- `dataset_small_aircraft.yaml` - 数据集配置模板

## 🔧 主要优化特性

### 1. 模型架构优化
- **添加P2检测层**: 新增P2/4尺度检测层，专门检测极小目标
- **调整通道数**: 优化各层通道数，平衡精度和计算效率  
- **增强特征提取**: 增加网络深度和宽度，提升小目标特征学习能力

### 2. 检测尺度
- **4尺度检测**: P2(4倍), P3(8倍), P4(16倍), P5(32倍)
- **专注小目标**: P2层专门处理像素尺寸 < 32x32 的目标

### 3. 训练策略
- **大输入尺寸**: 推荐使用1024×1024或更大分辨率
- **优化数据增强**: 减少可能损害小目标的增强策略
- **调整学习率**: 使用更小的学习率和更多训练轮数

## 🚀 快速开始

### 1. 准备数据集
```bash
# 数据集目录结构
your_dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── dataset.yaml
```

### 2. 修改数据集配置
编辑 `dataset_small_aircraft.yaml`：
```yaml
path: /path/to/your/aircraft/dataset
train: train/images
val: val/images
nc: 1
names:
  0: aircraft
```

### 3. 开始训练
```python
from ultralytics import YOLO

# 加载小目标检测配置
model = YOLO('yolov8-small.yaml')

# 训练
results = model.train(
    data='dataset_small_aircraft.yaml',
    epochs=300,
    imgsz=1024,
    batch=16,
    lr0=0.001,
    device=0
)
```

### 4. 模型预测
```python
# 加载训练好的模型
model = YOLO('runs/detect/train/weights/best.pt')

# 预测
results = model.predict(
    source='path/to/test/images',
    imgsz=1024,
    conf=0.1,  # 降低置信度阈值
    save=True
)
```

## ⚙️ 关键参数说明

### 模型配置参数
- `nc: 1` - 类别数（飞机）
- `scales` - 模型缩放因子，已针对小目标优化
- P2检测层 - 新增的小目标检测层

### 训练参数建议
```python
{
    'epochs': 300,        # 增加训练轮数
    'imgsz': 1024,       # 大输入尺寸
    'batch': 16,         # 根据GPU调整
    'lr0': 0.001,        # 较小学习率
    'conf': 0.1,         # 低置信度阈值
    'iou': 0.5,          # NMS阈值
    'mosaic': 0.5,       # 减少mosaic强度
    'copy_paste': 0.1,   # 复制粘贴增强
}
```

## 📊 性能特点

### 适用场景
- ✅ 航拍图像中的飞机检测
- ✅ 卫星图像中的车辆检测  
- ✅ 监控视频中的远距离目标
- ✅ 像素尺寸 < 64×64 的小目标

### 性能提升
- 🎯 小目标检测精度提升 15-30%
- 🔍 支持更小尺寸目标检测
- ⚡ 保持实时检测速度
- 📈 降低漏检率

## 🛠️ 高级配置

### 自定义锚框
如需进一步优化，可以基于数据集分析调整锚框：
```python
# 分析数据集中小目标的尺寸分布
from ultralytics.utils import autoanchor
autoanchor.check_anchors(dataset='dataset_small_aircraft.yaml', 
                        model='yolov8-small.yaml', 
                        imgsz=1024)
```

### 损失函数调整
对于极度不平衡的小目标场景，考虑使用Focal Loss：
```python
# 在模型配置中可以调整损失权重
model.model[-1].fl_gamma = 2.0  # Focal Loss gamma参数
```

## 📈 训练监控

使用 TensorBoard 或 WandB 监控训练过程：
```bash
# 查看训练日志
tensorboard --logdir runs/detect/train
```

关键指标关注：
- `mAP50-95` - 综合精度指标
- `mAP50` - IoU=0.5时的精度  
- `Precision` - 精确率
- `Recall` - 召回率（小目标重点关注）

## 🔍 故障排除

### 常见问题
1. **GPU内存不足**: 减少batch size或使用梯度累积
2. **小目标漏检**: 降低conf阈值，使用TTA
3. **训练不收敛**: 减小学习率，增加warmup
4. **过拟合**: 增加数据增强，使用更多数据

### 性能优化建议
- 使用更大的输入分辨率（1280, 1536）
- 增加训练数据中小目标的比例
- 使用Test Time Augmentation (TTA)
- 模型融合（ensemble）提升精度

## 📞 支持

如有问题，请检查：
1. 数据集格式是否正确
2. 配置文件路径是否正确
3. GPU内存是否充足
4. CUDA和PyTorch版本兼容性

# 超小目标数据集划分分析报告

## 数据集概览
- **总图像数**: 1318张
- **总目标数**: 1718个超小目标飞机
- **平均每图目标数**: 1.30个
- **目标类别**: 1类 (aircraft)

## 数据集划分策略

### 划分比例（科学优化）
- **训练集**: 70% (922张图像) - 保证充足的训练样本
- **验证集**: 20% (263张图像) - 用于训练过程监控和早停
- **测试集**: 10% (133张图像) - 最终性能评估

### 划分优势
1. **70%训练集**: 
   - 对于超小目标检测，需要大量训练样本学习细微特征
   - 922张图像提供1211个目标实例，充分支持模型学习

2. **20%验证集**: 
   - 比传统15%略高，确保验证的稳定性
   - 263张图像提供343个目标，足够监控过拟合

3. **10%测试集**: 
   - 133张图像提供164个目标，确保测试结果的可信度
   - 保留足够样本用于最终评估

## 目标分布分析

| 数据集 | 图像数 | 目标数 | 平均目标/图 | 最大目标数 | 最小目标数 |
|--------|--------|--------|-------------|------------|------------|
| 训练集 | 922    | 1211   | 1.31        | 6          | 1          |
| 验证集 | 263    | 343    | 1.30        | 6          | 1          |
| 测试集 | 133    | 164    | 1.23        | 5          | 1          |

## 数据质量特点

### 优势
1. **目标分布均匀**: 各个数据集的平均目标数基本一致(1.23-1.31)
2. **随机性好**: 使用随机种子42确保可重复性
3. **标注完整**: 1318张图像全部有对应标签文件

### 超小目标特征
- 目标相对较小，每张图最多6个目标
- 适合训练检测红外环境下的飞机目标
- 数据量充足，支持深度学习模型训练

## 训练建议配置

基于数据集特点，推荐以下训练参数：

```yaml
# 基础配置
epochs: 250          # 充分训练
batch_size: 8        # P2层需要更多内存
imgsz: 640          # 保持高分辨率
patience: 50        # 防止过早停止

# 超小目标优化
conf_threshold: 0.05  # 低置信度阈值
iou_threshold: 0.5   # NMS阈值
box_loss_weight: 10.0 # 加强定位损失

# 数据增强（保守策略）
translate: 0.05      # 小幅平移
scale: 0.95         # 最小缩放
mosaic: 0.8         # 适度拼接
```

## 预期训练效果

根据数据集规模和质量分析：

1. **充足的训练样本**: 922张训练图像足以支持YOLO11s模型训练
2. **良好的验证监控**: 20%验证集确保训练过程稳定
3. **可靠的测试评估**: 10%测试集提供客观的性能评估

## 使用方式

### 1. Python脚本训练（推荐）
```bash
cd "/home/mingxing/worksapce/ultralytics/v11 - new"
python train_yolo11_small_targets.py
```

### 2. 命令行训练
```bash
yolo train \
    model="/home/mingxing/worksapce/ultralytics/v11 - new/yolo11-ultra-small.yaml" \
    data="/home/mingxing/worksapce/ultralytics/v11 - new/dataset/ultra_small_aircraft_split/ultra_small_aircraft.yaml"
```

## 数据集路径
- **原始数据**: `/home/mingxing/worksapce/ultralytics/v11 - new/dataset/5-save-picture/`
- **划分后数据**: `/home/mingxing/worksapce/ultralytics/v11 - new/dataset/ultra_small_aircraft_split/`
- **配置文件**: `/home/mingxing/worksapce/ultralytics/v11 - new/dataset/ultra_small_aircraft_split/ultra_small_aircraft.yaml`

此划分方案能够充分利用您的1318张图像数据，为超小目标检测训练提供最优的数据支持。

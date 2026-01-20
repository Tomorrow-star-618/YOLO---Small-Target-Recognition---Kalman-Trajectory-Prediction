# YOLO11 超小目标检测优化说明

## 主要改进点

### 1. 检测头架构改进
- **原始YOLO11**: P3/8, P4/16, P5/32 三层检测头
- **超小目标版本**: **P2/4, P3/8, P4/16, P5/32 四层检测头**

### 2. P2层的优势
- **更高分辨率**: 4倍下采样 vs 8倍下采样
- **保留更多细节**: 适合检测像素级别的超小目标
- **更密集的特征**: 提高超小目标的召回率

### 3. 训练参数优化

#### 批次大小调整
- 从 `batch=16` 降至 `batch=8`
- 原因：P2层需要更多GPU内存

#### 数据增强优化
- `translate=0.05` (减小平移，避免超小目标移出视野)
- `scale=0.95` (极小缩放，避免超小目标消失)
- `mosaic=0.8` (适度减少，避免超小目标过度分割)
- `copy_paste=0.2` (增加复制粘贴，增加超小目标样本)

#### 损失函数权重
- `box=10.0` (大幅增加边界框损失，超小目标定位更重要)
- `cls=0.3` (适度降低分类损失)
- `dfl=2.0` (增加DFL损失，更精确的边界框)

## 训练方式选择

### 方式1：Python脚本（推荐）
```bash
cd "/home/mingxing/worksapce/ultralytics/v11 - new"
python train_yolo11_small_targets.py
```

### 方式2：Shell脚本
```bash
cd "/home/mingxing/worksapce/ultralytics/v11 - new"
chmod +x train_ultra_small.sh
./train_ultra_small.sh
```

### 方式3：直接命令行
```bash
yolo train model="/home/mingxing/worksapce/ultralytics/v11 - new/yolo11-ultra-small.yaml" \
           data=/home/mingxing/worksapce/ultralytics/yolo_dataset/dataset.yaml \
           epochs=250 batch=8 imgsz=640 \
           project="/home/mingxing/worksapce/ultralytics/v11 - new/train" \
           name="yolo11_ultra_small_aircraft"
```

## 预期效果

1. **提高召回率**: P2层能检测到更多超小目标
2. **减少漏检**: 高分辨率特征保留更多细节
3. **更精确定位**: 优化的损失函数权重
4. **适应性更强**: 针对红外图像的数据增强策略

## 注意事项

1. **内存需求**: P2层会增加GPU内存使用，如遇到OOM错误，可进一步减小batch_size
2. **训练时间**: 四层检测头会增加训练时间，但效果更好
3. **推理速度**: 会稍微降低推理速度，但对于精度要求高的应用是值得的

## 文件位置

- 配置文件: `/home/mingxing/worksapce/ultralytics/v11 - new/yolo11-ultra-small.yaml`
- 训练脚本: `/home/mingxing/worksapce/ultralytics/v11 - new/train_yolo11_small_targets.py`
- 训练结果: `/home/mingxing/worksapce/ultralytics/v11 - new/train/yolo11_ultra_small_aircraft/`

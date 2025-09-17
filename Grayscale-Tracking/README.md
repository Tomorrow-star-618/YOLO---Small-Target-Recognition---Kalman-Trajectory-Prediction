# 基于灰度值的目标追踪系统

## 📖 系统概述

这是一个结合YOLO目标检测和局部灰度值预测的目标追踪系统。当YOLO检测丢失目标超过5帧时，系统会自动切换到基于灰度相似性的预测模式，实现连续追踪。

## 🎯 核心功能

### 1. 双模式追踪
- **YOLO检测模式**: 使用训练好的小目标检测模型进行精确检测
- **灰度预测模式**: 当检测丢失时，基于局部灰度模板进行位置预测

### 2. 智能切换机制
- 目标连续检测时使用YOLO结果（绿色框显示）
- 目标丢失5帧以上时自动切换到灰度预测（红色框显示）
- 重新检测到目标时自动切回YOLO模式

### 3. 自适应搜索
- 在上一帧位置周围进行局部搜索
- 基于模板匹配和相似性评分确定最佳位置
- 动态调整搜索半径和相似度阈值

## 🚀 使用方法

### 基本用法
```bash
# 最简单的用法 - 仅指定视频文件
python grayscale_tracking_system.py -v /path/to/your/video.mp4

# 指定输出路径
python grayscale_tracking_system.py -v input_video.mp4 -o output-vedio/tracked_result.mp4

# 使用自定义模型
python grayscale_tracking_system.py -m custom_model.pt -v video.mp4
```

### 使用灰度模板
```bash
# 使用预定义的灰度模板（25x25数组）
python grayscale_tracking_system.py -v video.mp4 -t "[[100,105,110,...],[95,100,105,...],...]"
```

## 📁 文件结构
```
Grayscale-Tracking/
├── grayscale_tracking_system.py    # 主追踪系统
├── run_example.py                  # 使用示例和模板生成
├── README.md                       # 说明文档
├── output-vedio/                   # 输出视频目录
│   └── tracked_*.mp4               # 追踪结果视频
└── sample_template.npy             # 示例灰度模板
```

## ⚙️ 系统参数

### 可调节参数
- `max_lost_frames = 5`: 最大丢失帧数（超过后删除轨迹）
- `roi_size = 25`: ROI区域大小（与灰度模板匹配）
- `search_radius = 50`: 搜索半径（像素）
- 相似度阈值: 0.3（低于此值认为匹配失败）

### 模型要求
- 默认使用: `small_target_detection/yolov8_small_aircraft/weights/best.pt`
- 支持所有YOLOv8兼容的模型格式
- 推荐使用小目标优化的模型

## 🎨 可视化说明

### 显示元素
- **绿色边界框**: YOLO检测到的目标
  - 显示信息: `ID:轨迹号 YOLO:置信度`
- **红色边界框**: 灰度预测的目标
  - 显示信息: `ID:轨迹号 Pred:相似度 Lost:丢失帧数`
- **红色圆点**: 预测的目标中心位置
- **顶部文字**: 当前帧号、检测数量、预测数量

### 状态指示
- `detected`: 目标被YOLO检测到
- `predicted`: 目标通过灰度预测得到
- `lost`: 目标丢失但仍在跟踪
- 自动删除: 丢失超过5帧的目标

## 🔧 灰度模板说明

### 模板格式
- 尺寸: 25x25像素
- 数据类型: uint8 (0-255)
- 格式: NumPy数组或嵌套列表

### 模板创建建议
```python
# 创建典型的小目标模板
template = np.zeros((25, 25), dtype=np.uint8)
center = 12
for i in range(25):
    for j in range(25):
        dist = np.sqrt((i - center)**2 + (j - center)**2)
        if dist < 3:
            template[i, j] = 200  # 目标中心
        elif dist < 6:
            template[i, j] = 150  # 目标边缘
        else:
            template[i, j] = 100  # 背景
```

### 使用已有模板
```bash
# 使用保存的NumPy数组模板
python -c "
import numpy as np
template = np.load('sample_template.npy')
template_str = str(template.tolist())
print(f'使用模板: {template_str}')
"
```

## 🎯 应用场景

### 适用情况
- 小目标检测（飞机、车辆等）
- 红外视频目标追踪
- 复杂背景下的目标跟踪
- 目标经常被遮挡的场景

### 优势特点
- 检测丢失时自动切换预测模式
- 保持轨迹连续性
- 减少ID切换问题
- 适应目标外观变化

## 📊 性能说明

### 处理速度
- 取决于视频分辨率和目标数量
- YOLO检测: 主要耗时部分
- 灰度预测: 轻量级，耗时很少
- 建议使用GPU加速YOLO推理

### 内存使用
- 主要用于视频帧缓存
- 轨迹信息占用很少
- 灰度模板占用可忽略

## 🐛 故障排除

### 常见问题
1. **模型加载失败**: 检查模型路径和YOLO版本兼容性
2. **视频打开失败**: 确认视频格式和编解码器支持
3. **预测效果差**: 调整灰度模板或搜索参数
4. **处理速度慢**: 使用GPU或降低视频分辨率

### 调试选项
- 添加详细日志输出
- 保存中间处理结果
- 可视化搜索过程

## 📝 更新日志

### v1.0.0
- 实现基本的YOLO+灰度追踪功能
- 支持多目标追踪
- 自动轨迹管理
- 可视化输出

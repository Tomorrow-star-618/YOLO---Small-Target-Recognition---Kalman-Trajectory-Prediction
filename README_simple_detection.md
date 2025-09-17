# 简化版红外小目标检测程序使用说明

## 新功能特性

### 1. 智能输出路径管理
- **自动创建目录**：输出目录自动根据输入视频名称和当前时间创建
- **路径格式**：`otherplan/runs/{视频名称}-{时间戳}/`
- **时间戳格式**：`YYYYMMDD_HHMMSS`
- **视频输出**：保存在 `output-video/` 子目录中
- **图像保存**：保存在 `save-images/` 子目录中

**示例**：
- 输入视频：`vedio/complex-background.mp4`
- 处理时间：2024年12月15日 14:30:25
- 输出目录：`otherplan/runs/complex-background-20241215_143025/`
- 视频输出：`otherplan/runs/complex-background-20241215_143025/output-video/complex-background_detection.mp4`

### 2. 可选CSV输出功能
- **默认状态**：关闭CSV输出功能
- **启用方式**：使用 `--enable-csv` 命令行参数
- **文件位置**：`{输出目录}/{视频名称}_results.csv`

### 3. 多目标图像自动保存
- **默认状态**：关闭多目标图像保存功能
- **启用方式**：使用 `--enable-image-save` 命令行参数
- **触发条件**：当单帧检测到2个或更多目标时
- **保存位置**：`{输出目录}/save-images/`
- **文件命名**：`frame_XXXXXX_targets_N.jpg`
- **图像内容**：包含检测框、标注和帧信息的完整已标注图像

## 命令行使用方法

### 基本使用
```bash
python simple_detection.py
```

### 启用CSV输出
```bash
python simple_detection.py --enable-csv
```

### 指定输入视频
```bash
python simple_detection.py --input /path/to/your/video.mp4
```

### 指定YOLO权重文件
```bash
python simple_detection.py --weights /path/to/weights.pt
```

### 启用快速模式
```bash
python simple_detection.py --fast-mode
```

### 启用多目标图像保存
```bash
python simple_detection.py --enable-image-save
```

### 组合使用示例
```bash
python simple_detection.py --enable-csv --enable-image-save --input custom_video.mp4
```

## 命令行参数详解

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--enable-csv` | - | 启用CSV结果输出功能 | 禁用 |
| `--input` | `-i` | 指定输入视频路径 | `vedio/complex-background.mp4` |
| `--weights` | `-w` | 指定YOLO权重文件路径 | `otherplan/yolo11x.pt` |
| `--fast-mode` | - | 启用快速处理模式 | 禁用 |
| `--enable-image-save` | - | 启用多目标图像保存功能 | 禁用 |

## 输出文件结构

```
otherplan/runs/complex-background-20241215_143025/
├── output-video/
│   └── complex-background_detection.mp4 # 检测结果视频
├── complex-background_results.csv       # CSV结果文件（可选）
└── save-images/                         # 多目标图像目录（可选）
    ├── frame_000150_targets_2.jpg       # 第150帧，2个目标
    ├── frame_000287_targets_3.jpg       # 第287帧，3个目标
    └── ...
```

## CSV文件格式

### 快速模式CSV格式
```csv
frame,time_sec,source,conf,cx,cy,total_targets
0,0.00,yolo,0.85,320,240,2
1,0.04,yolo,0.72,325,245,1
```

### 标准模式CSV格式
```csv
frame,time_sec,target_id,source,conf,cx,cy,x1,y1,x2,y2,total_targets
0,0.000,target_1,yolo,0.850,320,240,310,230,330,250,2
0,0.000,target_2,yolo,0.720,450,180,440,170,460,190,2
```

## 多目标图像保存说明

### 保存条件
- 单帧检测到 ≥2 个目标
- `SAVE_MULTI_TARGET_IMAGES = True`（默认启用）
- 未使用 `--disable-image-save` 参数

### 图像内容
1. **原始帧图像**
2. **彩色检测框**：不同目标使用不同颜色
3. **目标标注**：显示目标编号、来源和置信度
4. **中心点标记**：实心圆点标记目标中心
5. **帧信息**：显示帧号和目标总数

### 文件命名规则
- `frame_XXXXXX_targets_N.jpg`
- `XXXXXX`：6位数帧号（前补零）
- `N`：该帧检测到的目标数量

## 性能监控功能

程序运行时会显示：
- **实时进度**：每100帧显示一次处理进度
- **处理速度**：实时FPS显示
- **自动优化**：FPS过低时自动启用快速模式
- **统计信息**：完成后显示详细检测统计

## 使用建议

### 提升处理速度
1. 使用 `--fast-mode` 启用快速模式
2. 降低输入视频分辨率
3. 确保GPU可用并正确配置
4. 使用 `--disable-image-save` 减少I/O开销

### 获得最佳检测效果
1. 不使用快速模式（默认）
2. 确保YOLO权重文件可用
3. 根据目标特征调整检测参数
4. 启用CSV输出进行详细分析

### 磁盘空间管理
- 多目标图像可能占用大量空间
- 可使用 `--disable-image-save` 禁用图像保存
- 定期清理旧的输出目录

## 故障排除

### 常见问题
1. **找不到输入视频**：检查文件路径是否正确
2. **YOLO权重加载失败**：确认权重文件存在且格式正确
3. **GPU不可用**：检查CUDA和PyTorch安装
4. **输出目录权限错误**：确保有写入权限

### 性能问题
- FPS过低：尝试快速模式或降低分辨率
- 内存不足：减少批处理大小或使用CPU模式
- 磁盘空间不足：禁用多目标图像保存

## 更新日志

### v2.0 新功能
- ✅ 智能输出路径管理
- ✅ 可选CSV输出功能  
- ✅ 多目标图像自动保存
- ✅ 完整的命令行参数支持
- ✅ 改进的进度监控和统计

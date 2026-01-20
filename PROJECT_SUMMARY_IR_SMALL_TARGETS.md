# 红外图像超小目标检测与追踪项目总结报告

> 面向云层复杂背景下的红外飞机小目标检测与持续追踪（YOLO11 + 灰度/梯度预测）。

## 1. 项目概览

- 目标：提升红外场景中“超小目标”（如远距离飞机）在复杂云背景下的检测与追踪鲁棒性。
- 方法：
  - 检测：基于 Ultralytics YOLO11（推荐使用 s 版），结合小目标友好配置与增强策略；可选超小目标结构（P2 检测头）。
  - 追踪：当检测短时丢失时，采用局部灰度与梯度幅值驱动的预测回填，支持 GPU 并行加速，避免轨迹中断与大幅漂移。
- 代码位置：
  - 训练脚本：`v11-new/train_yolo11_small_targets.py`
  - 可选结构文件：`v11-new/yolo11-ultra-small.yaml`（包含 P2 检测层）
  - 追踪系统：`Grayscale-Tracking/grayscale_tracking_system.py`
  - 数据分析：`v11-new/dataset_analysis_report.md`

---

## 2. 数据与配置概览（摘要）

数据统计（来自 `dataset_analysis_report.md`）：
- 总图像：1318；总目标：1718；类别：1（aircraft）
- 划分：训练/验证/测试 = 70%/20%/10%（可重复随机种子 42）
- 平均目标/图：1.23–1.31，分布均匀，适合小目标学习与评估。
- 数据路径：
  - 划分后：`v11-new/dataset/ultra_small_aircraft_split/`
  - 配置：`v11-new/dataset/ultra_small_aircraft_split/ultra_small_aircraft.yaml`

---

## 3. 训练与模型（`train_yolo11_small_targets.py`）

### 3.1 模型与输出组织
- 预训练权重：`yolo11{version}.pt`（默认 `version='s'`）
- 输出目录：`/home/mingxing/worksapce/ultralytics/v11 - new/train/yolo11{version}`
- 项目命名：`yolo11{version}_ultra_small_aircraft`
- 设备选择：自动 CUDA/CPU

### 3.2 关键训练参数（针对红外超小目标）
- 基础：
  - `epochs=250`（加长训练，学习微弱纹理与边缘）
  - `patience=50`（早停更稳健）
  - `batch=8`（考虑显存，兼容更高分辨率与浅层特征）
  - `imgsz=640`（保持较高输入分辨率）
- 学习率与优化：
  - `lr0=0.001`，`lrf=0.01`，`momentum=0.937`，`weight_decay=0.0005`
  - `optimizer='auto'`（Ultralytics 推荐）
- 数据增强（为红外与小目标保守设计）：
  - 关闭色彩相关：`hsv_h=0.0, hsv_s=0.0`；亮度适度：`hsv_v=0.2`
  - 几何变换：`degrees=0`，`translate=0.05`，`scale=0.95`，`shear=0.0`，`perspective=0.0`
  - 翻转：`flipud=0.5`，`fliplr=0.5`
  - 混合：`mosaic=0.8`（适度）、`mixup=0.05`（极少）、`copy_paste=0.2`（增加小样本）
- 损失权重（突出定位）：
  - `box=10.0`（显著提高框定位权重）
  - `cls=0.3`（适度降低分类权重）
  - `dfl=2.0`（提升边界回归细粒度精度）
- 训练策略：
  - `close_mosaic=15`（收尾更贴近真实分布）
  - `amp=True`（混合精度加速）
  - `fraction=1.0`（全量训练）
- 验证/保存：
  - `val=True, plots=True, save=True, save_period=10`（可视化曲线与周期性 checkpoint）
- 训练时阈值：
  - `conf=0.001`（降低正样本门槛，利于超小目标召回）
  - `iou=0.7`（较高 NMS IoU，有利于密集小目标保留）

### 3.3 针对性创新点（训练侧）
- 超小目标友好的增强与损失权重：
  - 保守几何/色彩增强避免“抹除”微弱小目标。
  - 强化 box/dfl，弱化 cls，使优化重心转向“中心-边框”精确定位。
- 收尾关闭 mosaic：减少过度拼接导致的小目标割裂，提升后期稳定收敛。
- 阈值策略：训练阶段使用低置信度阈值与较高 NMS IoU，兼顾召回与重叠目标分离。
- 结构扩展的可选性：
  - `yolo11-ultra-small.yaml` 中新增 P2 检测头与浅层融合，适配 1/4 尺度的细粒度特征（适合更“点状”的目标）。
  - 当前脚本默认加载标准 `yolo11{s}.pt`，保留自定义结构开关，便于后续快速切换实验。

### 3.4 训练产物与验证
- 产物路径：`.../train/yolo11s/yolo11s_ultra_small_aircraft/weights/best.pt`
- 验证接口：`validate_yolo11_model(model_path)`，输出 mAP 指标、生成 plots/COCO JSON。
- 推理接口：`predict_small_targets(model_path, source)`，支持 `augment=True`、`max_det=1000`，并保存标签与分数文件。

---

## 4. 识别与追踪系统（`grayscale_tracking_system.py`）

### 4.1 系统架构
- 检测（绿色框）：YOLO 推理 + 结果限流（`max_detections`，默认 5，仅保留最高置信度若干个）。
- 丢失回填（红色框）：当轨迹短时未被检测覆盖，使用局部灰度/梯度驱动的目标位置预测，维持轨迹连续。
- GPU 加速：若可用，梯度幅值法使用 PyTorch + unfold 并行窗口搜索，显著提速。
- 过程可视化：自动生成 `runs/<视频名_时间戳>/output-video/` 与 `process/`（ROI 拼图、热力图、对比图、灰度矩阵 txt）。

### 4.2 检测阶段（`yolo_detect`）
- 从 YOLO 结果提取 `xyxy, conf, cls`；按 `conf` 排序并截断至 `max_detections`。
- 内置“强制丢失”测试模式：在指定帧区间返回空检测，用于评估预测稳健性（`--test start,end`）。

### 4.3 轨迹关联（`associate_detections`）
- 采用最近邻匹配（< 100 像素）为检测分配/更新轨迹，保存：
  - `last_center / last_bbox / last_detection_frame / confidence / class_id`
  - `last_detection_roi` 与 `last_detection_info`（用于与后续预测做图像级对比）

### 4.4 丢失目标的持续预测（`predict_lost_targets`）
- 轨迹优先级排序与数量控制：
  - 时间衰减权重：$w_t = \frac{1}{1 + 0.1\cdot \text{lost\_frames}}$
  - 轨迹优先级：$p = \text{conf}_{last}\times w_t$
  - 当前帧已检测数量 = `d`，允许预测数量 = `max_detections - d`（不为负），仅保留前若干高优先级轨迹进入预测。
- 最大预测帧数：`max_prediction_frames`（默认 30），超过后删除轨迹避免漂移。
- 预测后更新：以“预测中心 + 上次 bbox 尺寸”形成新框，状态改为 `predicted` 并记录 `lost_frames/score`。

### 4.5 灰度与梯度幅值驱动的预测算法

#### A) 梯度幅值窗口搜索（CPU 版本：`gradient_magnitude_prediction`）
- ROI：以上次中心提取 40×40 灰度 ROI。
- 窗口：在 ROI 内滑动 5×5 窗口，计算综合得分：
  - 窗口均值 $\mu$（更亮的热点倾向于小目标）、
  - 中心 3×3 上的 Sobel 梯度幅值均值 $g$（边缘能量，抵抗亮度平坦噪声）。
- 得分函数：$\text{score} = \mu + 0.3\, g$，取全局最大位置为新中心。
- 归一化：$\hat s = \min(1, \text{score}/255)$；若 $\hat s < \text{min\_prediction\_confidence}$（默认 0.1），保留轨迹但不更新中心。

#### B) 梯度幅值窗口搜索（GPU 版本：`gradient_magnitude_prediction_gpu`）
- 将 40×40 ROI 转张量，使用 `unfold` 一次性展开所有 5×5 窗口（形如 `[H', W', 5, 5]`）。
- 逐窗口并行计算 $\mu$ 与中心 3×3 的 Sobel 梯度幅值 $g$，并以同一得分函数挑选最大值。
- 显著减少逐窗循环的 CPU 开销，适合多目标/高分辨率场景。

#### C) 模板相关匹配（`template_matching`）与灰度相似性搜索（`grayscale_similarity_search`）
- 若提供本地灰度模板（如 25×25 的典型小目标亮点模板），可进行：
  - OpenCV `matchTemplate` 的标准化互相关（在邻域内搜索最大响应）。
  - 自实现的归一化相关系数（NCC）在稀疏网格上搜索：
    $$ \rho = \frac{\sum (R-\bar R)(T-\bar T)}{\sqrt{\sum (R-\bar R)^2\,\sum (T-\bar T)^2}} $$
- 与梯度幅值法形成互补：若模板得分更高，则采用模板估计作为最终中心。

### 4.6 可视化与过程保存（`save_process_images`）
- 对“最后一次检测 ROI”与“当前预测 ROI”生成 2×3 对比图（灰度、热力、等高线），并导出对应灰度矩阵到 TXT。
- 文件命名包含时间（秒）、帧号与中心坐标，便于复盘与论文/汇报图例选取。

### 4.7 运行与输出结构
- 运行示例：
  - 基本：`python grayscale_tracking_system.py --video vedio/test.mp4`
  - 保存过程：加 `--save-process`
  - 测试模式：`--test 30,80`（在 30–80 帧强制丢失以评估回填）
- 输出目录：`Grayscale-Tracking/runs/<视频名_时间戳>/`
  - `output-video/`：叠加可视化的视频
  - `process/roi_patches/` 与 `process/grayscale_data/`：过程图与矩阵

### 4.8 设计取舍与边界条件
- ROI 与窗口大小（40×40 + 5×5）在“点状小目标”场景下经验有效，极端大目标可适当增大；极端弱目标可提高梯度权重或降阈值。
- 多目标拥挤时，通过 `max_detections` 与优先级裁剪，限制“检测+预测”的总体数量，避免画面拥堵与误跟。
- 预测上限（`max_prediction_frames`）防漂移；若长时间遮挡，可与 Kalman/运动模型融合进一步增强。

---

## 5. 关键创新点总表

1. 小目标友好训练策略：保守增强 + 强化定位损失 + 收尾关 mosaic + 低 conf / 高 IoU 组合阈值。
2. 可选 P2 检测头结构：提升 1/4 尺度细节表达，适配“点级”目标（`yolo11-ultra-small.yaml`）。
3. 检测-预测一体的追踪框架：按优先级裁剪预测数量，保证实时性与可读性。
4. 基于灰度与梯度幅值的局部搜索：轻量、可 GPU 并行，加速恢复短时丢失轨迹。
5. 完整过程可视化与数据化：ROI 对比、热力/等高线与矩阵导出，便于问题定位与研究复现。

---

## 6. 结果与经验

- 实测在复杂云层背景下，低阈值（训练/验证阶段）与高 IoU 的组合有助于提高召回并控制误检。
- 追踪阶段的 GPU 梯度搜索带来明显的处理 FPS 增益，尤其是多轨并行时。
- 过程导出（TXT/图像）对调参帮助极大，建议作为默认开启的诊断手段（可在长视频时择要开启）。

---

## 7. 局限与改进方向

- 运动建模：当前以“局部视觉搜索”为主，后续可与 Kalman/IMM/CF（恒速/恒加速度）融合，降低长遮挡时的飘移。
- 多模态：若可获取可见光/温度域等多模信息，考虑简单的特征级/决策级融合，提升稳健性。
- 结构自定义：进一步验证 `yolo11-ultra-small.yaml` 的 P2 头在本数据上的收益，对比标准 head 的召回/精度与速度折中。
- 自蒸馏与伪标注：增强弱标注段落的监督信号，增加小目标样本密度。

---

## 8. 附：核心参数速览

- 训练（默认 s 版）：`epochs=250, batch=8, imgsz=640, lr0=0.001, lrf=0.01, box=10.0, cls=0.3, dfl=2.0, mosaic=0.8, mixup=0.05, copy_paste=0.2, conf=0.001, iou=0.7`。
- 追踪：`roi_size=40, window=5×5, min_prediction_confidence=0.1, max_prediction_frames=30, max_detections=5`。
- 路径：
  - 数据：`v11-new/dataset/ultra_small_aircraft_split/ultra_small_aircraft.yaml`
  - 训练脚本：`v11-new/train_yolo11_small_targets.py`
  - 追踪脚本：`Grayscale-Tracking/grayscale_tracking_system.py`
  - 可选结构：`v11-new/yolo11-ultra-small.yaml`

---

## 9. 快速使用（可选）

- 训练：
```bash
cd "/home/mingxing/worksapce/ultralytics/v11 - new"
python train_yolo11_small_targets.py
```

- 追踪：
```bash
cd "/home/mingxing/worksapce/ultralytics/Grayscale-Tracking"
python grayscale_tracking_system.py --video vedio/test.mp4 --model ../v11-new/train/yolo11s_ultra_small_aircraft/weights/best.pt --save-process
```

---

以上总结覆盖训练细节、关键创新与追踪算法与实现，可直接用于项目汇报与后续技术沉淀。
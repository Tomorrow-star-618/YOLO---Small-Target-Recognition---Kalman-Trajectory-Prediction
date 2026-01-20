#!/bin/bash
# YOLO11 超小目标检测训练脚本
# 使用自定义超小目标配置

echo "========================================"
echo "YOLO11 超小目标检测训练"
echo "配置：P2+P3+P4+P5四层检测头"
echo "========================================"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 训练命令 - 使用自定义超小目标配置
yolo train \
    model="/home/mingxing/worksapce/ultralytics/v11 - new/yolo11-ultra-small.yaml" \
    data="/home/mingxing/worksapce/ultralytics/v11 - new/dataset/ultra_small_aircraft_split/ultra_small_aircraft.yaml" \
    epochs=250 \
    patience=50 \
    batch=8 \
    imgsz=640 \
    lr0=0.001 \
    lrf=0.01 \
    momentum=0.937 \
    weight_decay=0.0005 \
    hsv_h=0.0 \
    hsv_s=0.0 \
    hsv_v=0.2 \
    degrees=0 \
    translate=0.05 \
    scale=0.95 \
    shear=0.0 \
    perspective=0.0 \
    flipud=0.5 \
    fliplr=0.5 \
    mosaic=0.8 \
    mixup=0.05 \
    copy_paste=0.2 \
    optimizer=auto \
    box=10.0 \
    cls=0.3 \
    dfl=2.0 \
    close_mosaic=20 \
    amp=True \
    save=True \
    save_period=10 \
    val=True \
    plots=True \
    device=0 \
    project="/home/mingxing/worksapce/ultralytics/v11 - new/train" \
    name="yolo11_ultra_small_aircraft" \
    exist_ok=True \
    conf=0.001 \
    iou=0.7

echo "训练完成！检查结果在: /home/mingxing/worksapce/ultralytics/v11 - new/train/yolo11_ultra_small_aircraft/"

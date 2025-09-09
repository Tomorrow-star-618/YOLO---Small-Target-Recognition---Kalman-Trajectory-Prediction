#!/usr/bin/env python3
"""
Small Target Detection Training Script for Infrared Images 针对红外图像小目标检测（如飞机）的训练脚本.

红外图像特点：
- 通常为单通道灰度图像
- 目标与背景的温度差异形成对比
- 可能存在噪声和大气影响
- 小目标边界可能不够清晰
"""

import torch

from ultralytics import YOLO


def train_small_target_model():
    """训练小目标检测模型."""
    # 加载自定义小目标检测配置
    model = YOLO("ultralytics/cfg/models/v8/yolov8-small.yaml")

    # 小目标检测优化的训练参数
    results = model.train(
        # 数据配置
        data="yolo_dataset/dataset.yaml",  # 替换为您的数据集配置文件
        # 基础训练参数（红外图像优化）
        epochs=150,  # 增加训练轮数（红外图像特征学习需要更多时间）
        patience=30,  # 减少早停耐心值（红外图像收敛较快）
        batch=16,  # 适中的批次大小（640x512输入尺寸下内存友好）
        imgsz=640,  # 使用接近原始尺寸的输入（640x512 → 640x640，保持细节且高效）
        # 学习率配置（红外图像优化）
        lr0=0.0008,  # 稍微降低初始学习率（红外图像特征更敏感）
        lrf=0.005,  # 更小的最终学习率因子
        momentum=0.95,  # 增加动量（帮助红外特征稳定学习）
        weight_decay=0.0008,  # 增加权重衰减（防止过拟合）
        # 数据增强配置（针对红外图像小目标优化）
        hsv_h=0.0,  # 色调增强（红外图像通常是灰度，关闭色调变化）
        hsv_s=0.0,  # 饱和度增强（红外图像关闭饱和度变化）
        hsv_v=0.3,  # 明度增强（保留亮度变化，模拟不同温度条件）
        degrees=0,  # 旋转角度（关闭旋转增强）
        translate=0.05,  # 平移（减小平移幅度，保护小目标位置）
        scale=0.8,  # 缩放（减小缩放范围，避免小目标过度缩小）
        shear=0.0,  # 剪切（关闭以保护小目标）
        perspective=0.0,  # 透视变换（关闭以保护小目标）
        flipud=0.3,  # 垂直翻转（减少翻转概率）
        fliplr=0.3,  # 水平翻转（减少翻转概率）
        mosaic=0.3,  # 减少mosaic增强强度（红外图像拼接可能影响温度信息）
        mixup=0.05,  # 混合增强（减少混合，保持红外特征）
        copy_paste=0.15,  # 复制粘贴增强（适当增加，增加小目标样本）
        # 训练策略
        optimizer="AdamW",  # 使用AdamW优化器
        close_mosaic=20,  # 最后20轮关闭mosaic
        amp=True,  # 混合精度训练
        # 保存配置
        save=True,
        save_period=10,  # 每10轮保存一次
        # 验证配置
        val=True,
        plots=True,
        # 设备配置
        device=0 if torch.cuda.is_available() else "cpu",
        # 项目配置
        project="small_target_detection",
        name="yolov8_small_aircraft",
        exist_ok=True,
        # 小目标检测特殊配置（红外图像优化）
        conf=0.15,  # 适当降低置信度阈值（红外小目标可能对比度较低）
        iou=0.6,  # 稍微降低NMS IoU阈值（红外目标边界可能不够清晰）
    )

    return results


def validate_model(model_path):
    """验证训练好的模型."""
    model = YOLO(model_path)

    # 在验证集上评估
    results = model.val(
        data="yolo_dataset/dataset.yaml",  # 使用正确的数据集路径
        imgsz=640,  # 与训练时保持一致的输入尺寸
        batch=1,
        conf=0.15,  # 与训练配置保持一致
        iou=0.6,  # 与训练配置保持一致
        device=0 if torch.cuda.is_available() else "cpu",
    )

    return results


def predict_small_targets(model_path, source):
    """使用训练好的模型进行小目标预测."""
    model = YOLO(model_path)

    # 预测配置（针对红外小目标优化）
    results = model.predict(
        source=source,
        imgsz=640,  # 与训练时保持一致的输入尺寸
        conf=0.1,  # 降低置信度阈值以检测更多小目标
        iou=0.5,  # 调整NMS阈值
        max_det=1000,  # 增加最大检测数量
        augment=True,  # 使用TTA（测试时增强）
        agnostic_nms=False,
        save=True,
        save_txt=True,
        save_conf=True,
    )

    return results


if __name__ == "__main__":
    print("开始训练小目标检测模型...")

    # 训练模型
    train_results = train_small_target_model()

    print("训练完成！")
    print(f"最佳模型路径: {train_results.save_dir}/weights/best.pt")

    # 可选：验证模型
    # val_results = validate_model("path/to/best.pt")

    # 可选：测试预测
    # pred_results = predict_small_targets("path/to/best.pt", "path/to/test/images")

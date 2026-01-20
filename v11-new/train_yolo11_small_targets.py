#!/usr/bin/env python3
"""
YOLO11 Small Target Detection Training Script for Infrared Images
针对红外图像小目标检测（如飞机）的YOLO11训练脚本

Version: YOLO11s (推荐用于小目标检测的平衡版本)
- 参数量适中：9.46M参数，21.7 GFLOPs
- 在准确性和速度之间有很好的平衡
- 适合复杂云背景下的红外小目标检测
"""

from ultralytics import YOLO
import torch
import os
from pathlib import Path

def train_yolo11_small_target_model(version='s'):
    """
    训练YOLO11小目标检测模型
    
    Args:
        version: 模型版本 ('n', 's', 'm', 'l', 'x')
    """
    
    # 设置训练输出目录（根据版本区分）
    project_dir = Path(f'/home/mingxing/worksapce/ultralytics/v11 - new/train/yolo11{version}')
    project_dir.mkdir(parents=True, exist_ok=True)
    print(f"训练输出目录: {project_dir}")
    
    # 加载YOLO11指定版本的预训练模型
    model_name = f'yolo11{version}.pt'
    print(f"加载预训练模型: {model_name} ({version.upper()}版本)")
    model = YOLO(model_name)
    
    # 检查是否有自定义配置文件（可选）
    custom_config = '/home/mingxing/worksapce/ultralytics/v11 - new/yolo11-ultra-small.yaml'
    if os.path.exists(custom_config):
        print(f"检测到超小目标优化配置文件: {custom_config}")
        print("注意：当前使用标准预训练模型，如需自定义配置请手动修改")
    
    # YOLO11小目标检测优化的训练参数
    results = model.train(
        # 数据配置
        data='/home/mingxing/worksapce/ultralytics/v11 - new/dataset/ultra_small_aircraft_split/ultra_small_aircraft.yaml',
        
        # 基础训练参数（超小目标优化）
        epochs=250,          # 增加训练轮数（超小目标需要更多训练）
        patience=50,         # 增加早停耐心值
        batch=8,             # 减小批次大小（P2层需要更多内存）
        imgsz=640,           # 输入图像尺寸（保持高分辨率）
        
        # 学习率配置（YOLO11优化）
        lr0=0.001,          # 初始学习率（v11默认值）
        lrf=0.01,           # 最终学习率因子
        momentum=0.937,     # 动量
        weight_decay=0.0005, # 权重衰减
        
        # 数据增强配置（针对超小目标优化）
        hsv_h=0.0,          # 色调增强（红外图像是灰度，关闭）
        hsv_s=0.0,          # 饱和度增强（红外图像关闭）
        hsv_v=0.2,          # 明度增强（减少变化，保护小目标对比度）
        degrees=0,          # 旋转角度（关闭以保护超小目标）
        translate=0.05,     # 平移（大幅减小，避免超小目标移出视野）
        scale=0.95,         # 缩放（极小缩放范围，避免超小目标消失）
        shear=0.0,          # 剪切（关闭）
        perspective=0.0,    # 透视变换（关闭）
        flipud=0.5,         # 垂直翻转
        fliplr=0.5,         # 水平翻转
        mosaic=0.8,         # 适度减少mosaic（避免超小目标过度分割）
        mixup=0.05,         # 最小混合增强
        copy_paste=0.2,     # 增加复制粘贴（增加超小目标样本）
        
        # 优化器配置
        optimizer='auto',   # YOLO11推荐使用auto
        
        # 损失函数权重（超小目标优化）
        box=10.0,          # 大幅增加边界框损失权重（超小目标定位更重要）
        cls=0.3,           # 适度降低分类损失权重
        dfl=2.0,           # 增加DFL损失权重（更精确的边界框）
        
        # 训练策略
        close_mosaic=15,   # 最后15轮关闭mosaic
        amp=True,          # 混合精度训练
        fraction=1.0,      # 使用全部数据集
        
        # 保存配置
        save=True,
        save_period=10,    # 每10轮保存一次检查点
        
        # 验证配置
        val=True,
        plots=True,
        
        # 设备配置
        device=0 if torch.cuda.is_available() else 'cpu',
        
        # 项目配置
        project=str(project_dir),
        name=f'yolo11{version}_ultra_small_aircraft',
        exist_ok=True,
        
        # 检测阈值配置（小目标优化）
        conf=0.001,        # 训练时置信度阈值
        iou=0.7,           # NMS IoU阈值
    )
    
    return results

def validate_yolo11_model(model_path):
    """验证训练好的YOLO11模型"""
    model = YOLO(model_path)
    
    results = model.val(
        data='/home/mingxing/worksapce/ultralytics/v11 - new/dataset/ultra_small_aircraft_split/ultra_small_aircraft.yaml',
        imgsz=640,
        batch=1,
        conf=0.1,          # 验证时降低置信度阈值
        iou=0.6,
        device=0 if torch.cuda.is_available() else 'cpu',
        plots=True,
        save_json=True,
    )
    
    return results

def predict_small_targets(model_path, source):
    """使用训练好的YOLO11模型进行小目标预测"""
    model = YOLO(model_path)
    
    results = model.predict(
        source=source,
        imgsz=640,
        conf=0.05,         # 预测时使用更低的置信度阈值
        iou=0.5,           # NMS IoU阈值
        max_det=1000,      # 最大检测数量
        augment=True,      # 测试时增强
        agnostic_nms=False,
        save=True,
        save_txt=True,
        save_conf=True,
        project='/home/mingxing/worksapce/ultralytics/v11 - new/train',
        name='predictions',
    )
    
    return results

def main():
    """主函数"""
    print("=" * 60)
    print("YOLO11 红外小目标检测训练")
    print("=" * 60)
    
    # 选择训练版本
    version = 's'  # 推荐使用s版本进行小目标检测
    print(f"使用YOLO11{version.upper()}版本进行训练")
    
    try:
        # 训练模型
        print("\n开始训练...")
        train_results = train_yolo11_small_target_model(version)
        
        print("\n训练完成！")
        best_model_path = train_results.save_dir / "weights" / "best.pt"
        print(f"最佳模型保存在: {best_model_path}")
        
        # 验证模型
        print("\n开始验证模型...")
        val_results = validate_yolo11_model(str(best_model_path))
        
        print("\n验证完成！")
        print(f"mAP@0.5: {val_results.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {val_results.box.map:.4f}")
        
        return str(best_model_path)
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        return None

if __name__ == "__main__":
    best_model = main()
    
    if best_model:
        print(f"\n训练成功完成！最佳模型: {best_model}")
        print("\n下一步可以使用该模型进行预测:")
        print(f"python -c \"from train_yolo11_small_targets import predict_small_targets; predict_small_targets('{best_model}', 'path/to/test/images')\"")

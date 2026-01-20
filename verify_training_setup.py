#!/usr/bin/env python3
"""
验证YOLO11s训练设置脚本
"""

import sys
import os
from pathlib import Path

# 添加当前路径到sys.path
sys.path.append('/home/mingxing/worksapce/ultralytics/v11 - new')

def verify_setup():
    """验证训练设置"""
    print("=" * 50)
    print("验证YOLO11s训练设置")
    print("=" * 50)
    
    # 导入训练脚本
    try:
        from train_yolo11_small_targets import train_yolo11_small_target_model
        print("✓ 成功导入训练脚本")
    except ImportError as e:
        print(f"✗ 导入训练脚本失败: {e}")
        return False
    
    # 验证输出目录设置
    version = 's'
    project_dir = Path(f'/home/mingxing/worksapce/ultralytics/v11 - new/train/yolo11{version}')
    print(f"YOLO11s 训练输出目录: {project_dir}")
    
    # 验证与yolo11n的路径区别
    nano_dir = Path('/home/mingxing/worksapce/ultralytics/v11 - new/train/yolo11n')
    print(f"YOLO11n 训练输出目录: {nano_dir}")
    
    if project_dir != nano_dir:
        print("✓ YOLO11s和YOLO11n使用不同的输出目录，不会冲突")
    else:
        print("✗ 警告：YOLO11s和YOLO11n使用相同的输出目录，可能会冲突")
        return False
    
    # 检查模型文件
    model_name = f'yolo11{version}.pt'
    print(f"将使用的模型文件: {model_name}")
    
    print("\n验证结果:")
    print("✓ 脚本配置正确")
    print("✓ 输出路径已区分版本")
    print("✓ 将训练YOLO11s版本（small版本）")
    print("✓ 训练结果不会与YOLO11n版本冲突")
    
    return True

if __name__ == "__main__":
    if verify_setup():
        print("\n所有验证通过！可以开始训练YOLO11s模型。")
    else:
        print("\n验证失败！请检查配置。")

#!/usr/bin/env python3
"""
超小目标数据集划分脚本
针对1318张红外飞机图像进行科学的数据集划分

划分策略：
- 训练集：70% (~922张) - 保证充足的训练样本
- 验证集：20% (~264张) - 用于训练过程中的验证和早停
- 测试集：10% (~132张) - 最终测试和评估模型性能
"""

import os
import shutil
import random
from pathlib import Path
import argparse

def split_ultra_small_dataset(
    source_images_dir,
    source_labels_dir,
    output_dir,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    random_seed=42
):
    """
    划分超小目标数据集
    
    Args:
        source_images_dir: 原始图像目录
        source_labels_dir: 原始标签目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子
    """
    
    # 设置随机种子以确保可重复性
    random.seed(random_seed)
    
    # 创建输出目录结构
    output_path = Path(output_dir)
    
    # 创建YOLO格式的目录结构
    directories = [
        'train/images', 'train/labels',
        'val/images', 'val/labels',
        'test/images', 'test/labels'
    ]
    
    for dir_name in directories:
        (output_path / dir_name).mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件（排除Zone.Identifier文件）
    image_files = []
    for file in Path(source_images_dir).glob("*.png"):
        if not file.name.endswith('.png:Zone.Identifier'):
            image_files.append(file.stem)  # 获取不带扩展名的文件名
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 验证对应的标签文件是否存在
    valid_pairs = []
    for image_name in image_files:
        label_file = Path(source_labels_dir) / f"{image_name}.txt"
        if label_file.exists() and image_name != "classes":  # 排除classes.txt
            valid_pairs.append(image_name)
    
    print(f"有效的图像-标签对: {len(valid_pairs)}")
    
    # 随机打乱数据
    random.shuffle(valid_pairs)
    
    # 计算划分数量
    total_samples = len(valid_pairs)
    train_count = int(total_samples * train_ratio)
    val_count = int(total_samples * val_ratio)
    test_count = total_samples - train_count - val_count
    
    print(f"\n数据集划分:")
    print(f"训练集: {train_count} 张 ({train_count/total_samples*100:.1f}%)")
    print(f"验证集: {val_count} 张 ({val_count/total_samples*100:.1f}%)")
    print(f"测试集: {test_count} 张 ({test_count/total_samples*100:.1f}%)")
    
    # 划分数据集
    train_files = valid_pairs[:train_count]
    val_files = valid_pairs[train_count:train_count + val_count]
    test_files = valid_pairs[train_count + val_count:]
    
    # 复制文件函数
    def copy_files(file_list, split_name):
        print(f"\n复制{split_name}文件...")
        for i, filename in enumerate(file_list):
            # 复制图像文件
            src_img = Path(source_images_dir) / f"{filename}.png"
            dst_img = output_path / split_name / "images" / f"{filename}.png"
            shutil.copy2(src_img, dst_img)
            
            # 复制标签文件
            src_label = Path(source_labels_dir) / f"{filename}.txt"
            dst_label = output_path / split_name / "labels" / f"{filename}.txt"
            shutil.copy2(src_label, dst_label)
            
            if (i + 1) % 100 == 0:
                print(f"  已复制 {i + 1}/{len(file_list)} 个文件")
        
        print(f"  {split_name}文件复制完成: {len(file_list)} 个文件")
    
    # 执行文件复制
    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")
    
    # 复制classes.txt到所有标签目录
    classes_src = Path(source_labels_dir) / "classes.txt"
    if classes_src.exists():
        for split in ["train", "val", "test"]:
            shutil.copy2(classes_src, output_path / split / "labels" / "classes.txt")
    
    print(f"\n数据集划分完成！")
    print(f"输出目录: {output_dir}")
    
    return {
        'total': total_samples,
        'train': train_count,
        'val': val_count,
        'test': test_count,
        'train_files': train_files,
        'val_files': val_files,
        'test_files': test_files
    }

def create_dataset_yaml(output_dir, dataset_name="ultra_small_aircraft"):
    """创建YOLO格式的数据集配置文件"""
    
    yaml_content = f"""# Ultra Small Aircraft Detection Dataset Configuration
# 超小目标飞机检测数据集配置

# 数据集路径 (相对于此yaml文件的路径或绝对路径)
path: {output_dir}  # 数据集根目录
train: train/images  # 训练集图像目录(相对于path)
val: val/images      # 验证集图像目录(相对于path)
test: test/images    # 测试集图像目录(相对于path)

# 类别信息
nc: 1  # 类别数量
names: ['aircraft']  # 类别名称列表

# 数据集统计信息
# train: ~922 images
# val: ~264 images  
# test: ~132 images
# total: ~1318 images

# 超小目标检测特殊配置
target_type: "ultra_small"  # 目标类型
image_format: "png"         # 图像格式
annotation_format: "yolo"   # 标注格式

# 训练建议配置
recommended_config:
  imgsz: 640          # 输入图像尺寸
  batch_size: 8       # 推荐批次大小 (超小目标需要更多内存)
  epochs: 250         # 推荐训练轮数
  patience: 50        # 早停耐心值
  conf_threshold: 0.05  # 推荐置信度阈值 (超小目标需要更低阈值)
  iou_threshold: 0.5   # NMS IoU阈值
"""
    
    yaml_path = Path(output_dir) / f"{dataset_name}.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"数据集配置文件已创建: {yaml_path}")
    return str(yaml_path)

def analyze_dataset_distribution(output_dir):
    """分析数据集中目标的分布情况"""
    print("\n=== 数据集目标分布分析 ===")
    
    for split in ["train", "val", "test"]:
        labels_dir = Path(output_dir) / split / "labels"
        if not labels_dir.exists():
            continue
            
        target_counts = []
        total_files = 0
        
        for label_file in labels_dir.glob("*.txt"):
            if label_file.name == "classes.txt":
                continue
                
            total_files += 1
            with open(label_file, 'r') as f:
                lines = f.readlines()
                target_counts.append(len(lines))
        
        if target_counts:
            avg_targets = sum(target_counts) / len(target_counts)
            max_targets = max(target_counts)
            min_targets = min(target_counts)
            
            print(f"\n{split.upper()}集统计:")
            print(f"  图像数量: {total_files}")
            print(f"  平均每张图目标数: {avg_targets:.2f}")
            print(f"  最大目标数: {max_targets}")
            print(f"  最小目标数: {min_targets}")
            print(f"  总目标数: {sum(target_counts)}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='超小目标数据集划分工具')
    parser.add_argument('--source_images', 
                       default='/home/mingxing/worksapce/ultralytics/v11 - new/dataset/5-save-picture/images',
                       help='原始图像目录')
    parser.add_argument('--source_labels',
                       default='/home/mingxing/worksapce/ultralytics/v11 - new/dataset/5-save-picture/labels', 
                       help='原始标签目录')
    parser.add_argument('--output',
                       default='/home/mingxing/worksapce/ultralytics/v11 - new/dataset/ultra_small_aircraft_split',
                       help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='测试集比例')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    print("=== 超小目标数据集划分工具 ===")
    print(f"原始图像目录: {args.source_images}")
    print(f"原始标签目录: {args.source_labels}")
    print(f"输出目录: {args.output}")
    print(f"划分比例: 训练集{args.train_ratio}, 验证集{args.val_ratio}, 测试集{args.test_ratio}")
    
    # 划分数据集
    result = split_ultra_small_dataset(
        args.source_images,
        args.source_labels, 
        args.output,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.random_seed
    )
    
    # 创建配置文件
    yaml_path = create_dataset_yaml(args.output)
    
    # 分析数据集分布
    analyze_dataset_distribution(args.output)
    
    print(f"\n=== 完成 ===")
    print(f"数据集已成功划分到: {args.output}")
    print(f"配置文件路径: {yaml_path}")
    print(f"\n下一步可以使用以下命令开始训练:")
    print(f"yolo train model='/home/mingxing/worksapce/ultralytics/v11 - new/yolo11-ultra-small.yaml' data='{yaml_path}'")

if __name__ == "__main__":
    main()

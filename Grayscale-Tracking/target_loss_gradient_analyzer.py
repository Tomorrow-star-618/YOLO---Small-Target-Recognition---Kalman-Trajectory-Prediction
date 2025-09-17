#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
目标丢失区域梯度分析工具
使用梯度幅值法分析目标丢失区域，输出带中心点的可视化矩阵图像
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import argparse
import os

class GradientROIAnalyzer:
    """梯度ROI分析器"""
    
    def __init__(self, roi_size=5, output_dir=None):
        """初始化"""
        self.roi_size = roi_size
        self.output_dir = output_dir
        
    def process_image(self, image_path):
        """处理单张图像"""
        print(f"📝 处理图像: {Path(image_path).name}")
        
        # 读取图像并转为灰度
        color_image = cv2.imread(str(image_path))
        if color_image is None:
            print(f"❌ 无法读取图像: {image_path}")
            return None
            
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        h, w = gray_image.shape
        
        # 使用梯度幅值法查找最佳ROI
        result = self.find_roi_gradient_magnitude(gray_image)
        
        # 生成并保存可视化结果
        self.save_visualization(image_path, gray_image, result)
        
        return result
        
    def find_roi_gradient_magnitude(self, gray_image):
        """使用梯度幅值法寻找最佳ROI"""
        h, w = gray_image.shape
        half_roi = self.roi_size // 2
        
        # 计算梯度
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        best_score = -1
        best_center = (0, 0)
        
        # 遍历所有可能的ROI中心
        for y in range(half_roi, h - half_roi):
            for x in range(half_roi, w - half_roi):
                roi_grad = gradient_magnitude[y-half_roi:y+half_roi+1, x-half_roi:x+half_roi+1]
                roi_intensity = gray_image[y-half_roi:y+half_roi+1, x-half_roi:x+half_roi+1]
                
                # 结合梯度和强度计算得分
                score = np.mean(roi_grad) * np.mean(roi_intensity) / 255.0
                
                if score > best_score:
                    best_score = score
                    best_center = (x, y)
        
        # 提取最佳ROI数据
        x, y = best_center
        roi_data = gray_image[y-half_roi:y+half_roi+1, x-half_roi:x+half_roi+1]
        
        return {
            'center_x': x,
            'center_y': y,
            'confidence': float(best_score),
            'roi_data': roi_data,
            'roi_mean': float(np.mean(roi_data)),
            'roi_max': int(np.max(roi_data)),
            'roi_min': int(np.min(roi_data))
        }
        
    def save_visualization(self, image_path, gray_image, result):
        """生成并保存可视化结果"""
        if not result:
            return
            
        # 创建输出目录
        if self.output_dir:
            output_path = Path(self.output_dir)
        else:
            output_path = Path(image_path).parent / 'output'
            
        output_path.mkdir(exist_ok=True)
        
        # 提取原始文件名
        original_name = Path(image_path).stem
        output_file = output_path / f"{original_name}_analysis.png"
        
        # 绘制图像
        fig = plt.figure(figsize=(12, 10))
        
        # 子图1: 原始灰度图像和ROI位置
        ax1 = plt.subplot(2, 1, 1)
        ax1.imshow(gray_image, cmap='gray')
        ax1.set_title(f"原始灰度图 ({gray_image.shape[1]}x{gray_image.shape[0]})")
        
        # 标记ROI位置
        x, y = result['center_x'], result['center_y']
        half_roi = self.roi_size // 2
        rect = plt.Rectangle((x-half_roi-0.5, y-half_roi-0.5), self.roi_size, self.roi_size, 
                             linewidth=2, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)
        ax1.plot(x, y, 'r+', markersize=10)
        ax1.text(x+self.roi_size, y, f"({x},{y})", color='red', fontsize=12, 
                backgroundcolor='white', bbox=dict(facecolor='white', alpha=0.7))
        
        # 子图2: ROI区域矩阵可视化
        ax2 = plt.subplot(2, 1, 2)
        roi_data = result['roi_data']
        
        # 显示ROI灰度图
        ax2.imshow(roi_data, cmap='inferno')
        
        # 添加数值标签
        for i in range(roi_data.shape[0]):
            for j in range(roi_data.shape[1]):
                text_color = 'white' if roi_data[i, j] < 128 else 'black'
                ax2.text(j, i, str(roi_data[i, j]), ha='center', va='center', 
                        color=text_color, fontsize=9, fontweight='bold')
        
        # 设置标题和其他参数
        ax2.set_title(f"ROI 矩阵 ({self.roi_size}x{self.roi_size}), 均值: {result['roi_mean']:.2f}")
        ax2.set_xticks(range(roi_data.shape[1]))
        ax2.set_yticks(range(roi_data.shape[0]))
        ax2.grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # 中心点标记
        plt.figtext(0.5, 0.02, f"中心坐标: ({result['center_x']}, {result['center_y']}), 置信度: {result['confidence']:.2f}", 
                   ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ 已保存分析结果: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='目标丢失区域梯度分析工具')
    parser.add_argument('--input', default='/home/mingxing/worksapce/ultralytics/target_loss_patches',
                        help='输入图像目录')
    parser.add_argument('--output', default='/home/mingxing/worksapce/ultralytics/target_loss_patches/output',
                        help='输出结果目录')
    parser.add_argument('--roi-size', type=int, default=5,
                        help='ROI区域大小')
    parser.add_argument('--max', type=int, default=0,
                        help='最多处理图像数量，0表示处理全部')
    
    args = parser.parse_args()
    
    # 获取输入图像列表
    input_path = Path(args.input)
    if not input_path.exists() or not input_path.is_dir():
        print(f"❌ 输入目录不存在: {args.input}")
        return
    
    # 获取所有图像文件，排除output子目录中的文件
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(list(input_path.glob(ext)))
    
    # 过滤掉output目录中的文件
    output_path = Path(args.output)
    image_files = [f for f in image_files if not str(f).startswith(str(output_path))]
    
    if not image_files:
        print(f"❌ 未找到图像文件: {args.input}")
        return
    
    print(f"🖼️ 共找到 {len(image_files)} 张图像")
    
    # 限制处理数量
    if args.max > 0 and args.max < len(image_files):
        image_files = image_files[:args.max]
        print(f"⚠️ 限制处理数量为前 {args.max} 张图像")
    
    # 创建分析器
    analyzer = GradientROIAnalyzer(roi_size=args.roi_size, output_dir=args.output)
    
    # 处理每张图像
    results = []
    for image_file in image_files:
        result = analyzer.process_image(image_file)
        if result:
            results.append((str(image_file), result))
    
    # 保存汇总结果
    summary_file = Path(args.output) / "analysis_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("梯度幅值法分析结果汇总\n")
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'文件名':<50} {'中心坐标':<12} {'置信度':<10} {'ROI均值':<10} {'最大值':<8} {'最小值':<8}\n")
        f.write("-" * 100 + "\n")
        
        for image_path, data in results:
            filename = Path(image_path).name
            f.write(f"{filename:<50} ({data['center_x']:>2},{data['center_y']:>2})      {data['confidence']:<10.2f} {data['roi_mean']:<10.2f} {data['roi_max']:<8} {data['roi_min']:<8}\n")
    
    print(f"\n📄 已保存汇总结果: {summary_file}")
    print(f"🎉 分析完成！共处理 {len(results)} 张图像")


if __name__ == "__main__":
    from datetime import datetime
    main()

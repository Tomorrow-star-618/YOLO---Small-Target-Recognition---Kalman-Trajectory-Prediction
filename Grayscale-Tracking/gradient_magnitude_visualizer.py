#!/usr/bin/env python3
"""
改进的梯度幅值可视化分析器
支持可选的圆形热点法对比分析
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

class GradientMagnitudeAnalyzer:
    def __init__(self, roi_size=5):
        """
        初始化梯度幅值分析器
        
        Args:
            roi_size: ROI窗口大小
        """
        self.roi_size = roi_size
        self.half_size = roi_size // 2
        
    def analyze_image(self, image_path, output_dir=None, enable_circular=False):
        """
        分析图像并生成可视化结果
        
        Args:
            image_path: 输入图像路径
            output_dir: 输出目录
            enable_circular: 是否启用圆形热点法对比分析
        """
        print(f"🔍 分析图像: {Path(image_path).name}")
        
        # 读取和预处理图像
        gray_image = self._load_image(image_path)
        if gray_image is None:
            return None
        
        # 执行分析（根据参数决定是否包含圆形热点法）
        analysis_result = self._gradient_magnitude_analysis(gray_image, enable_circular)
        
        # 创建输出目录
        if output_dir is None:
            output_dir = Path(image_path).parent / "gradient_visualization"
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 生成可视化结果
        self._create_visualizations(gray_image, analysis_result, image_path, output_dir, enable_circular)
        
        # 保存详细报告
        self._save_analysis_report(analysis_result, image_path, output_dir, enable_circular)
        
        return analysis_result
    
    def _load_image(self, image_path):
        """加载并转换图像为灰度"""
        color_image = cv2.imread(str(image_path))
        if color_image is None:
            print(f"❌ 无法读取图像: {image_path}")
            return None
        
        if len(color_image.shape) == 3:
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = color_image
        
        print(f"📐 图像尺寸: {gray_image.shape}")
        print(f"🎯 窗口大小: {self.roi_size}x{self.roi_size}")
        
        return gray_image
    
    def _gradient_magnitude_analysis(self, gray_image, enable_circular=False):
        """执行梯度幅值分析，可选择是否包含圆形热点法对比"""
        h, w = gray_image.shape
        
        # 初始化梯度法结果存储
        gradient_results = []
        gradient_score_map = np.zeros((h - self.roi_size + 1, w - self.roi_size + 1))
        
        # 梯度法最佳结果
        grad_best_score = -1
        grad_best_center = (0, 0)
        grad_best_window = None
        
        # 如果启用圆形热点法，初始化相关变量
        if enable_circular:
            circular_results = []
            circular_score_map = np.zeros((h - self.roi_size + 1, w - self.roi_size + 1))
            circ_best_score = -1
            circ_best_center = (0, 0)
            circ_best_window = None
            print(f"🔄 开始双方法对比分析 {(h - self.roi_size + 1) * (w - self.roi_size + 1)} 个窗口...")
        else:
            print(f"🔄 开始梯度法分析 {(h - self.roi_size + 1) * (w - self.roi_size + 1)} 个窗口...")
        
        # 滑动窗口分析
        for y in range(self.half_size, h - self.half_size):
            for x in range(self.half_size, w - self.half_size):
                # 提取ROI窗口
                window = gray_image[y-self.half_size:y+self.half_size+1, 
                                  x-self.half_size:x+self.half_size+1]
                
                if window.shape != (self.roi_size, self.roi_size):
                    continue
                
                # 方法1: 梯度幅值法
                grad_result = self._calculate_gradient_features(window)
                window_mean = np.mean(window.astype(np.float32))
                grad_mean = grad_result['gradient_mean']
                grad_score = window_mean + (grad_mean * 0.3)
                
                gradient_results.append({
                    'center': (x, y),
                    'window': window.copy(),
                    'score': grad_score,
                    'window_mean': window_mean,
                    'gradient_mean': grad_mean,
                    'method': 'gradient'
                })
                gradient_score_map[y-self.half_size, x-self.half_size] = grad_score
                
                if grad_score > grad_best_score:
                    grad_best_score = grad_score
                    grad_best_center = (x, y)
                    grad_best_window = {
                        'center': (x, y),
                        'window': window.copy(),
                        'window_mean': window_mean,
                        'gradient_mean': grad_mean,
                        'score': grad_score,
                        'gradient_x': grad_result['grad_x'],
                        'gradient_y': grad_result['grad_y'],
                        'gradient_magnitude': grad_result['gradient_magnitude']
                    }
                
                # 方法2: 圆形热点法 (仅在启用时执行)
                if enable_circular:
                    circ_score = self._calculate_circular_hotspot_score(window)
                    
                    circular_results.append({
                        'center': (x, y),
                        'window': window.copy(),
                        'score': circ_score,
                        'method': 'circular'
                    })
                    circular_score_map[y-self.half_size, x-self.half_size] = circ_score
                    
                    if circ_score > circ_best_score:
                        circ_best_score = circ_score
                        circ_best_center = (x, y)
                        circ_best_window = {
                            'center': (x, y),
                            'window': window.copy(),
                            'score': circ_score,
                            'method': 'circular'
                        }
        
        # 输出结果信息
        if enable_circular:
            print(f"✅ 双方法分析完成")
            print(f"   梯度法最佳: {grad_best_center}，评分: {grad_best_score:.3f}")
            print(f"   圆形法最佳: {circ_best_center}，评分: {circ_best_score:.3f}")
        else:
            print(f"✅ 梯度法分析完成")
            print(f"   最佳位置: {grad_best_center}，评分: {grad_best_score:.3f}")
        
        # 构建返回结果
        result = {
            # 梯度法结果
            'gradient_best_center': grad_best_center,
            'gradient_best_score': grad_best_score,
            'gradient_best_window': grad_best_window,
            'gradient_score_map': gradient_score_map,
            'gradient_results': gradient_results,
            
            # 保持向后兼容
            'best_center': grad_best_center,
            'best_score': grad_best_score,
            'best_window': grad_best_window,
            'score_map': gradient_score_map,
            'all_windows': gradient_results,
            'image_shape': gray_image.shape,
            'enable_circular': enable_circular
        }
        
        # 添加圆形热点法结果（如果启用）
        if enable_circular:
            result.update({
                'circular_best_center': circ_best_center,
                'circular_best_score': circ_best_score,
                'circular_best_window': circ_best_window,
                'circular_score_map': circular_score_map,
                'circular_results': circular_results
            })
        
        return result
    
    def _calculate_gradient_features(self, window):
        """计算窗口的梯度特征"""
        # 确保是浮点类型以避免溢出
        window_float = window.astype(np.float32)
        
        # 计算梯度
        grad_x = np.gradient(window_float, axis=1)
        grad_y = np.gradient(window_float, axis=0)
        
        # 计算梯度幅值
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 计算平均梯度幅值
        gradient_mean = np.mean(gradient_magnitude)
        
        return {
            'grad_x': grad_x,
            'grad_y': grad_y,
            'gradient_magnitude': gradient_magnitude,
            'gradient_mean': gradient_mean
        }
    
    def _calculate_circular_hotspot_score(self, window):
        """计算圆形热点评分"""
        h, w = window.shape
        center_x, center_y = w // 2, h // 2
        
        # 生成圆形权重掩码 (中心权重高，边缘权重低)
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_radius = min(center_x, center_y)
        
        # 创建圆形权重：中心为1，边缘趋近于0
        weights = np.exp(-distances / (max_radius * 0.5))
        weights = weights / np.sum(weights)  # 归一化
        
        # 加权平均强度
        weighted_intensity = np.sum(window.astype(np.float32) * weights)
        
        # 计算中心区域与外围区域的对比度
        center_mask = distances <= max_radius * 0.3
        outer_mask = (distances > max_radius * 0.6) & (distances <= max_radius)
        
        if np.sum(center_mask) > 0 and np.sum(outer_mask) > 0:
            center_mean = np.mean(window[center_mask])
            outer_mean = np.mean(window[outer_mask])
            contrast = center_mean - outer_mean
        else:
            contrast = 0
        
        # 综合评分：加权强度 + 对比度增强
        score = weighted_intensity + (contrast * 0.2)
        
        return score
    
    def _create_visualizations(self, gray_image, result, image_path, output_dir, enable_circular=False):
        """创建可视化结果：根据参数决定是否包含方法对比"""
        image_name = Path(image_path).stem
        
        if enable_circular:
            # 双方法对比模式：1x2布局
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # 原始图像 + 两种方法对比
            axes[0].imshow(gray_image, cmap='gray')
            axes[0].set_title('Original Image with Method Comparison', fontsize=14, fontweight='bold')
            
            # 绘制梯度方法结果（红色）
            grad_x, grad_y = result['gradient_best_center']
            rect1 = plt.Rectangle((grad_x - self.half_size, grad_y - self.half_size), 
                                self.roi_size, self.roi_size, 
                                fill=False, edgecolor='red', linewidth=2, label='Gradient Method')
            axes[0].add_patch(rect1)
            axes[0].plot(grad_x, grad_y, 'r+', markersize=15, markeredgewidth=3)
            axes[0].text(grad_x + 3, grad_y - 3, f'Grad({grad_x},{grad_y})', 
                        color='red', fontweight='bold', fontsize=10)
            
            # 绘制圆形热点方法结果（蓝色）
            circ_x, circ_y = result['circular_best_center']
            rect2 = plt.Rectangle((circ_x - self.half_size, circ_y - self.half_size), 
                                self.roi_size, self.roi_size, 
                                fill=False, edgecolor='blue', linewidth=2, label='Circular Hotspot Method')
            axes[0].add_patch(rect2)
            axes[0].plot(circ_x, circ_y, 'b*', markersize=15, markeredgewidth=2)
            axes[0].text(circ_x + 3, circ_y + 3, f'Circ({circ_x},{circ_y})', 
                        color='blue', fontweight='bold', fontsize=10)
            
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # RGB混合热力图：梯度法(红色) + 圆形法(蓝色)
            grad_heatmap = result['gradient_score_map']
            circ_heatmap = result['circular_score_map']
            
            # 归一化到0-1
            grad_norm = (grad_heatmap - grad_heatmap.min()) / (grad_heatmap.max() - grad_heatmap.min() + 1e-8)
            circ_norm = (circ_heatmap - circ_heatmap.min()) / (circ_heatmap.max() - circ_heatmap.min() + 1e-8)
            
            # 创建RGB图像
            rgb_heatmap = np.zeros((grad_norm.shape[0], grad_norm.shape[1], 3))
            rgb_heatmap[:, :, 0] = grad_norm    # 红色通道 - 梯度法
            rgb_heatmap[:, :, 2] = circ_norm    # 蓝色通道 - 圆形法
            
            im2 = axes[1].imshow(rgb_heatmap, alpha=0.8)
            axes[1].set_title('RGB Composite Heatmap\n(Red: Gradient, Blue: Circular)', 
                            fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            # 保存图片
            plt.tight_layout()
            save_path = output_dir / f'{image_name}_method_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        else:
            # 仅梯度法模式：简化的单一分析图
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # 原始图像 + 梯度方法结果
            axes[0].imshow(gray_image, cmap='gray')
            axes[0].set_title('Original Image with Gradient Analysis', fontsize=14, fontweight='bold')
            
            # 绘制梯度方法结果
            grad_x, grad_y = result['best_center']
            rect = plt.Rectangle((grad_x - self.half_size, grad_y - self.half_size), 
                               self.roi_size, self.roi_size, 
                               fill=False, edgecolor='red', linewidth=2)
            axes[0].add_patch(rect)
            axes[0].plot(grad_x, grad_y, 'r+', markersize=15, markeredgewidth=3)
            axes[0].text(grad_x + 3, grad_y - 3, f'Best({grad_x},{grad_y})', 
                        color='red', fontweight='bold', fontsize=10)
            axes[0].grid(True, alpha=0.3)
            
            # 梯度法热力图
            heatmap = result['score_map']
            im = axes[1].imshow(heatmap, cmap='hot', alpha=0.8)
            axes[1].set_title('Gradient Magnitude Score Heatmap', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            # 添加颜色条
            plt.colorbar(im, ax=axes[1], shrink=0.8)
            
            # 保存图片
            plt.tight_layout()
            save_path = output_dir / f'{image_name}_gradient_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📊 可视化结果已保存到: {output_dir}")
    
    def _save_analysis_report(self, result, image_path, output_dir, enable_circular=False):
        """保存详细分析报告"""
        image_name = Path(image_path).stem
        report_path = output_dir / f'{image_name}_analysis_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            if enable_circular:
                # 双方法报告
                f.write("🔍 梯度幅值双方法对比分析报告\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"📁 图像文件: {Path(image_path).name}\n")
                f.write(f"📐 图像尺寸: {result['image_shape']}\n")
                f.write(f"🎯 ROI窗口大小: {self.roi_size}x{self.roi_size}\n")
                f.write(f"📊 分析窗口总数: {len(result['gradient_results'])}\n\n")
                
                # 梯度法结果
                f.write("🔶 梯度法分析结果:\n")
                f.write(f"  🎯 最佳位置: {result['gradient_best_center']}\n")
                f.write(f"  📊 最佳评分: {result['gradient_best_score']:.6f}\n")
                if result['gradient_best_window']:
                    best = result['gradient_best_window']
                    f.write(f"  📈 灰度均值: {best.get('window_mean', 0):.3f}\n")
                    f.write(f"  📊 梯度均值: {best.get('gradient_mean', 0):.3f}\n")
                
                # 圆形热点法结果
                f.write("\n🔵 圆形热点法分析结果:\n")
                f.write(f"  🎯 最佳位置: {result['circular_best_center']}\n")
                f.write(f"  📊 最佳评分: {result['circular_best_score']:.6f}\n")
                
                # 方法对比
                f.write("\n⚖️ 方法对比:\n")
                if result['gradient_best_center'] == result['circular_best_center']:
                    f.write("  ✅ 两种方法检测到相同位置\n")
                else:
                    f.write("  ⚠️ 两种方法检测到不同位置\n")
                
                grad_score = result['gradient_best_score']
                circ_score = result['circular_best_score']
                if grad_score > circ_score:
                    f.write(f"  🏆 梯度法评分更高 (+{grad_score - circ_score:.6f})\n")
                else:
                    f.write(f"  🏆 圆形法评分更高 (+{circ_score - grad_score:.6f})\n")
                    
            else:
                # 仅梯度法报告
                f.write("🔍 梯度幅值分析报告\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"📁 图像文件: {Path(image_path).name}\n")
                f.write(f"📐 图像尺寸: {result['image_shape']}\n")
                f.write(f"🎯 ROI窗口大小: {self.roi_size}x{self.roi_size}\n")
                f.write(f"📊 分析窗口总数: {len(result['all_windows'])}\n\n")
                
                # 最佳结果
                best = result['best_window']
                f.write("🏆 最佳结果:\n")
                f.write(f"  中心位置: {result['best_center']}\n")
                f.write(f"  综合评分: {result['best_score']:.6f}\n")
                f.write(f"  灰度均值: {best['window_mean']:.3f}\n")
                f.write(f"  梯度均值: {best['gradient_mean']:.3f}\n\n")
                
                # 评分公式
                f.write("📊 评分公式:\n")
                f.write("  Score = 灰度均值 + 0.3 × 梯度幅值均值\n")
                f.write(f"  Score = {best['window_mean']:.3f} + 0.3 × {best['gradient_mean']:.3f}\n")
                f.write(f"  Score = {result['best_score']:.6f}\n\n")
                
                # 最佳窗口灰度矩阵
                f.write("🎯 最佳窗口灰度矩阵:\n")
                window = best['window']
                for i in range(self.roi_size):
                    row_str = '  '.join([f'{val:3d}' for val in window[i]])
                    f.write(f"  [{row_str}]\n")
                f.write("\n")
                
                # Top 10 结果
                top_10 = sorted(result['all_windows'], key=lambda x: x['score'], reverse=True)[:10]
                f.write("🔝 Top 10 候选位置:\n")
                f.write("  排名   中心位置    评分      灰度均值  梯度均值\n")
                f.write("  " + "-" * 50 + "\n")
                for i, window in enumerate(top_10):
                    f.write(f"  {i+1:2d}    {window['center']}   {window['score']:8.3f}  "
                           f"{window['window_mean']:7.1f}  {window['gradient_mean']:7.1f}\n")
        
        print(f"📄 详细报告已保存: {report_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='梯度幅值法ROI可视化分析器')
    parser.add_argument('--image', '-i', type=str, required=True,
                       help='输入图像路径')
    parser.add_argument('--roi-size', type=int, default=5,
                       help='ROI窗口大小 (默认: 5)')
    parser.add_argument('--output', '-o', type=str,
                       help='输出目录 (默认: 图像同目录下的gradient_visualization)')
    parser.add_argument('--enable-circular', action='store_true',
                       help='启用圆形热点法对比分析 (默认: 仅使用梯度法)')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not Path(args.image).exists():
        print(f"❌ 图像文件不存在: {args.image}")
        return 1
    
    try:
        # 创建分析器并执行分析
        analyzer = GradientMagnitudeAnalyzer(roi_size=args.roi_size)
        result = analyzer.analyze_image(args.image, args.output, enable_circular=args.enable_circular)
        
        if result:
            print(f"\n🎉 分析完成！")
            if args.enable_circular:
                print(f"📊 梯度法最佳位置: {result['gradient_best_center']}")
                print(f"🏆 梯度法最高评分: {result['gradient_best_score']:.6f}")
                print(f"📊 圆形法最佳位置: {result['circular_best_center']}")
                print(f"🏆 圆形法最高评分: {result['circular_best_score']:.6f}")
            else:
                print(f"📊 最佳位置: {result['best_center']}")
                print(f"🏆 最高评分: {result['best_score']:.6f}")
            print(f"📁 结果保存在: {args.output or Path(args.image).parent / 'gradient_visualization'}")
        else:
            print("❌ 分析失败")
            return 1
        
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
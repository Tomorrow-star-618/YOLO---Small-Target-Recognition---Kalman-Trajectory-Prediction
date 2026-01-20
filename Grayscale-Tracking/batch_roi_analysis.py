#!/usr/bin/env python3
"""
批量ROI双方法对比分析脚本
处理目录下的所有ROI图片，生成对比分析结果
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import time
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

def analyze_single_image(image_path, roi_size=5, enable_circular=False):
    """分析单张图片"""
    try:
        # 构建命令
        cmd = [
            'python', 'gradient_magnitude_visualizer.py',
            '--image', str(image_path),
            '--roi-size', str(roi_size)
        ]
        
        # 如果启用圆形热点法，添加参数
        if enable_circular:
            cmd.append('--enable-circular')
        
        # 执行分析
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='/home/mingxing/worksapce/ultralytics/Grayscale-Tracking')
        
        if result.returncode == 0:
            # 解析输出获取结果
            output_lines = result.stdout.strip().split('\n')
            grad_best = None
            circ_best = None
            grad_score = None
            circ_score = None
            
            for line in output_lines:
                if '梯度法最佳:' in line:
                    parts = line.split('，')
                    if len(parts) >= 2:
                        pos_part = parts[0].split('(')[1].split(')')[0]
                        grad_best = tuple(map(int, pos_part.split(', ')))
                        score_part = parts[1].split(':')[1].strip()
                        grad_score = float(score_part)
                elif '圆形法最佳:' in line:
                    parts = line.split('，')
                    if len(parts) >= 2:
                        pos_part = parts[0].split('(')[1].split(')')[0]
                        circ_best = tuple(map(int, pos_part.split(', ')))
                        score_part = parts[1].split(':')[1].strip()
                        circ_score = float(score_part)
            
            return {
                'image_path': str(image_path),
                'image_name': image_path.name,
                'status': 'success',
                'gradient_position': grad_best,
                'gradient_score': grad_score,
                'circular_position': circ_best,
                'circular_score': circ_score,
                'position_match': grad_best == circ_best if grad_best and circ_best else False,
                'score_difference': circ_score - grad_score if grad_score and circ_score else None,
                'better_method': 'circular' if (circ_score and grad_score and circ_score > grad_score) else 'gradient'
            }
        else:
            return {
                'image_path': str(image_path),
                'image_name': image_path.name,
                'status': 'failed',
                'error': result.stderr
            }
            
    except Exception as e:
        return {
            'image_path': str(image_path),
            'image_name': image_path.name,
            'status': 'error',
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description='批量ROI双方法对比分析')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='输入ROI图片目录')
    parser.add_argument('--roi-size', type=int, default=5,
                       help='ROI窗口大小')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='并行处理进程数')
    parser.add_argument('--enable-circular', action='store_true',
                       help='启用圆形热点法对比分析')
    parser.add_argument('--output', type=str,
                       help='结果汇总输出目录')
    
    args = parser.parse_args()
    
    # 检查输入目录
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        return 1
    
    # 设置输出目录
    output_dir = Path(args.output) if args.output else input_dir / 'batch_analysis_results'
    output_dir.mkdir(exist_ok=True)
    
    # 查找所有PNG图片
    image_files = list(input_dir.glob('*.png'))
    if not image_files:
        print(f"❌ 在目录 {input_dir} 中未找到PNG图片")
        return 1
    
    print(f"🔍 找到 {len(image_files)} 张ROI图片")
    print(f"📊 ROI窗口大小: {args.roi_size}x{args.roi_size}")
    print(f"⚡ 并行进程数: {args.max_workers}")
    print(f"� 分析模式: {'双方法对比' if args.enable_circular else '仅梯度法'}")
    print(f"�📁 结果输出到: {output_dir}")
    print("-" * 60)
    
    # 记录开始时间
    start_time = time.time()
    results = []
    
    # 并行处理
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # 提交所有任务
        future_to_image = {
            executor.submit(analyze_single_image, img_path, args.roi_size, args.enable_circular): img_path 
            for img_path in image_files
        }
        
        # 收集结果
        completed = 0
        for future in as_completed(future_to_image):
            result = future.result()
            results.append(result)
            completed += 1
            
            # 显示进度
            if completed % 10 == 0 or completed == len(image_files):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(image_files) - completed) / rate if rate > 0 else 0
                print(f"📈 进度: {completed}/{len(image_files)} ({completed/len(image_files)*100:.1f}%) "
                      f"速度: {rate:.1f}张/秒 预计剩余: {eta:.0f}秒")
    
    # 统计结果
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']
    
    print("\n" + "="*60)
    print("📊 批量分析结果统计:")
    print(f"✅ 成功处理: {len(successful)}/{len(results)} 张")
    print(f"❌ 处理失败: {len(failed)}/{len(results)} 张")
    
    if successful:
        # 方法对比统计
        position_matches = sum(1 for r in successful if r.get('position_match', False))
        gradient_better = sum(1 for r in successful if r.get('better_method') == 'gradient')
        circular_better = sum(1 for r in successful if r.get('better_method') == 'circular')
        
        print(f"🎯 位置匹配: {position_matches}/{len(successful)} ({position_matches/len(successful)*100:.1f}%)")
        print(f"🔶 梯度法更优: {gradient_better}/{len(successful)} ({gradient_better/len(successful)*100:.1f}%)")
        print(f"🔵 圆形法更优: {circular_better}/{len(successful)} ({circular_better/len(successful)*100:.1f}%)")
        
        # 计算平均分数差异
        score_diffs = [r['score_difference'] for r in successful if r.get('score_difference') is not None]
        if score_diffs:
            avg_diff = sum(score_diffs) / len(score_diffs)
            print(f"📈 平均分数差异: {avg_diff:.3f} (圆形法 - 梯度法)")
    
    # 保存详细结果
    if successful:
        csv_path = output_dir / 'batch_analysis_summary.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            if successful:
                fieldnames = successful[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(successful)
        print(f"📄 详细结果已保存: {csv_path}")
    
    # 保存失败列表
    if failed:
        failed_path = output_dir / 'failed_images.txt'
        with open(failed_path, 'w') as f:
            f.write("处理失败的图片列表:\n")
            f.write("-" * 30 + "\n")
            for item in failed:
                f.write(f"{item['image_name']}: {item.get('error', 'unknown error')}\n")
        print(f"⚠️ 失败列表已保存: {failed_path}")
    
    total_time = time.time() - start_time
    print(f"⏱️ 总耗时: {total_time:.1f}秒")
    print("🎉 批量分析完成!")
    
    return 0

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)  # 避免multiprocessing问题
    sys.exit(main())
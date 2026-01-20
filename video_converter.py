#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
视频格式转换工具
支持将各种视频格式转换为MP4格式
"""

import cv2
import os
from pathlib import Path
import argparse
import time

def convert_video_to_mp4(input_path, output_path=None, fps=None, quality=None):
    """
    将视频转换为MP4格式
    
    Args:
        input_path: 输入视频路径
        output_path: 输出MP4路径（可选，默认自动生成）
        fps: 输出帧率（可选，默认保持原帧率）
        quality: 视频质量 ('high', 'medium', 'low')（可选，默认medium）
    
    Returns:
        bool: 转换是否成功
    """
    input_path = Path(input_path)
    
    # 检查输入文件是否存在
    if not input_path.exists():
        print(f"❌ 输入文件不存在: {input_path}")
        return False
    
    # 生成输出路径
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_converted.mp4"
    else:
        output_path = Path(output_path)
        # 确保输出文件扩展名为.mp4
        if output_path.suffix.lower() != '.mp4':
            output_path = output_path.with_suffix('.mp4')
    
    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📹 开始转换视频:")
    print(f"   输入: {input_path}")
    print(f"   输出: {output_path}")
    
    # 打开输入视频
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"❌ 无法打开输入视频: {input_path}")
        return False
    
    try:
        # 获取视频属性
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 设置输出帧率
        output_fps = fps if fps is not None else original_fps
        if output_fps <= 0:
            output_fps = 30  # 默认30fps
        
        print(f"   原始分辨率: {width}x{height}")
        print(f"   原始帧率: {original_fps:.1f} FPS")
        print(f"   输出帧率: {output_fps:.1f} FPS")
        print(f"   总帧数: {total_frames}")
        
        # 设置视频编码器和质量
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # 根据质量调整参数
        if quality == 'high':
            # 高质量：保持原分辨率
            out_width, out_height = width, height
        elif quality == 'low':
            # 低质量：降低分辨率
            out_width = width // 2
            out_height = height // 2
        else:
            # 中等质量：保持原分辨率
            out_width, out_height = width, height
        
        print(f"   输出分辨率: {out_width}x{out_height}")
        
        # 创建视频写入器
        out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (out_width, out_height))
        
        if not out.isOpened():
            print(f"❌ 无法创建输出视频文件: {output_path}")
            return False
        
        # 转换视频
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 调整帧尺寸（如果需要）
            if (out_width, out_height) != (width, height):
                frame = cv2.resize(frame, (out_width, out_height))
            
            # 写入帧
            out.write(frame)
            frame_count += 1
            
            # 显示进度
            if frame_count % 30 == 0 or frame_count == total_frames:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed if elapsed > 0 else 0
                
                # 创建进度条
                bar_length = 30
                filled_length = int(bar_length * progress / 100)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                print(f"   进度: [{bar}] {progress:.1f}% ({frame_count}/{total_frames}), "
                      f"用时: {elapsed:.1f}s, 处理FPS: {fps_current:.1f}")
        
        processing_time = time.time() - start_time
        
        print(f"✅ 视频转换完成!")
        print(f"   处理帧数: {frame_count}")
        print(f"   处理时间: {processing_time:.2f}s")
        print(f"   平均处理FPS: {frame_count / processing_time:.2f}")
        print(f"   输出文件: {output_path}")
        
        # 检查输出文件大小
        input_size = input_path.stat().st_size / (1024*1024)  # MB
        output_size = output_path.stat().st_size / (1024*1024)  # MB
        print(f"   文件大小: {input_size:.1f}MB -> {output_size:.1f}MB "
              f"({'压缩' if output_size < input_size else '增大'} "
              f"{abs(output_size-input_size)/input_size*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ 转换过程中出现错误: {e}")
        return False
        
    finally:
        cap.release()
        if 'out' in locals():
            out.release()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='视频格式转换工具')
    parser.add_argument('input', type=str, help='输入视频文件路径')
    parser.add_argument('--output', '-o', type=str, help='输出MP4文件路径（可选）')
    parser.add_argument('--fps', type=float, help='输出帧率（可选，默认保持原帧率）')
    parser.add_argument('--quality', choices=['high', 'medium', 'low'], 
                       default='medium', help='视频质量 (默认: medium)')
    
    args = parser.parse_args()
    
    # 转换视频
    success = convert_video_to_mp4(
        args.input, 
        args.output, 
        args.fps, 
        args.quality
    )
    
    if success:
        print("\n🎉 转换成功完成！")
        return 0
    else:
        print("\n❌ 转换失败！")
        return 1

if __name__ == "__main__":
    # 直接转换指定的视频文件
    input_video = "/home/mingxing/worksapce/ultralytics/runs/detect/predict2/complex-background.avi"
    output_video = "/home/mingxing/worksapce/ultralytics/runs/detect/predict2/complex-background.mp4"
    
    print("🚀 自动转换模式")
    success = convert_video_to_mp4(input_video, output_video, quality='medium')
    
    if success:
        print("\n🎉 转换成功完成！")
        print("💡 你也可以通过命令行使用此工具:")
        print(f"   python {__file__} input_video.avi --output output_video.mp4 --fps 30 --quality high")
    else:
        print("\n❌ 转换失败！")
        
        # 如果自动模式失败，提供命令行模式
        print("💡 你可以尝试命令行模式:")
        print(f"   python {__file__} {input_video}")

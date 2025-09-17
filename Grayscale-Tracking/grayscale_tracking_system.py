#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于局部灰度值的目标追踪系统
结合YOLO检测和灰度区域预测进行目标追踪

使用方法 / Usage:
-----------------
python grayscale_tracking_system.py --video <视频路径> [其他参数]

必需参数 / Required Arguments:
  --video, -v <PATH>         输入视频文件路径 (必填)

可选参数 / Optional Arguments:
  --model, -m <PATH>         YOLO模型文件路径
                            默认: small_target_detection/yolov8_small_aircraft/weights/best.pt
  --output, -o <PATH>        输出视频文件路径 (可选，默认自动生成)
  --template, -t <STR>       局部灰度值模板 (可选，格式为数组字符串)
  --test <START,END>         测试模式，指定强制丢失帧范围 (例如: 50,100)
  --save-process             保存处理过程中的ROI图像和灰度矩阵数据

使用示例 / Examples:
------------------
# 基本使用 - 使用默认模型处理视频
python grayscale_tracking_system.py --video vedio/test.mp4

# 指定模型和保存处理过程
python grayscale_tracking_system.py --video vedio/test.mp4 --model yolo11x.pt --save-process

# 测试模式 - 在指定帧范围强制目标丢失
python grayscale_tracking_system.py --video vedio/test.mp4 --test 30,80

# 完整参数示例
python grayscale_tracking_system.py \
    --video vedio/aircraft.mp4 \
    --model yolo11x.pt \
    --output results/tracked_aircraft.mp4 \
    --test 50,150 \
    --save-process

输出说明 / Output:
----------------
程序将在 Grayscale-Tracking/runs/ 目录下自动创建以下结构:
  视频名_日期时间/
  ├── output-video/          # 输出的追踪视频
  └── process/              # 处理过程文件 (如果启用 --save-process)
      ├── roi_patches/      # ROI图像块
      └── grayscale_data/   # 灰度矩阵数据和对比图

功能特性 / Features:
------------------
✓ YOLO目标检测与灰度预测结合
✓ GPU加速处理 (自动检测CUDA)  
✓ 丢失后持续预测直到重新检测
✓ 实时进度条和FPS显示
✓ 自动目录结构管理
✓ 可视化处理过程保存
✓ 测试模式支持
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
from pathlib import Path
import time
from collections import defaultdict
import argparse

class GrayscaleTracker:
    """基于灰度值的目标追踪器"""
    
    def __init__(self, model_path, local_grayscale_template=None, save_process=False):
        """初始化追踪器
        
        Args:
            model_path: YOLO模型路径
            local_grayscale_template: 局部灰度值模板 (25x25 numpy数组)
            save_process: 是否保存处理过程图像
        """
        self.model = YOLO(model_path)
        self.local_grayscale_template = local_grayscale_template
        self.save_process = save_process
        
        # 追踪参数
        self.roi_size = 40  # ROI区域大小
        self.search_radius = 50  # 搜索半径
        self.min_prediction_confidence = 0.1  # 最低预测置信度阈值
        
        # GPU加速设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            print(f"🚀 GPU加速已启用: {torch.cuda.get_device_name()}")
        else:
            print("⚠️ GPU不可用，使用CPU处理")
        
        # 追踪状态
        self.tracks = defaultdict(dict)  # 轨迹信息
        self.track_id_counter = 0
        self.video_fps = 30  # 视频帧率，处理视频时会更新
        
        # 创建处理过程保存目录
        if self.save_process:
            # 暂时设置默认目录，实际目录在process_video中创建
            self.process_dir = None
            self.roi_patches_dir = None
            self.grayscale_data_dir = None
        
        print(f"✅ 初始化灰度追踪系统")
        print(f"   模型路径: {model_path}")
        print(f"   ROI大小: {self.roi_size}x{self.roi_size}")
        print(f"   最低预测置信度: {self.min_prediction_confidence}")
        print(f"   保存过程: {'是' if save_process else '否'}")
        print(f"   追踪策略: 持续预测直到重新检测到目标")
        print(f"   GPU加速: {'启用' if self.use_gpu else '禁用'} ({'CUDA' if self.use_gpu else 'CPU'})")
    
    def create_results_directory(self, video_path):
        """创建结果目录，基于视频名称和当前日期时间
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            results_dir: 创建的结果目录路径
        """
        import datetime
        
        # 获取视频文件名（不含扩展名）
        video_name = Path(video_path).stem
        
        # 获取当前日期时间
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        
        # 创建结果目录名
        results_dir_name = f"{video_name}_{timestamp}"
        
        # 在Grayscale-Tracking/runs目录下创建结果目录
        script_dir = Path(__file__).parent
        runs_dir = script_dir / "runs"
        runs_dir.mkdir(exist_ok=True)  # 确保runs目录存在
        results_dir = runs_dir / results_dir_name
        
        # 创建主目录和子目录
        results_dir.mkdir(exist_ok=True)
        
        output_video_dir = results_dir / "output-video"
        process_dir = results_dir / "process"
        
        output_video_dir.mkdir(exist_ok=True)
        process_dir.mkdir(exist_ok=True)
        
        # 如果需要保存处理过程，创建子目录
        if self.save_process:
            self.process_dir = process_dir
            self.roi_patches_dir = process_dir / "roi_patches"
            self.grayscale_data_dir = process_dir / "grayscale_data"
            self.roi_patches_dir.mkdir(exist_ok=True)
            self.grayscale_data_dir.mkdir(exist_ok=True)
        
        print(f"📁 创建结果目录: {results_dir}")
        print(f"   - 输出视频: {output_video_dir}")
        if self.save_process:
            print(f"   - 处理过程: {process_dir}")
        
        return results_dir, output_video_dir, process_dir

    def save_process_images(self, frame, track_id, frame_id, roi_center, roi_data, prediction_type="gradient", 
                           last_detection_info=None):
        """保存处理过程中的图像和数据，包含丢失前后的对比
        
        Args:
            frame: 原始帧
            track_id: 轨迹ID
            frame_id: 帧ID
            roi_center: ROI中心位置 (x, y) 在整个图像中的位置
            roi_data: ROI区域数据
            prediction_type: 预测类型 ("gradient" 或 "template")
            last_detection_info: 最后一次检测的信息 {"center": (x,y), "roi": np.array, "frame_id": int, "confidence": float}
        """
        if not self.save_process:
            return
        
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端，避免Qt依赖问题
        import matplotlib.pyplot as plt
        
        # 计算视频秒数
        video_seconds = frame_id / self.video_fps
        
        # ROI小方块内的中心位置 (相对于ROI区域的中心)
        roi_local_center = (roi_data.shape[1] // 2, roi_data.shape[0] // 2)
        
        # 1. 保存ROI方块图像 - 命名：秒数+帧数+中心位置
        roi_filename = f"{video_seconds:.1f}s_f{frame_id:04d}_center{roi_center[0]}-{roi_center[1]}_roi.png"
        roi_path = self.roi_patches_dir / roi_filename
        
        # 保存原始ROI图像
        cv2.imwrite(str(roi_path), roi_data)
        
        # 2. 保存灰度矩阵数据图像 - 支持对比显示
        data_filename = f"{video_seconds:.1f}s_f{frame_id:04d}_gray_center{roi_center[0]}-{roi_center[1]}_comparison.png"
        data_path = self.grayscale_data_dir / data_filename
        
        # 创建对比图像
        if last_detection_info is not None:
            # 有最后检测信息，创建对比图
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # 上排：最后检测的ROI
            last_roi = last_detection_info["roi"]
            last_center = last_detection_info["center"]
            last_frame_id = last_detection_info["frame_id"]
            last_confidence = last_detection_info["confidence"]
            last_seconds = last_frame_id / self.video_fps
            
            axes[0, 0].imshow(last_roi, cmap='gray')
            axes[0, 0].set_title(f'Last Detection ROI\nFrame {last_frame_id} ({last_seconds:.1f}s)\nConf: {last_confidence:.3f}')
            axes[0, 0].plot(last_roi.shape[1]//2, last_roi.shape[0]//2, 'r+', markersize=10, markeredgewidth=2)
            axes[0, 0].text(last_roi.shape[1]//2, last_roi.shape[0]//2 + 3, f'({last_center[0]},{last_center[1]})', 
                          ha='center', va='top', color='red', fontsize=8, weight='bold')
            
            axes[0, 1].imshow(last_roi, cmap='hot', interpolation='nearest')
            axes[0, 1].set_title(f'Last Detection Heatmap')
            im1 = axes[0, 1].imshow(last_roi, cmap='hot')
            plt.colorbar(im1, ax=axes[0, 1])
            
            axes[0, 2].contour(last_roi, levels=10)
            axes[0, 2].set_title(f'Last Detection Contours')
            
            # 下排：当前预测的ROI
            axes[1, 0].imshow(roi_data, cmap='gray')
            axes[1, 0].set_title(f'Current Prediction ROI\nFrame {frame_id} ({video_seconds:.1f}s)\n{prediction_type.title()}')
            axes[1, 0].plot(roi_data.shape[1]//2, roi_data.shape[0]//2, 'r+', markersize=10, markeredgewidth=2)
            axes[1, 0].text(roi_data.shape[1]//2, roi_data.shape[0]//2 + 3, f'({roi_center[0]},{roi_center[1]})', 
                          ha='center', va='top', color='red', fontsize=8, weight='bold')
            
            axes[1, 1].imshow(roi_data, cmap='hot', interpolation='nearest')
            axes[1, 1].set_title(f'Current Prediction Heatmap')
            im2 = axes[1, 1].imshow(roi_data, cmap='hot')
            plt.colorbar(im2, ax=axes[1, 1])
            
            axes[1, 2].contour(roi_data, levels=10)
            axes[1, 2].set_title(f'Current Prediction Contours')
            
            plt.suptitle(f'Track {track_id} - Detection vs Prediction Comparison\n'
                        f'Lost Frames: {frame_id - last_frame_id}', fontsize=14)
        else:
            # 没有最后检测信息，使用原来的布局
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            
            axes[0, 0].imshow(roi_data, cmap='gray')
            axes[0, 0].set_title(f'ROI Image ({roi_data.shape[0]}x{roi_data.shape[1]})')
            axes[0, 0].plot(roi_data.shape[1]//2, roi_data.shape[0]//2, 'r+', markersize=10, markeredgewidth=2)
            axes[0, 0].text(roi_data.shape[1]//2, roi_data.shape[0]//2 + 3, f'({roi_center[0]},{roi_center[1]})', 
                          ha='center', va='top', color='red', fontsize=8, weight='bold')
            
            axes[0, 1].imshow(roi_data, cmap='hot', interpolation='nearest')
            axes[0, 1].set_title('Grayscale Heatmap')
            im = axes[0, 1].imshow(roi_data, cmap='hot')
            plt.colorbar(im, ax=axes[0, 1])
            
            axes[1, 0].contour(roi_data, levels=10)
            axes[1, 0].set_title('Grayscale Contours')
            
            axes[1, 1].axis('off')
            # 显示统计信息
            stats_text = f"""Statistics:
Shape: {roi_data.shape}
Min: {np.min(roi_data)}
Max: {np.max(roi_data)}
Mean: {np.mean(roi_data):.1f}
Std: {np.std(roi_data):.1f}
Center: {roi_center}"""
            axes[1, 1].text(0.1, 0.9, stats_text, 
                           transform=axes[1, 1].transAxes, fontsize=10, verticalalignment='top')
            
            plt.suptitle(f'Track {track_id} - Frame {frame_id} - {prediction_type.title()} Prediction')
        
        plt.tight_layout()
        plt.savefig(data_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. 保存数值数据到txt文件 - 命名：秒数+帧数+灰度中心位置
        txt_filename = f"{video_seconds:.1f}s_f{frame_id:04d}_gray_local{roi_local_center[0]}-{roi_local_center[1]}_global{roi_center[0]}-{roi_center[1]}_matrix.txt"
        txt_path = self.grayscale_data_dir / txt_filename
        
        with open(txt_path, 'w') as f:
            f.write(f"Track ID: {track_id}\n")
            f.write(f"Frame ID: {frame_id}\n")
            f.write(f"Video Time: {video_seconds:.1f}s\n")
            f.write(f"Prediction Type: {prediction_type}\n")
            f.write(f"ROI Center (Global): {roi_center}\n")
            f.write(f"ROI Center (Local): {roi_local_center}\n")
            f.write(f"ROI Shape: {roi_data.shape}\n")
            f.write(f"Min Value: {np.min(roi_data)}\n")
            f.write(f"Max Value: {np.max(roi_data)}\n")
            f.write(f"Mean Value: {np.mean(roi_data):.2f}\n")
            f.write(f"Std Value: {np.std(roi_data):.2f}\n")
            f.write(f"\nGrayscale Matrix:\n")
            
            for i, row in enumerate(roi_data):
                row_str = ' '.join([f'{val:3d}' for val in row])
                f.write(f"Row {i:2d}: [{row_str}]\n")
        
        print(f"💾 保存处理过程: {video_seconds:.1f}s Frame{frame_id} -> {roi_filename}, {data_filename}, {txt_filename}")

    def set_template(self, template):
        """设置局部灰度值模板"""
        if isinstance(template, list):
            template = np.array(template)
        self.local_grayscale_template = template
        print(f"✅ 设置灰度模板: {template.shape}")
    
    def yolo_detect(self, frame, force_loss_frames=None):
        """YOLO目标检测
        
        Args:
            frame: 输入帧
            force_loss_frames: 强制目标丢失的帧范围 (start_frame, end_frame)，用于测试
            
        Returns:
            检测结果列表 [(x1, y1, x2, y2, conf, class_id), ...]
        """
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                for box, conf, cls in zip(boxes, confs, classes):
                    x1, y1, x2, y2 = map(int, box)
                    detections.append((x1, y1, x2, y2, conf, int(cls)))
        
        # 如果设置了强制丢失帧范围，在该范围内返回空检测（用于测试）
        if (force_loss_frames is not None and 
            hasattr(self, 'current_frame_id') and
            force_loss_frames[0] <= self.current_frame_id <= force_loss_frames[1]):
            print(f"🧪 测试模式: 强制目标丢失 (帧 {self.current_frame_id})")
            return []
        
        return detections
    
    def calculate_center(self, x1, y1, x2, y2):
        """计算边界框中心点"""
        return int((x1 + x2) / 2), int((y1 + y2) / 2)
    
    def extract_roi(self, frame, center_x, center_y, size=None):
        """提取ROI区域
        
        Args:
            frame: 输入帧
            center_x, center_y: 中心点坐标
            size: ROI尺寸
            
        Returns:
            roi: ROI区域图像
            roi_coords: ROI坐标 (x1, y1, x2, y2)
        """
        if size is None:
            size = self.roi_size
        
        half_size = size // 2
        h, w = frame.shape[:2]
        
        # 计算ROI边界，确保不越界
        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(w, center_x + half_size)
        y2 = min(h, center_y + half_size)
        
        roi = frame[y1:y2, x1:x2]
        return roi, (x1, y1, x2, y2)
    
    def template_matching(self, frame, last_center, search_radius=None):
        """基于模板匹配的位置预测
        
        Args:
            frame: 当前帧
            last_center: 上一帧中心位置
            search_radius: 搜索半径
            
        Returns:
            predicted_center: 预测的中心位置
            match_score: 匹配得分
        """
        if self.local_grayscale_template is None:
            return last_center, 0.0
        
        if search_radius is None:
            search_radius = self.search_radius
        
        last_x, last_y = last_center
        h, w = frame.shape[:2]
        
        # 将彩色帧转为灰度
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
        
        # 定义搜索区域
        search_x1 = max(0, last_x - search_radius)
        search_y1 = max(0, last_y - search_radius)
        search_x2 = min(w, last_x + search_radius)
        search_y2 = min(h, last_y + search_radius)
        
        search_region = gray_frame[search_y1:search_y2, search_x1:search_x2]
        
        if search_region.size == 0:
            return last_center, 0.0
        
        # 确保模板尺寸合适
        template = self.local_grayscale_template.astype(np.uint8)
        
        # 模板匹配
        result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
        
        if result.size == 0:
            return last_center, 0.0
        
        # 找到最佳匹配位置
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # 转换到原始坐标系
        match_x = search_x1 + max_loc[0] + template.shape[1] // 2
        match_y = search_y1 + max_loc[1] + template.shape[0] // 2
        
        return (match_x, match_y), max_val
    
    def gradient_magnitude_prediction(self, frame, last_center, search_radius=None):
        """基于局部灰度值的位置预测
        
        逻辑：
        1. 以最后检测中心为基准，提取40x40的ROI区域
        2. 在40x40区域内寻找5x5窗口中灰度值最高的位置
        3. 该5x5窗口的中心就是预测的新中心位置
        
        Args:
            frame: 当前帧
            last_center: 上一帧中心位置 (最后检测到的位置)
            search_radius: 搜索半径 (此方法中不使用，保持接口一致)
            
        Returns:
            best_center: 预测的中心位置
            best_score: 最佳匹配得分
        """
        last_x, last_y = last_center
        h, w = frame.shape[:2]
        
        # 将彩色帧转为灰度
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
        
        # 步骤1: 以最后检测中心为基准，提取40x40的ROI区域
        roi_size = self.roi_size  # 40x40
        half_roi = roi_size // 2  # 20
        
        # 计算ROI边界，确保不越界
        roi_x1 = max(0, last_x - half_roi)
        roi_y1 = max(0, last_y - half_roi)
        roi_x2 = min(w, last_x + half_roi)
        roi_y2 = min(h, last_y + half_roi)
        
        # 提取40x40的ROI区域
        roi_40x40 = gray_frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if roi_40x40.size == 0:
            print(f"⚠️ ROI区域为空，返回原位置 {last_center}")
            return last_center, 0.0
        
        # 步骤2: 在40x40区域内寻找5x5窗口中灰度值最高的位置
        window_size = 5
        half_window = window_size // 2  # 2
        
        best_score = -1
        best_local_center = (roi_40x40.shape[1] // 2, roi_40x40.shape[0] // 2)  # 默认中心
        
        # 在40x40区域内滑动5x5窗口
        for y in range(half_window, roi_40x40.shape[0] - half_window):
            for x in range(half_window, roi_40x40.shape[1] - half_window):
                # 提取5x5窗口
                window_5x5 = roi_40x40[y-half_window:y+half_window+1, 
                                     x-half_window:x+half_window+1]
                
                if window_5x5.shape != (window_size, window_size):
                    continue
                
                # 计算5x5窗口的灰度评分（平均灰度值）
                window_mean = np.mean(window_5x5.astype(np.float32))
                
                # 也可以考虑梯度信息增强评分
                grad_x = cv2.Sobel(window_5x5.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(window_5x5.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                grad_mean = np.mean(gradient_magnitude)
                
                # 综合评分：灰度值 + 梯度权重
                score = window_mean + (grad_mean * 0.3)
                
                if score > best_score:
                    best_score = score
                    best_local_center = (x, y)  # 在40x40区域内的坐标
        
        # 步骤3: 将局部坐标转换为全局坐标
        # best_local_center是在40x40 ROI内的坐标，需要转换为全图坐标
        global_x = roi_x1 + best_local_center[0]
        global_y = roi_y1 + best_local_center[1]
        
        predicted_center = (global_x, global_y)
        
        # 归一化评分到0-1范围
        normalized_score = min(1.0, best_score / 255.0)
        
        print(f"📍 灰度预测: 原中心({last_x},{last_y}) -> 40x40 ROI({roi_x1},{roi_y1},{roi_x2},{roi_y2}) -> "
              f"5x5最佳位置({best_local_center[0]},{best_local_center[1]}) -> 全局位置({global_x},{global_y}), 评分{normalized_score:.3f}")
        
        return predicted_center, normalized_score
    
    def gradient_magnitude_prediction_gpu(self, frame, last_center, search_radius=None):
        """基于局部灰度值的位置预测 - GPU加速版本
        
        使用PyTorch和GPU并行计算滑动窗口，大幅提升处理速度
        
        Args:
            frame: 当前帧
            last_center: 上一帧中心位置 (最后检测到的位置)
            search_radius: 搜索半径 (此方法中不使用，保持接口一致)
            
        Returns:
            best_center: 预测的中心位置
            best_score: 最佳匹配得分
        """
        last_x, last_y = last_center
        h, w = frame.shape[:2]
        
        # 将彩色帧转为灰度
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
        
        # 步骤1: 以最后检测中心为基准，提取40x40的ROI区域
        roi_size = self.roi_size  # 40x40
        half_roi = roi_size // 2  # 20
        
        # 计算ROI边界，确保不越界
        roi_x1 = max(0, last_x - half_roi)
        roi_y1 = max(0, last_y - half_roi)
        roi_x2 = min(w, last_x + half_roi)
        roi_y2 = min(h, last_y + half_roi)
        
        # 提取40x40的ROI区域
        roi_40x40 = gray_frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if roi_40x40.size == 0:
            print(f"⚠️ ROI区域为空，返回原位置 {last_center}")
            return last_center, 0.0
        
        # 步骤2: GPU加速的滑动窗口搜索
        window_size = 5
        half_window = window_size // 2  # 2
        
        # 转换为PyTorch张量并移至GPU
        roi_tensor = torch.from_numpy(roi_40x40.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # 使用unfold操作高效提取所有5x5窗口
        # unfold(dimension, size, step)
        windows = roi_tensor.unfold(2, window_size, 1).unfold(3, window_size, 1)
        # 形状: (1, 1, valid_h, valid_w, 5, 5)
        
        if windows.numel() == 0:
            print(f"⚠️ 滑动窗口为空，返回原位置 {last_center}")
            return last_center, 0.0
        
        # 重塑为 (num_windows, 5, 5)
        num_h, num_w = windows.shape[2], windows.shape[3]
        windows = windows.reshape(num_h * num_w, window_size, window_size)
        
        # 计算每个窗口的平均灰度值
        window_means = windows.mean(dim=(1, 2))
        
        # 计算梯度（使用简化的Sobel算子）
        # Sobel X核
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=self.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=self.device)
        
        # 对每个5x5窗口应用Sobel算子
        grad_x_list = []
        grad_y_list = []
        
        for i in range(windows.shape[0]):
            window = windows[i]
            # 应用3x3 Sobel到5x5窗口的中心3x3区域
            center_3x3 = window[1:4, 1:4]
            
            # 计算梯度
            grad_x = torch.sum(center_3x3 * sobel_x)
            grad_y = torch.sum(center_3x3 * sobel_y)
            
            grad_x_list.append(grad_x)
            grad_y_list.append(grad_y)
        
        grad_x_tensor = torch.stack(grad_x_list)
        grad_y_tensor = torch.stack(grad_y_list)
        gradient_magnitudes = torch.sqrt(grad_x_tensor**2 + grad_y_tensor**2)
        
        # 综合评分：灰度值 + 梯度权重
        scores = window_means + (gradient_magnitudes * 0.3)
        
        # 找到最佳评分位置
        best_idx = torch.argmax(scores).item()
        best_score = scores[best_idx].item()
        
        # 转换索引回二维坐标（在滑动窗口坐标系中）
        best_y = best_idx // num_w
        best_x = best_idx % num_w
        
        # 转换为40x40 ROI内的坐标（加上half_window偏移）
        best_local_center = (best_x + half_window, best_y + half_window)
        
        # 步骤3: 将局部坐标转换为全局坐标
        global_x = roi_x1 + best_local_center[0]
        global_y = roi_y1 + best_local_center[1]
        
        predicted_center = (global_x, global_y)
        
        # 归一化评分到0-1范围
        normalized_score = min(1.0, best_score / 255.0)
        
        print(f"🚀 GPU加速预测: 原中心({last_x},{last_y}) -> 40x40 ROI({roi_x1},{roi_y1},{roi_x2},{roi_y2}) -> "
              f"最佳位置({best_local_center[0]},{best_local_center[1]}) -> 全局位置({global_x},{global_y}), 评分{normalized_score:.3f}")
        
        return predicted_center, normalized_score
    
    def grayscale_similarity_search(self, frame, last_center, search_radius=None):
        """基于灰度相似性的搜索
        
        Args:
            frame: 当前帧
            last_center: 上一帧中心位置
            search_radius: 搜索半径
            
        Returns:
            best_center: 最佳匹配中心位置
            best_score: 最佳匹配得分
        """
        if self.local_grayscale_template is None:
            return last_center, 0.0
        
        if search_radius is None:
            search_radius = self.search_radius
        
        last_x, last_y = last_center
        h, w = frame.shape[:2]
        
        # 将彩色帧转为灰度
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
        
        best_score = -1
        best_center = last_center
        template_size = self.local_grayscale_template.shape[0]
        half_template = template_size // 2
        
        # 在搜索半径内寻找最佳匹配
        for dy in range(-search_radius, search_radius + 1, 2):
            for dx in range(-search_radius, search_radius + 1, 2):
                test_x = last_x + dx
                test_y = last_y + dy
                
                # 检查边界
                if (test_x - half_template < 0 or test_x + half_template >= w or
                    test_y - half_template < 0 or test_y + half_template >= h):
                    continue
                
                # 提取候选ROI
                roi, _ = self.extract_roi(gray_frame, test_x, test_y, template_size)
                
                if roi.shape != self.local_grayscale_template.shape:
                    continue
                
                # 计算相似度（归一化相关系数）
                roi_norm = roi.astype(np.float32)
                template_norm = self.local_grayscale_template.astype(np.float32)
                
                # 归一化
                roi_mean = np.mean(roi_norm)
                template_mean = np.mean(template_norm)
                
                roi_centered = roi_norm - roi_mean
                template_centered = template_norm - template_mean
                
                # 计算相关系数
                numerator = np.sum(roi_centered * template_centered)
                denominator = np.sqrt(np.sum(roi_centered**2) * np.sum(template_centered**2))
                
                if denominator > 0:
                    score = numerator / denominator
                    if score > best_score:
                        best_score = score
                        best_center = (test_x, test_y)
        
        return best_center, best_score
    
    def associate_detections(self, detections, frame, frame_id):
        """关联检测结果到轨迹
        
        Args:
            detections: 检测结果
            frame: 当前帧图像
            frame_id: 帧ID
            
        Returns:
            updated_tracks: 更新后的轨迹信息
        """
        # 为新检测分配轨迹ID或更新现有轨迹
        current_frame_tracks = {}
        
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            center = self.calculate_center(x1, y1, x2, y2)
            
            # 简单的最近距离关联
            best_track_id = None
            min_distance = float('inf')
            
            for track_id, track_info in self.tracks.items():
                if 'last_center' in track_info:
                    last_center = track_info['last_center']
                    distance = np.sqrt((center[0] - last_center[0])**2 + 
                                     (center[1] - last_center[1])**2)
                    if distance < min_distance and distance < 100:  # 距离阈值
                        min_distance = distance
                        best_track_id = track_id
            
            if best_track_id is None:
                # 创建新轨迹
                best_track_id = self.track_id_counter
                self.track_id_counter += 1
            
            # 更新轨迹信息，保存检测时的ROI用于后续对比
            roi_x1 = max(0, center[0] - self.roi_size // 2)
            roi_y1 = max(0, center[1] - self.roi_size // 2)
            roi_x2 = min(frame.shape[1], center[0] + self.roi_size // 2)
            roi_y2 = min(frame.shape[0], center[1] + self.roi_size // 2)
            detection_roi = cv2.cvtColor(frame[roi_y1:roi_y2, roi_x1:roi_x2], cv2.COLOR_BGR2GRAY)
            
            self.tracks[best_track_id] = {
                'last_center': center,
                'last_bbox': (x1, y1, x2, y2),
                'last_detection_frame': frame_id,
                'lost_frames': 0,
                'confidence': conf,
                'class_id': cls,
                'status': 'detected',
                'last_detection_roi': detection_roi.copy(),  # 保存检测时的ROI
                'last_detection_info': {
                    'center': center,
                    'roi': detection_roi.copy(),
                    'frame_id': frame_id,
                    'confidence': conf
                }
            }
            
            # 调试信息：YOLO检测
            if frame_id % 30 == 0:  # 每30帧输出一次
                print(f"🎯 YOLO检测 轨迹{best_track_id}: 中心({center[0]}, {center[1]}), "
                      f"边界框({x1}, {y1}, {x2}, {y2}), 置信度{conf:.3f}")
            
            current_frame_tracks[best_track_id] = self.tracks[best_track_id]
        
        return current_frame_tracks
    
    def predict_lost_targets(self, frame, frame_id):
        """预测丢失的目标
        
        Args:
            frame: 当前帧
            frame_id: 帧ID
            
        Returns:
            predicted_tracks: 预测的轨迹信息
        """
        predicted_tracks = {}
        
        for track_id, track_info in list(self.tracks.items()):
            # 检查是否为丢失的轨迹
            if track_info['last_detection_frame'] < frame_id:
                lost_frames = frame_id - track_info['last_detection_frame']
                
                # 持续预测，不限制最大丢失帧数
                # 使用连续预测方法：
                # - 第一次预测：使用最后检测位置作为基准
                # - 后续预测：使用上一次预测位置作为基准（连续传递）
                current_center = track_info['last_center']
                
                print(f"🔄 轨迹{track_id}持续预测: 丢失{lost_frames}帧, 当前中心({current_center[0]}, {current_center[1]})")
                
                # 方法1: 梯度幅值预测（主要方法） - 优先使用GPU加速版本
                if self.use_gpu:
                    predicted_center, match_score = self.gradient_magnitude_prediction_gpu(
                        frame, current_center
                    )
                    prediction_type = "gradient_gpu"
                else:
                    predicted_center, match_score = self.gradient_magnitude_prediction(
                        frame, current_center
                    )
                    prediction_type = "gradient_cpu"
                
                # 方法2: 如果有灰度模板，使用灰度相似性作为辅助
                if self.local_grayscale_template is not None:
                    template_center, template_score = self.grayscale_similarity_search(
                        frame, current_center
                    )
                    # 如果模板匹配更好，使用模板结果
                    if template_score > match_score:
                        predicted_center = template_center
                        match_score = template_score
                        prediction_type = "template"
                
                # 使用最低置信度阈值判断是否继续预测
                if match_score > self.min_prediction_confidence:
                    # 基于预测中心生成边界框 - 使用上次检测的边界框大小
                    if 'last_bbox' in track_info:
                        last_x1, last_y1, last_x2, last_y2 = track_info['last_bbox']
                        last_w = last_x2 - last_x1
                        last_h = last_y2 - last_y1
                        pred_x1 = predicted_center[0] - last_w // 2
                        pred_y1 = predicted_center[1] - last_h // 2
                        pred_x2 = predicted_center[0] + last_w // 2
                        pred_y2 = predicted_center[1] + last_h // 2
                    else:
                        # 如果没有历史边界框，使用ROI尺寸
                        half_size = self.roi_size // 2
                        pred_x1 = predicted_center[0] - half_size
                        pred_y1 = predicted_center[1] - half_size
                        pred_x2 = predicted_center[0] + half_size
                        pred_y2 = predicted_center[1] + half_size
                    
                    # 提取ROI数据用于保存
                    if self.save_process:
                        # 转为灰度图像
                        if len(frame.shape) == 3:
                            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        else:
                            gray_frame = frame
                        
                        # 提取预测位置的ROI
                        roi_data, _ = self.extract_roi(gray_frame, predicted_center[0], predicted_center[1])
                        
                        # 保存处理过程图像
                        # 保存处理过程，包含最后检测信息用于对比
                        last_detection_info = track_info.get('last_detection_info', None)
                        self.save_process_images(frame, track_id, frame_id, predicted_center, 
                                               roi_data, prediction_type, last_detection_info)
                    
                    # 更新轨迹信息
                    self.tracks[track_id].update({
                        'last_center': predicted_center,
                        'last_bbox': (pred_x1, pred_y1, pred_x2, pred_y2),
                        'lost_frames': lost_frames,
                        'confidence': match_score,
                        'status': 'predicted'
                    })
                    
                    predicted_tracks[track_id] = self.tracks[track_id]
                    
                    pred_w = pred_x2 - pred_x1
                    pred_h = pred_y2 - pred_y1
                    print(f"🔍 轨迹{track_id}预测成功: 中心({predicted_center[0]}, {predicted_center[1]}), "
                          f"边界框({pred_x1}, {pred_y1}, {pred_x2}, {pred_y2}), 尺寸{pred_w}x{pred_h}, "
                          f"得分{match_score:.3f}, 丢失{lost_frames}帧, 方法{prediction_type}")
                else:
                    # 预测置信度太低，但保持轨迹继续尝试下一帧
                    self.tracks[track_id]['lost_frames'] = lost_frames
                    self.tracks[track_id]['status'] = 'lost_low_confidence'
                    print(f"⚠️ 轨迹{track_id}预测置信度低: 得分{match_score:.3f} < {self.min_prediction_confidence}，保持轨迹继续尝试")
        
        return predicted_tracks
    
    def draw_tracks(self, frame, detected_tracks, predicted_tracks):
        """绘制轨迹
        
        Args:
            frame: 输入帧
            detected_tracks: 检测到的轨迹
            predicted_tracks: 预测的轨迹
            
        Returns:
            annotated_frame: 标注后的帧
        """
        annotated_frame = frame.copy()
        
        # 绘制检测到的目标 (绿色)
        for track_id, track_info in detected_tracks.items():
            x1, y1, x2, y2 = track_info['last_bbox']
            conf = track_info['confidence']
            center = track_info['last_center']
            
            # 绘制边界框
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 在边界框上方显示ID和置信度
            cv2.putText(annotated_frame, f'ID:{track_id} YOLO:{conf:.2f}', 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 在中心点跟随显示置信度和坐标 (绿色背景)
            conf_coord_text = f'{conf:.3f} ({center[0]},{center[1]})'
            text_size = cv2.getTextSize(conf_coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            # 绘制半透明背景
            cv2.rectangle(annotated_frame, 
                         (center[0] - text_size[0]//2 - 2, center[1] - text_size[1] - 8),
                         (center[0] + text_size[0]//2 + 2, center[1] - 5),
                         (0, 255, 0), -1)
            # 绘制置信度和坐标文本
            cv2.putText(annotated_frame, conf_coord_text,
                       (center[0] - text_size[0]//2, center[1] - 7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # 绘制预测的目标 (红色)
        for track_id, track_info in predicted_tracks.items():
            x1, y1, x2, y2 = track_info['last_bbox']
            score = track_info['confidence']
            lost_frames = track_info['lost_frames']
            center = track_info['last_center']
            
            # 绘制边界框 - 使用更细的线条(1像素)以减少遮挡
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            
            # 在边界框上方显示ID、预测得分和丢失帧数 - 使用更细的文本线条
            cv2.putText(annotated_frame, f'ID:{track_id} Pred:{score:.2f} Lost:{lost_frames}', 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # 绘制预测中心点 - 使用更小的圆点
            cv2.circle(annotated_frame, center, 2, (0, 0, 255), -1)
            
            # 在中心点跟随显示置信度和坐标 (红色背景)
            conf_coord_text = f'{score:.3f} ({center[0]},{center[1]})'
            text_size = cv2.getTextSize(conf_coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            # 绘制半透明背景
            cv2.rectangle(annotated_frame, 
                         (center[0] - text_size[0]//2 - 2, center[1] + 5),
                         (center[0] + text_size[0]//2 + 2, center[1] + text_size[1] + 8),
                         (0, 0, 255), -1)
            # 绘制置信度和坐标文本
            cv2.putText(annotated_frame, conf_coord_text,
                       (center[0] - text_size[0]//2, center[1] + text_size[1] + 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return annotated_frame
    
    def process_video(self, video_path, output_path=None, test_mode=False):
        """处理视频
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径（如果为None，自动创建目录结构）
            test_mode: 是否启用测试模式（强制目标丢失以测试预测功能）
        """
        # 创建结果目录结构
        results_dir, output_video_dir, process_dir = self.create_results_directory(video_path)
        
        # 如果未指定输出路径，自动生成
        if output_path is None:
            video_name = Path(video_path).stem
            suffix = "_test" if test_mode else ""
            output_path = output_video_dir / f"{video_name}_tracked{suffix}.mp4"
        else:
            # 如果指定了输出路径，确保其在正确的目录中
            output_path = output_video_dir / Path(output_path).name
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 设置视频帧率用于文件命名
        self.video_fps = fps if fps > 0 else 30  # 默认30fps
        
        print(f"📹 处理视频: {Path(video_path).name}")
        print(f"   分辨率: {width}x{height}")
        print(f"   帧率: {fps} FPS")
        print(f"   总帧数: {total_frames}")
        print(f"   输出路径: {output_path}")
        # 解析测试模式参数
        force_loss_frames = None
        if test_mode:
            try:
                start_frame, end_frame = map(int, test_mode.split(','))
                force_loss_frames = (start_frame, end_frame)
                print(f"🧪 测试模式: 将在帧{start_frame}-{end_frame}强制目标丢失以测试预测功能")
            except:
                print("⚠️ 测试模式参数格式错误，使用默认帧30-60")
                force_loss_frames = (30, 60)
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_id = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.current_frame_id = frame_id  # 用于测试模式
                
                # YOLO检测
                detections = self.yolo_detect(frame, force_loss_frames)
                
                # 关联检测结果
                detected_tracks = self.associate_detections(detections, frame, frame_id)
                
                # 预测丢失的目标
                predicted_tracks = self.predict_lost_targets(frame, frame_id)
                
                # 绘制结果
                annotated_frame = self.draw_tracks(frame, detected_tracks, predicted_tracks)
                
                # 添加信息文本
                info_text = f"Frame: {frame_id}, Detected: {len(detected_tracks)}, Predicted: {len(predicted_tracks)}"
                cv2.putText(annotated_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 测试模式信息
                if test_mode and force_loss_frames and force_loss_frames[0] <= frame_id <= force_loss_frames[1]:
                    test_text = f"TEST MODE: Forced target loss"
                    cv2.putText(annotated_frame, test_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 写入输出视频
                out.write(annotated_frame)
                
                # 显示进度
                if frame_id % 30 == 0 or frame_id == total_frames - 1:
                    progress = (frame_id / total_frames) * 100
                    elapsed = time.time() - start_time
                    current_fps = frame_id / elapsed if elapsed > 0 else 0
                    
                    # 创建进度条
                    bar_length = 30
                    filled_length = int(bar_length * progress / 100)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    
                    print(f"   进度: [{bar}] {progress:.1f}% ({frame_id}/{total_frames}), "
                          f"用时: {elapsed:.1f}s, 处理FPS: {current_fps:.1f}, "
                          f"检测: {len(detected_tracks)}, 预测: {len(predicted_tracks)}")
                
                frame_id += 1
        
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断处理")
        
        finally:
            cap.release()
            out.release()
        
        processing_time = time.time() - start_time
        print(f"✅ 视频处理完成!")
        print(f"   输出文件: {output_path}")
        print(f"   处理帧数: {frame_id}")
        print(f"   处理时间: {processing_time:.2f}s")
        print(f"   平均FPS: {frame_id / processing_time:.2f}")


def parse_grayscale_template(template_str):
    """解析灰度模板字符串"""
    try:
        # 尝试将字符串转换为numpy数组
        if template_str.startswith('[') and template_str.endswith(']'):
            # 处理列表格式
            import ast
            template_list = ast.literal_eval(template_str)
            template = np.array(template_list, dtype=np.uint8)
        else:
            # 处理其他格式
            template = np.fromstring(template_str, sep=',', dtype=np.uint8)
            # 假设是25x25的模板
            if template.size == 625:
                template = template.reshape(25, 25)
        
        return template
    except Exception as e:
        print(f"⚠️ 解析灰度模板失败: {e}")
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于灰度值的目标追踪系统')
    parser.add_argument('--model', '-m', type=str, 
                       default='small_target_detection/yolov8_small_aircraft/weights/best.pt',
                       help='YOLO模型路径')
    parser.add_argument('--video', '-v', type=str, required=True,
                       help='输入视频路径')
    parser.add_argument('--output', '-o', type=str,
                       help='输出视频路径（默认自动生成）')
    parser.add_argument('--template', '-t', type=str,
                       help='局部灰度值模板')
    parser.add_argument('--test', type=str,
                       help='启用测试模式，指定丢失帧范围 (格式: start,end 例如: 100,150)')
    parser.add_argument('--save-process', action='store_true',
                       help='保存处理过程中的ROI图像和灰度矩阵数据到process目录')
    
    args = parser.parse_args()
    
    # 设置路径
    script_dir = Path(__file__).parent
    model_path = script_dir.parent / args.model
    video_path = Path(args.video)
    
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return 1
    
    if not video_path.exists():
        print(f"❌ 视频文件不存在: {video_path}")
        return 1
    
    # 设置输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = script_dir / "output-vedio"
        output_dir.mkdir(exist_ok=True)
        suffix = "_test" if args.test else ""
        output_path = output_dir / f"tracked{suffix}_{video_path.name}"
    
    try:
        # 创建追踪器
        tracker = GrayscaleTracker(str(model_path), save_process=args.save_process)
        
        # 设置灰度模板
        if args.template:
            template = parse_grayscale_template(args.template)
            if template is not None:
                tracker.set_local_grayscale_template(template)
            else:
                print("⚠️ 使用默认梯度预测策略")
        else:
            print("ℹ️ 未提供灰度模板，使用基于梯度的预测方法")
        
        # 处理视频
        result_info = tracker.process_video(str(video_path), str(output_path) if args.output else None, test_mode=args.test)
        
        print(f"\n🎉 追踪完成！")
        
        if args.test:
            print(f"🧪 测试模式完成，检查视频中的红色预测框")
        
        # 输出信息由process_video方法内部处理，不需要重复输出
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
红外小目标智能追踪系统
结合YOLO11检测与灰度预测的目标追踪算法

====================================
命令行参数 / Parameters
====================================

必需参数:
  --video, -v <PATH>              输入视频路径

可选参数:
  --model, -m <PATH>              YOLO模型路径 (默认: best.pt)
  --output, -o <PATH>             输出视频文件名
  --max-prediction-frames <INT>   最大预测帧数 (默认: 30, 0=纯检测模式)
  --max-detections <INT>          最大追踪目标数 (默认: 5)
  --test <START,END>              测试模式，强制丢失帧范围
  --save-process                  保存处理过程数据
  --template, -t <STR>            灰度模板 (可选)

====================================
使用示例 / Examples
====================================

# 基本使用
python grayscale_tracking_system.py --video test.mp4

# 纯检测模式（不启用预测）
python grayscale_tracking_system.py --video test.mp4 --max-prediction-frames 0

# 自定义参数
python grayscale_tracking_system.py --video test.mp4 --max-detections 3 --model custom.pt

====================================
输出目录 / Output
====================================

结果保存在: Grayscale-Tracking/runs/视频名_时间戳/
  ├── output-video/
  │   ├── 视频名_tracked.mp4        # 追踪视频
  │   └── tracking_statistics.txt   # 统计报告
  └── process/ (可选)
      ├── roi_patches/              # ROI图像
      └── grayscale_data/           # 灰度数据

====================================
核心功能 / Features
====================================

✓ YOLO11小目标检测 + 灰度预测混合追踪
✓ 自适应关联距离 (50-150px动态调整)
✓ 智能ID管理 (ID池复用 + 30帧冷却期)
✓ 速度可视化 (绿色=正常, 黄色=干扰>50px/s)
✓ 噪声过滤 (至少5帧连续检测才触发预测)
✓ GPU加速 (~68 FPS)
✓ 自动目录管理 (时间戳命名)

详细文档: 项目根目录 PROJECT_GUIDE.md
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
    
    def __init__(self, model_path, local_grayscale_template=None, save_process=False, 
                 max_prediction_frames=30, max_detections=5):
        """初始化追踪器
        
        Args:
            model_path: YOLO模型路径
            local_grayscale_template: 局部灰度值模板 (25x25 numpy数组)
            save_process: 是否保存处理过程图像
            max_prediction_frames: 最大预测帧数，目标丢失后进行预测的最大帧数，避免漂移
            max_detections: 最大检测目标数，只保留置信度最高的指定数量目标，避免干扰
        """
        self.model = YOLO(model_path)
        self.local_grayscale_template = local_grayscale_template
        self.save_process = save_process
        
        # 新增控制参数
        self.max_prediction_frames = max_prediction_frames  # 最大预测帧数，避免漂移
        self.max_detections = max_detections  # 最大检测目标数，避免干扰
        
        # 追踪参数
        self.roi_size = 40  # ROI区域大小
        self.search_radius = 50  # 搜索半径
        self.min_prediction_confidence = 0.1  # 最低预测置信度阈值
        # 🎯 自适应关联距离：根据轨迹丢失帧数动态调整
        self.base_association_distance = 50  # 基础关联距离（像素），适用于缓慢移动飞机
        self.max_association_distance = 150  # 最大关联距离（像素），防止误关联
        self.association_distance_per_frame = 5  # 每丢失1帧增加的关联距离（像素）
        self.max_association_frames = 20  # 最大关联帧数限制
        
        # 干扰目标速度阈值（像素/秒）- 超过此速度视为干扰目标，不启动预测直接删除
        self.interference_velocity_threshold = 50  # 干扰目标速度阈值（像素/秒）
        
        # 最小存在帧数要求（目标必须存在这么多帧才能触发预测，避免白噪点触发预测）
        self.min_frames_for_prediction = 5  # 帧数
        
        # 以下参数保留用于其他逻辑
        self.max_aircraft_velocity = 50  # 飞机最大合理速度（像素/秒），超过此速度标记为疑似干扰
        self.velocity_history_window = 10  # 速度历史窗口大小
        self.min_stable_frames = 15  # 最少稳定帧数才进行速度判断
        self.direction_change_threshold = 180  # 方向变化阈值（度）- 允许任意方向变化
        self.velocity_variance_threshold = 100  # 速度方差阈值（像素/秒）
        
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
        self.available_track_ids = []  # ID池：存储已删除可复用的轨迹ID
        self.recently_deleted_ids = {}  # 记录最近删除的ID及其删除帧号 {track_id: frame_id}
        self.id_reuse_cooldown = 30  # ID复用冷却期（帧数），避免同一飞机ID来回切换
        self.max_track_id_ever = 0  # 历史最大ID，用于统计
        self.video_fps = 30  # 视频帧率，处理视频时会更新
        
        # 统计信息
        self.statistics = {
            'total_frames': 0,
            'detection_frames': 0,  # 有YOLO检测到目标的帧数
            'prediction_frames': 0,  # 有预测目标的帧数
            'both_frames': 0,  # 同时有检测和预测的帧数
            'empty_frames': 0,  # 没有任何目标的帧数
            'detection_count': 0,  # 总检测次数
            'prediction_count': 0,  # 总预测次数
            'track_history': defaultdict(list),  # 每个轨迹的历史记录
            'frame_details': [],  # 每帧的详细信息
        }
        
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
        print(f"   自适应关联距离: {self.base_association_distance}-{self.max_association_distance}像素 (基础{self.base_association_distance}px + 每帧{self.association_distance_per_frame}px, 最大{self.max_association_frames}帧)")
        print(f"   最大预测帧数: {self.max_prediction_frames}帧 ({'纯检测模式，宽限期15帧' if self.max_prediction_frames == 0 else '目标丢失后停止预测避免漂移'})")
        print(f"   最大检测目标数: {self.max_detections}个 (只保留置信度最高的目标)")
        print(f"   速度阈值: {self.max_aircraft_velocity}像素/秒 (超过此速度标记为黄色疑似干扰)")
        print(f"   保存过程: {'是' if save_process else '否'}")
        print(f"   追踪策略: 持续预测直到重新检测到目标或超过最大预测帧数")
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
        output_video_dir.mkdir(exist_ok=True)
        
        # 只有在需要保存处理过程时才创建process目录
        process_dir = results_dir / "process"
        if self.save_process:
            process_dir.mkdir(exist_ok=True)
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
            检测结果列表 [(x1, y1, x2, y2, conf, class_id), ...] 按置信度降序排列，数量不超过max_detections
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
        
        # 按置信度降序排列，只保留最高置信度的max_detections个目标
        if detections:
            detections.sort(key=lambda x: x[4], reverse=True)  # 按置信度排序
            detections = detections[:self.max_detections]  # 限制最大检测数量
            
            if len(detections) < len(result.boxes) if result.boxes is not None else 0:
                print(f"🎯 检测数量限制: 从{len(result.boxes) if result.boxes is not None else 0}个检测结果中保留置信度最高的{len(detections)}个")
        
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
    
    def calculate_velocity(self, track_info, frame_id):
        """计算轨迹的瞬时速度和平均速度
        
        Args:
            track_info: 轨迹信息
            frame_id: 当前帧ID
            
        Returns:
            velocity: 当前瞬时速度（像素/秒）
            avg_velocity: 平均速度（像素/秒）
        """
        if 'position_history' not in track_info or len(track_info['position_history']) < 2:
            return 0.0, 0.0
        
        position_history = track_info['position_history']
        
        # 计算瞬时速度（最近两个位置）
        if len(position_history) >= 2:
            last_pos = position_history[-1]
            prev_pos = position_history[-2]
            
            # 兼容两种格式：元组 (x, y) 或字典 {'center': (x, y), 'frame_id': ...}
            if isinstance(last_pos, (tuple, list)):
                dx = last_pos[0] - prev_pos[0]
                dy = last_pos[1] - prev_pos[1]
                # 元组格式假设连续帧，时间差=1帧
                time_diff = 1.0 / self.video_fps
            else:
                dx = last_pos['center'][0] - prev_pos['center'][0]
                dy = last_pos['center'][1] - prev_pos['center'][1]
                time_diff = (last_pos['frame_id'] - prev_pos['frame_id']) / self.video_fps
            
            distance = np.sqrt(dx**2 + dy**2)
            velocity = distance / time_diff if time_diff > 0 else 0.0
        else:
            velocity = 0.0
        
        # 计算平均速度（最近几个位置）
        if len(position_history) >= 3:
            recent_positions = position_history[-min(self.velocity_history_window, len(position_history)):]
            total_distance = 0.0
            total_time = 0.0
            
            for i in range(1, len(recent_positions)):
                # 兼容两种格式
                if isinstance(recent_positions[i], (tuple, list)):
                    dx = recent_positions[i][0] - recent_positions[i-1][0]
                    dy = recent_positions[i][1] - recent_positions[i-1][1]
                    time_diff = 1.0 / self.video_fps
                else:
                    dx = recent_positions[i]['center'][0] - recent_positions[i-1]['center'][0]
                    dy = recent_positions[i]['center'][1] - recent_positions[i-1]['center'][1]
                    time_diff = (recent_positions[i]['frame_id'] - recent_positions[i-1]['frame_id']) / self.video_fps
                
                distance = np.sqrt(dx**2 + dy**2)
                total_distance += distance
                total_time += time_diff
            
            avg_velocity = total_distance / total_time if total_time > 0 else 0.0
        else:
            avg_velocity = velocity
        
        return velocity, avg_velocity
    
    def calculate_motion_stability(self, track_info):
        """计算运动稳定性指标
        
        Args:
            track_info: 轨迹信息
            
        Returns:
            direction_variance: 方向变化方差
            velocity_variance: 速度变化方差
            is_stable: 是否稳定运动
        """
        if 'position_history' not in track_info or len(track_info['position_history']) < 3:
            return 0.0, 0.0, True
        
        position_history = track_info['position_history']
        recent_positions = position_history[-min(self.velocity_history_window, len(position_history)):]
        
        if len(recent_positions) < 3:
            return 0.0, 0.0, True
        
        # 计算方向变化
        directions = []
        velocities = []
        
        for i in range(1, len(recent_positions)):
            dx = recent_positions[i]['center'][0] - recent_positions[i-1]['center'][0]
            dy = recent_positions[i]['center'][1] - recent_positions[i-1]['center'][1]
            
            # 计算方向角度
            if dx != 0 or dy != 0:
                direction = np.arctan2(dy, dx) * 180 / np.pi
                directions.append(direction)
                
                # 计算速度
                distance = np.sqrt(dx**2 + dy**2)
                time_diff = (recent_positions[i]['frame_id'] - recent_positions[i-1]['frame_id']) / self.video_fps
                velocity = distance / time_diff if time_diff > 0 else 0.0
                velocities.append(velocity)
        
        # 计算方向变化方差
        if len(directions) >= 2:
            direction_changes = []
            for i in range(1, len(directions)):
                change = abs(directions[i] - directions[i-1])
                # 处理角度跨越问题（例如从-179度到179度）
                if change > 180:
                    change = 360 - change
                direction_changes.append(change)
            # 使用最大方向变化而不是方差
            direction_variance = max(direction_changes) if direction_changes else 0.0
        else:
            direction_variance = 0.0
        
        # 计算速度变化 - 使用标准差而不是方差
        velocity_variance = np.std(velocities) if len(velocities) > 1 else 0.0
        
        # 判断是否稳定运动
        is_stable = (direction_variance < self.direction_change_threshold and 
                    velocity_variance < self.velocity_variance_threshold)
        
        return direction_variance, velocity_variance, is_stable
    
    def is_interference_target(self, track_info, current_center, frame_id):
        """判断是否为干扰目标
        
        Args:
            track_info: 轨迹信息
            current_center: 当前中心位置
            frame_id: 当前帧ID
            
        Returns:
            is_interference: 是否为干扰目标
            reason: 判断原因
        """
        # 更新位置历史
        if 'position_history' not in track_info:
            track_info['position_history'] = []
        
        track_info['position_history'].append({
            'center': current_center,
            'frame_id': frame_id
        })
        
        # 保持历史记录窗口大小
        if len(track_info['position_history']) > self.velocity_history_window:
            track_info['position_history'].pop(0)
        
        # 需要足够的历史数据才能判断
        if len(track_info['position_history']) < self.min_stable_frames:
            return False, "历史数据不足"
        
        # 计算速度
        velocity, avg_velocity = self.calculate_velocity(track_info, frame_id)
        
        # 速度过快判断
        if avg_velocity > self.max_aircraft_velocity:
            return True, f"平均速度过快: {avg_velocity:.1f} > {self.max_aircraft_velocity} 像素/秒"
        
        # 计算运动稳定性
        direction_var, velocity_var, is_stable = self.calculate_motion_stability(track_info)
        
        # 运动不稳定判断
        if not is_stable:
            return True, f"运动不稳定: 方向变化{direction_var:.1f}°, 速度变化{velocity_var:.1f}"
        
        return False, "正常目标"
    
    def associate_detections(self, detections, frame, frame_id):
        """关联检测结果到轨迹，使用扩大的关联距离和干扰目标过滤
        
        Args:
            detections: 检测结果
            frame: 当前帧图像
            frame_id: 帧ID
            
        Returns:
            updated_tracks: 更新后的轨迹信息
        """
        # 为新检测分配轨迹ID或更新现有轨迹
        current_frame_tracks = {}
        filtered_detections = []
        used_track_ids = set()  # 记录已经使用的轨迹ID，避免一个轨迹被多个检测关联
        
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            center = self.calculate_center(x1, y1, x2, y2)
            
            # 简单的最近距离关联，使用扩大的阈值
            best_track_id = None
            min_distance = float('inf')
            
            for track_id, track_info in self.tracks.items():
                # 🔑 关键修复：跳过本帧已经被关联过的轨迹，确保每个检测都能独立显示
                if track_id in used_track_ids:
                    continue
                    
                if 'last_center' in track_info:
                    last_center = track_info['last_center']
                    lost_frames = frame_id - track_info.get('last_detection_frame', frame_id)
                    
                    # 🎯 自适应关联距离：根据丢失帧数动态调整
                    # 刚检测到：50像素
                    # 丢失1帧：55像素
                    # 丢失5帧：75像素
                    # 丢失10帧：100像素
                    # 丢失20帧：150像素（最大）
                    # 限制最大关联帧数为20帧
                    effective_lost_frames = min(lost_frames, self.max_association_frames)
                    adaptive_threshold = min(
                        self.base_association_distance + effective_lost_frames * self.association_distance_per_frame,
                        self.max_association_distance
                    )
                    
                    # 🎯 纯检测模式优化：使用速度预测辅助关联（不显示预测框）
                    if self.max_prediction_frames == 0 and 'position_history' in track_info and len(track_info['position_history']) >= 2:
                        # 计算简单的线性速度
                        history = track_info['position_history']
                        last_pos = history[-1]  # 最后一个位置 (x, y)
                        prev_pos = history[-2]  # 前一个位置 (x, y)
                        # 确保是元组或列表格式
                        if isinstance(last_pos, (tuple, list)) and isinstance(prev_pos, (tuple, list)):
                            velocity_x = last_pos[0] - prev_pos[0]
                            velocity_y = last_pos[1] - prev_pos[1]
                        else:
                            # 格式错误，跳过速度预测
                            velocity_x = 0
                            velocity_y = 0
                        
                        # 预测当前帧位置
                        predicted_x = last_center[0] + velocity_x * lost_frames
                        predicted_y = last_center[1] + velocity_y * lost_frames
                        predicted_center = (predicted_x, predicted_y)
                        
                        # 计算到预测位置的距离
                        distance = np.sqrt((center[0] - predicted_center[0])**2 + 
                                         (center[1] - predicted_center[1])**2)
                    else:
                        # 计算到最后已知位置的距离
                        distance = np.sqrt((center[0] - last_center[0])**2 + 
                                         (center[1] - last_center[1])**2)
                    
                    # 使用自适应阈值进行关联判断
                    if distance < min_distance and distance < adaptive_threshold:
                        min_distance = distance
                        best_track_id = track_id
            
            # 创建或更新轨迹
            if best_track_id is None:
                # 创建新轨迹：智能ID分配策略
                # 1. 清理冷却期已过的ID，移入可用ID池
                cooled_ids = [tid for tid, del_frame in self.recently_deleted_ids.items() 
                             if frame_id - del_frame >= self.id_reuse_cooldown]
                for tid in cooled_ids:
                    del self.recently_deleted_ids[tid]
                    if tid not in self.available_track_ids:
                        self.available_track_ids.append(tid)
                        self.available_track_ids.sort()
                
                # 2. 从可用ID池中复用（已度过冷却期）
                if self.available_track_ids:
                    best_track_id = self.available_track_ids.pop(0)
                    print(f"🔄 复用轨迹ID{best_track_id}: 距离所有轨迹>{self.base_association_distance}像素 (冷却期{self.id_reuse_cooldown}帧已过)")
                else:
                    # 3. ID池为空，分配新ID
                    best_track_id = self.track_id_counter
                    self.track_id_counter += 1
                    self.max_track_id_ever = max(self.max_track_id_ever, best_track_id)
                    print(f"🆕 创建新轨迹{best_track_id}: 距离所有轨迹>{self.base_association_distance}像素")
            else:
                # 获取该轨迹的自适应阈值用于日志
                lost_frames = frame_id - self.tracks[best_track_id].get('last_detection_frame', frame_id)
                adaptive_threshold = min(
                    self.base_association_distance + lost_frames * self.association_distance_per_frame,
                    self.max_association_distance
                )
                print(f"🔗 关联到轨迹{best_track_id}: 距离{min_distance:.1f}像素 (阈值{adaptive_threshold:.0f}px, 丢失{lost_frames}帧)")
            
            # 🔑 关键修复：标记该轨迹ID已被使用
            used_track_ids.add(best_track_id)
            
            # 更新轨迹信息，保存检测时的ROI用于后续对比
            roi_x1 = max(0, center[0] - self.roi_size // 2)
            roi_y1 = max(0, center[1] - self.roi_size // 2)
            roi_x2 = min(frame.shape[1], center[0] + self.roi_size // 2)
            roi_y2 = min(frame.shape[0], center[1] + self.roi_size // 2)
            detection_roi = cv2.cvtColor(frame[roi_y1:roi_y2, roi_x1:roi_x2], cv2.COLOR_BGR2GRAY)
            
            # 临时轨迹信息用于干扰检测
            temp_track_info = self.tracks.get(best_track_id, {})
            
            # 暂时禁用干扰目标检测，专注解决编号问题
            is_interference = False
            interference_reason = "干扰检测已禁用"
            
            # 原始干扰检测逻辑（暂时注释）
            # is_interference, interference_reason = self.is_interference_target(
            #     temp_track_info, center, frame_id
            # )
            
            if is_interference:
                print(f"🚫 过滤干扰目标 轨迹{best_track_id}: {interference_reason}")
                # 如果是干扰目标，删除该轨迹并跳过，ID进入冷却期
                if best_track_id in self.tracks:
                    del self.tracks[best_track_id]
                    # ID进入冷却期，避免立即复用导致同一飞机ID来回切换
                    self.recently_deleted_ids[best_track_id] = frame_id
                continue
            
            # 正常目标，更新轨迹信息
            self.tracks[best_track_id] = {
                'last_center': center,
                'last_bbox': (x1, y1, x2, y2),
                'last_detection_frame': frame_id,
                'lost_frames': 0,
                'confidence': conf,
                'class_id': cls,
                'status': 'detected',
                'last_detection_roi': detection_roi.copy(),
                'last_detection_info': {
                    'center': center,
                    'roi': detection_roi.copy(),
                    'frame_id': frame_id,
                    'confidence': conf
                },
                'position_history': temp_track_info.get('position_history', []) + [center],
                'detected_frames': temp_track_info.get('detected_frames', 0) + 1  # 累计检测到的帧数
            }
            
            # 限制历史长度，只保留最近10个位置用于速度计算
            if len(self.tracks[best_track_id]['position_history']) > 10:
                self.tracks[best_track_id]['position_history'] = self.tracks[best_track_id]['position_history'][-10:]
            
            # 🔑 计算速度并判断是否为干扰目标
            velocity, avg_velocity = self.calculate_velocity(self.tracks[best_track_id], frame_id)
            self.tracks[best_track_id]['avg_velocity'] = avg_velocity
            self.tracks[best_track_id]['is_interference'] = (avg_velocity >= self.interference_velocity_threshold)
            
            # 调试信息：YOLO检测
            if frame_id % 30 == 0:  # 每30帧输出一次
                velocity, avg_velocity = self.calculate_velocity(self.tracks[best_track_id], frame_id)
                print(f"🎯 YOLO检测 轨迹{best_track_id}: 中心({center[0]}, {center[1]}), "
                      f"边界框({x1}, {y1}, {x2}, {y2}), 置信度{conf:.3f}, 速度{avg_velocity:.1f}像素/秒")
            
            current_frame_tracks[best_track_id] = self.tracks[best_track_id]
            filtered_detections.append(detection)
        
        print(f"🔍 检测关联: 原始{len(detections)}个 → 过滤后{len(filtered_detections)}个, 自适应阈值{self.base_association_distance}-{self.max_association_distance}像素")
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
        tracks_to_remove = []  # 记录需要删除的轨迹
        
        # 1. 收集所有需要预测的丢失轨迹
        lost_tracks_candidates = []
        
        for track_id, track_info in list(self.tracks.items()):
            # 检查是否为丢失的轨迹
            if track_info['last_detection_frame'] < frame_id:
                lost_frames = frame_id - track_info['last_detection_frame']
                
                # 🔑 关键逻辑：干扰目标丢失后不进行预测，直接删除
                is_interference = track_info.get('is_interference', False)
                if is_interference:
                    print(f"⚠️ 干扰目标{track_id}丢失{lost_frames}帧，速度{track_info.get('avg_velocity', 0):.1f}px/s ≥ {self.interference_velocity_threshold}px/s，不启动预测直接删除")
                    tracks_to_remove.append(track_id)
                    continue
                
                # 🔑 最小帧数要求：目标必须存在足够帧数才能触发预测（避免白噪点触发预测浪费算力）
                total_detected_frames = track_info.get('detected_frames', 0)
                if total_detected_frames < self.min_frames_for_prediction:
                    print(f"⚠️ 目标{track_id}仅存在{total_detected_frames}帧 < {self.min_frames_for_prediction}帧（白噪点），不启动预测直接删除")
                    tracks_to_remove.append(track_id)
                    continue
                
                # 🔑 关键修复：当max_prediction_frames=0时，给予宽限期避免频繁删除重建
                # 纯检测模式下，允许轨迹丢失更多帧仍保留，避免短暂遮挡、检测波动导致重新编号
                # 15帧约0.5秒@30fps，足以应对大部分检测间隙
                grace_period = 15 if self.max_prediction_frames == 0 else 0
                max_lost_frames = max(self.max_prediction_frames, grace_period)
                
                # 检查是否超过最大预测帧数限制（含宽限期）
                if lost_frames > max_lost_frames:
                    if self.max_prediction_frames == 0:
                        print(f"❌ 正常目标{track_id}已丢失{lost_frames}帧，超过宽限期{grace_period}帧，删除轨迹")
                    else:
                        print(f"❌ 正常目标{track_id}已丢失{lost_frames}帧，超过最大预测帧数{self.max_prediction_frames}，停止追踪避免漂移")
                    tracks_to_remove.append(track_id)
                    continue
                
                # 添加到候选预测轨迹列表
                # 计算轨迹优先级：最后检测置信度 * 时间权重 (越近的权重越高)
                time_weight = 1.0 / (1.0 + lost_frames * 0.1)  # 时间衰减权重
                priority_score = track_info.get('confidence', 0.0) * time_weight
                
                lost_tracks_candidates.append({
                    'track_id': track_id,
                    'track_info': track_info,
                    'lost_frames': lost_frames,
                    'priority_score': priority_score
                })
        
        # 2. 按优先级排序，只保留最高优先级的轨迹进行预测
        # 排序：优先级得分高的在前
        lost_tracks_candidates.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # 3. 确定可以预测的轨迹数量上限
        # 考虑当前帧已检测到的目标数量
        current_detected_count = len([t for t in self.tracks.values() 
                                    if t.get('last_detection_frame', -1) == frame_id])
        available_prediction_slots = max(0, self.max_detections - current_detected_count)
        
        # 只处理前N个优先级最高的丢失轨迹
        selected_lost_tracks = lost_tracks_candidates[:available_prediction_slots]
        
        if len(lost_tracks_candidates) > available_prediction_slots:
            # 删除优先级较低的轨迹
            for candidate in lost_tracks_candidates[available_prediction_slots:]:
                tracks_to_remove.append(candidate['track_id'])
                print(f"🗑️ 删除低优先级轨迹{candidate['track_id']}: 优先级{candidate['priority_score']:.3f}，"
                      f"为满足最大检测数量{self.max_detections}限制")
        
        print(f"📊 预测控制: 当前检测{current_detected_count}个，可预测{available_prediction_slots}个，"
              f"候选{len(lost_tracks_candidates)}个，选择{len(selected_lost_tracks)}个")
        
        # 🔑 关键修复：当max_prediction_frames=0时，跳过预测逻辑，只保留轨迹等待重新检测
        if self.max_prediction_frames == 0:
            # 纯检测模式：不进行预测，只保留轨迹信息等待下一帧重新检测
            print(f"⏸️ 纯检测模式: 跳过预测，{len(selected_lost_tracks)}个丢失轨迹保留等待重新检测")
            # 5. 删除超过宽限期的轨迹，ID进入冷却期避免立即复用
            for track_id in tracks_to_remove:
                del self.tracks[track_id]
                # ID进入冷却期，避免立即复用导致同一飞机ID来回切换
                self.recently_deleted_ids[track_id] = frame_id
            return predicted_tracks
        
        # 4. 对选中的轨迹进行预测
        for candidate in selected_lost_tracks:
            track_id = candidate['track_id']
            track_info = candidate['track_info']
            lost_frames = candidate['lost_frames']
            
            # 在最大预测帧数内继续预测
            # 使用连续预测方法：
            # - 第一次预测：使用最后检测位置作为基准
            # - 后续预测：使用上一次预测位置作为基准（连续传递）
            current_center = track_info['last_center']
            
            print(f"🔄 轨迹{track_id}持续预测: 丢失{lost_frames}/{self.max_prediction_frames}帧, 优先级{candidate['priority_score']:.3f}, 当前中心({current_center[0]}, {current_center[1]})")
            
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
                      f"得分{match_score:.3f}, 丢失{lost_frames}/{self.max_prediction_frames}帧, 方法{prediction_type}")
            else:
                # 预测置信度太低，但在最大预测帧数内保持轨迹继续尝试下一帧
                self.tracks[track_id]['lost_frames'] = lost_frames
                self.tracks[track_id]['status'] = 'lost_low_confidence'
                print(f"⚠️ 轨迹{track_id}预测置信度低: 得分{match_score:.3f} < {self.min_prediction_confidence}，"
                      f"丢失{lost_frames}/{self.max_prediction_frames}帧，保持轨迹继续尝试")
        
        # 5. 删除超过最大预测帧数或优先级过低的轨迹，避免无限漂移，ID进入冷却期
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            # ID进入冷却期，避免立即复用导致同一飞机ID来回切换
            self.recently_deleted_ids[track_id] = frame_id
        
        return predicted_tracks
    
    def update_statistics(self, frame_id, detected_tracks, predicted_tracks):
        """更新统计信息
        
        Args:
            frame_id: 当前帧ID
            detected_tracks: 检测到的轨迹
            predicted_tracks: 预测的轨迹
        """
        self.statistics['total_frames'] = max(self.statistics['total_frames'], frame_id + 1)
        
        has_detection = len(detected_tracks) > 0
        has_prediction = len(predicted_tracks) > 0
        
        # 更新帧类型统计
        if has_detection:
            self.statistics['detection_frames'] += 1
        if has_prediction:
            self.statistics['prediction_frames'] += 1
        if has_detection and has_prediction:
            self.statistics['both_frames'] += 1
        if not has_detection and not has_prediction:
            self.statistics['empty_frames'] += 1
        
        # 更新目标数量统计
        self.statistics['detection_count'] += len(detected_tracks)
        self.statistics['prediction_count'] += len(predicted_tracks)
        
        # 记录每帧的详细信息
        frame_detail = {
            'frame_id': frame_id,
            'video_time': frame_id / self.video_fps,
            'detection_count': len(detected_tracks),
            'prediction_count': len(predicted_tracks),
            'detected_track_ids': list(detected_tracks.keys()),
            'predicted_track_ids': list(predicted_tracks.keys()),
        }
        self.statistics['frame_details'].append(frame_detail)
        
        # 更新轨迹历史
        for track_id, track_info in detected_tracks.items():
            self.statistics['track_history'][track_id].append({
                'frame_id': frame_id,
                'type': 'detection',
                'confidence': track_info['confidence'],
                'center': track_info['last_center']
            })
        
        for track_id, track_info in predicted_tracks.items():
            self.statistics['track_history'][track_id].append({
                'frame_id': frame_id,
                'type': 'prediction',
                'confidence': track_info['confidence'],
                'center': track_info['last_center'],
                'lost_frames': track_info['lost_frames']
            })
    
    def save_statistics_report(self, output_dir, video_path, processing_time):
        """保存统计报告到输出目录
        
        Args:
            output_dir: 输出目录
            video_path: 视频文件路径
            processing_time: 处理时间
        """
        import json
        from datetime import datetime
        
        stats = self.statistics
        video_name = Path(video_path).stem
        
        # 计算百分比
        total_frames = stats['total_frames']
        detection_percentage = (stats['detection_frames'] / total_frames * 100) if total_frames > 0 else 0
        prediction_percentage = (stats['prediction_frames'] / total_frames * 100) if total_frames > 0 else 0
        both_percentage = (stats['both_frames'] / total_frames * 100) if total_frames > 0 else 0
        empty_percentage = (stats['empty_frames'] / total_frames * 100) if total_frames > 0 else 0
        
        # 创建报告内容
        report_content = f"""# 视频追踪统计报告
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Video: {video_name}
Processing Time: {processing_time:.2f}s
Video Duration: {total_frames / self.video_fps:.1f}s ({total_frames} frames at {self.video_fps} FPS)

## 📊 整体统计

### 帧数统计
- **总帧数**: {total_frames}
- **有检测的帧数**: {stats['detection_frames']} ({detection_percentage:.2f}%)
- **有预测的帧数**: {stats['prediction_frames']} ({prediction_percentage:.2f}%)
- **同时有检测和预测的帧数**: {stats['both_frames']} ({both_percentage:.2f}%)
- **空帧数** (无目标): {stats['empty_frames']} ({empty_percentage:.2f}%)

### 目标统计
- **总检测次数**: {stats['detection_count']}
- **总预测次数**: {stats['prediction_count']}
- **平均每帧检测数**: {stats['detection_count'] / total_frames:.2f}
- **平均每帧预测数**: {stats['prediction_count'] / total_frames:.2f}

### 轨迹统计
- **总轨迹数**: {len(stats['track_history'])}
- **平均轨迹长度**: {sum(len(track) for track in stats['track_history'].values()) / len(stats['track_history']) if stats['track_history'] else 0:.1f} 帧

## 📈 性能指标

### 追踪覆盖率
- **检测覆盖率**: {detection_percentage:.2f}% (有YOLO检测的帧占比)
- **预测覆盖率**: {prediction_percentage:.2f}% (有灰度预测的帧占比)
- **总覆盖率**: {((total_frames - stats['empty_frames']) / total_frames * 100) if total_frames > 0 else 0:.2f}% (有目标的帧占比)

### 处理效率
- **处理FPS**: {total_frames / processing_time:.2f}
- **原视频FPS**: {self.video_fps}
- **实时性**: {'✅ 实时' if (total_frames / processing_time) >= self.video_fps else '❌ 非实时'}

## 🎯 轨迹详情

"""
        
        # 添加每个轨迹的详细信息
        for track_id, track_history in stats['track_history'].items():
            detection_count = sum(1 for record in track_history if record['type'] == 'detection')
            prediction_count = sum(1 for record in track_history if record['type'] == 'prediction')
            total_track_frames = len(track_history)
            
            if total_track_frames > 0:
                start_frame = track_history[0]['frame_id']
                end_frame = track_history[-1]['frame_id']
                duration = (end_frame - start_frame + 1) / self.video_fps
                
                report_content += f"""### 轨迹 {track_id}
- 持续时间: {duration:.1f}s (帧 {start_frame} - {end_frame})
- 总帧数: {total_track_frames}
- 检测帧数: {detection_count} ({detection_count/total_track_frames*100:.1f}%)
- 预测帧数: {prediction_count} ({prediction_count/total_track_frames*100:.1f}%)

"""

        # 保存文本报告
        report_path = output_dir / f"{video_name}_tracking_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📊 统计报告已保存:")
        print(f"   - 详细报告: {report_path}")
        print(f"\n📈 关键指标:")
        print(f"   - 检测覆盖率: {detection_percentage:.2f}% ({stats['detection_frames']}/{total_frames} 帧)")
        print(f"   - 预测覆盖率: {prediction_percentage:.2f}% ({stats['prediction_frames']}/{total_frames} 帧)")
        print(f"   - 总覆盖率: {((total_frames - stats['empty_frames']) / total_frames * 100) if total_frames > 0 else 0:.2f}%")
        print(f"   - 轨迹数量: {len(stats['track_history'])} 条")
        print(f"   - ID管理: 历史最大ID={self.max_track_id_ever}, 当前活跃={len(self.tracks)}, 冷却中={len(self.recently_deleted_ids)}, 可复用={len(self.available_track_ids)}")
        print(f"   - 处理效率: {total_frames / processing_time:.1f} FPS ({'实时' if (total_frames / processing_time) >= self.video_fps else '非实时'})")
    
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
        
        # 绘制检测到的目标 (绿色=正常, 黄色=疑似干扰)
        for track_id, track_info in detected_tracks.items():
            x1, y1, x2, y2 = track_info['last_bbox']
            conf = track_info['confidence']
            center = track_info['last_center']
            
            # 计算速度
            velocity, avg_velocity = self.calculate_velocity(track_info, -1)  # 使用-1表示当前帧
            
            # 🎯 基于速度判断是否为疑似干扰目标
            # 真实飞机速度 < 50 px/s，超过此速度标记为黄色警告
            is_suspicious = avg_velocity > self.max_aircraft_velocity
            
            # 根据速度选择颜色：绿色=正常飞机，黄色=疑似干扰
            color = (0, 255, 255) if is_suspicious else (0, 255, 0)  # BGR: 黄色 or 绿色
            label_prefix = "⚠️" if is_suspicious else ""
            
            # 绘制边界框
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # 在边界框上方显示ID、置信度和速度（带警告标记）
            label = f'{label_prefix}ID:{track_id} YOLO:{conf:.2f} {avg_velocity:.1f}px/s'
            cv2.putText(annotated_frame, label, 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 在中心点跟随显示置信度、坐标和速度 (颜色根据速度判断)
            conf_coord_text = f'{conf:.3f} ({center[0]},{center[1]}) {avg_velocity:.1f}px/s'
            text_size = cv2.getTextSize(conf_coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            # 绘制背景（绿色=正常，黄色=疑似干扰）
            cv2.rectangle(annotated_frame, 
                         (center[0] - text_size[0]//2 - 2, center[1] - text_size[1] - 8),
                         (center[0] + text_size[0]//2 + 2, center[1] - 5),
                         color, -1)
            # 绘制置信度、坐标和速度文本
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
                
                # 更新统计信息
                self.update_statistics(frame_id, detected_tracks, predicted_tracks)
                
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
        
        # 保存统计报告
        self.save_statistics_report(output_video_dir, video_path, processing_time)
        
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
                       default='v11-new/train/yolo11s_ultra_small_aircraft/weights/best.pt',
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
    parser.add_argument('--max-prediction-frames', type=int, default=30,
                       help='最大预测帧数，目标丢失后进行预测的最大帧数，避免漂移 (默认: 30)')
    parser.add_argument('--max-detections', type=int, default=5,
                       help='最大检测目标数，只保留置信度最高的指定数量目标，避免干扰 (默认: 5)')
    
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
    
    try:
        # 创建追踪器
        tracker = GrayscaleTracker(
            str(model_path), 
            save_process=args.save_process,
            max_prediction_frames=args.max_prediction_frames,
            max_detections=args.max_detections
        )
        
        # 设置灰度模板
        if args.template:
            template = parse_grayscale_template(args.template)
            if template is not None:
                tracker.set_local_grayscale_template(template)
            else:
                print("⚠️ 使用默认梯度预测策略")
        else:
            print("ℹ️ 未提供灰度模板，使用基于梯度的预测方法")
        
        # 处理视频（输出路径由process_video内部的create_results_directory自动管理）
        result_info = tracker.process_video(str(video_path), str(args.output) if args.output else None, test_mode=args.test)
        
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

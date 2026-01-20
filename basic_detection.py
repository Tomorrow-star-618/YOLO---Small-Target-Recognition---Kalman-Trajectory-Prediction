#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基础目标检测程序
使用YOLO检测视频中的目标，保存输出视频、检测图像帧和对应的标签
"""

import os
import cv2
import numpy as np
from pathlib import Path
import time
import json
from datetime import datetime

# 尝试导入YOLO
try:
    from ultralytics import YOLO
    HAVE_ULTRALYTICS = True
except ImportError:
    print("❌ 未找到ultralytics库，请安装: pip install ultralytics")
    HAVE_ULTRALYTICS = False

class BasicDetector:
    """基础检测器"""
    
    def __init__(self, 
                 video_path="vedio/1s_60s_complex-background.mp4",
                 output_video="vedio/output.mp4",
                 image_dir="picture/images",
                 label_dir="picture/labels",
                 json_label_dir="picture/json-labels",
                 model_weights="/home/mingxing/worksapce/ultralytics/small_target_detection/yolov8_small_aircraft/weights/best.pt"):
        """
        初始化基础检测器
        
        Args:
            video_path: 输入视频路径
            output_video: 输出视频路径
            image_dir: 图像保存目录
            label_dir: 标签保存目录
            json_label_dir: JSON标签保存目录
            model_weights: 小目标检测模型权重文件路径
        """
        self.video_path = video_path
        self.output_video = output_video
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.json_label_dir = Path(json_label_dir)
        self.model_weights = model_weights
        
        # 创建输出目录
        self._create_directories()
        
        # 检测参数 - 专门针对小目标优化
        self.conf_threshold = 0.15   # 小目标检测的置信度阈值
        self.iou_threshold = 0.45    # IoU阈值
        self.img_size = 640          # 推理分辨率
        
        # 连续检测控制
        self.last_saved_frame = -2  # 上次保存的帧号，初始化为-2确保第一帧会被保存
        self.save_interval = 2      # 连续检测时每2帧保存一次
        
        # 初始化模型和视频
        self.model = None
        self.cap = None
        self.writer = None
        self.video_info = {}
        
        self._initialize()
    
    def _create_directories(self):
        """创建必要的目录"""
        # 创建输出视频目录
        os.makedirs(os.path.dirname(self.output_video), exist_ok=True)
        
        # 创建图像和标签目录
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.label_dir.mkdir(parents=True, exist_ok=True)
        self.json_label_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 图像保存目录: {self.image_dir}")
        print(f"📁 YOLO标签保存目录: {self.label_dir}")
        print(f"📁 JSON标签保存目录: {self.json_label_dir}")
        print(f"📁 输出视频: {self.output_video}")
    
    def _initialize(self):
        """初始化模型和视频"""
        # 初始化YOLO模型
        if not HAVE_ULTRALYTICS:
            raise RuntimeError("需要安装ultralytics库")
        
        if not os.path.exists(self.model_weights):
            print(f"❌ 小目标检测模型权重文件不存在: {self.model_weights}")
            raise FileNotFoundError(f"模型权重文件不存在: {self.model_weights}")
        
        self.model = YOLO(self.model_weights)
        print(f"✅ 已加载小目标检测模型: {self.model_weights}")
        
        # 初始化视频
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"视频文件不存在: {self.video_path}")
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")
        
        # 获取视频信息
        self.video_info = {
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        
        print(f"📊 视频信息:")
        print(f"   📏 分辨率: {self.video_info['width']}x{self.video_info['height']}")
        print(f"   🎥 帧率: {self.video_info['fps']:.1f} fps")
        print(f"   📈 总帧数: {self.video_info['total_frames']}")
        print(f"   ⏱️ 时长: {self.video_info['total_frames']/self.video_info['fps']:.2f}秒")
        
        print(f"🎯 小目标检测参数:")
        print(f"   置信度阈值: {self.conf_threshold}")
        print(f"   推理分辨率: {self.img_size}")
        print(f"   保存间隔: 每{self.save_interval}帧")
        
        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.output_video, 
            fourcc, 
            self.video_info['fps'], 
            (self.video_info['width'], self.video_info['height'])
        )
    
    def detect_frame(self, frame, frame_idx):
        """
        检测单帧图像
        
        Args:
            frame: 输入帧
            frame_idx: 帧索引
            
        Returns:
            vis_frame: 可视化帧
            detections: 检测结果列表
        """
        # 使用小目标检测模型进行检测
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, 
                           imgsz=self.img_size, verbose=False)
        
        # 解析检测结果
        detections = []
        vis_frame = frame.copy()
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i, box in enumerate(boxes):
                # 获取坐标和置信度
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                # 转换为整数坐标
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 计算YOLO格式坐标 (归一化的中心点和宽高)
                img_w, img_h = self.video_info['width'], self.video_info['height']
                center_x = (x1 + x2) / 2.0 / img_w
                center_y = (y1 + y2) / 2.0 / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                
                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'conf': conf,
                    'cls': cls,
                    'size': (x2 - x1, y2 - y1),
                    'yolo_format': (0, center_x, center_y, width, height)  # 强制类别为0
                }
                detections.append(detection)
                
                # 绘制检测框 - 小目标用绿色
                color = (0, 255, 0)  # 绿色
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签 - 显示尺寸信息
                actual_width = x2 - x1
                actual_height = y2 - y1
                label = f"Small Target: {conf:.2f} [{actual_width}x{actual_height}]"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 8), 
                             (x1 + label_size[0], y1), color, -1)
                cv2.putText(vis_frame, label, (x1, y1 - 4), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 添加帧信息
        frame_info = f"Frame: {frame_idx} | Small Targets: {len(detections)}"
        cv2.putText(vis_frame, frame_info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_frame, detections
    
    def save_detection_data(self, frame, detections, frame_idx):
        """
        保存检测数据（图像和标签）
        
        Args:
            frame: 原始帧
            detections: 检测结果
            frame_idx: 帧索引
        """
        if not detections:
            return False
        
        # 检查是否需要保存（连续检测时每2帧保存一次）
        if frame_idx - self.last_saved_frame < self.save_interval:
            return False
        
        # 保存图像
        image_filename = f"frame_{frame_idx:06d}.png"
        image_path = self.image_dir / image_filename
        success = cv2.imwrite(str(image_path), frame)
        
        if not success:
            print(f"❌ 保存图像失败: {image_filename}")
            return False
        
        # 保存YOLO格式标签
        label_filename = f"frame_{frame_idx:06d}.txt"
        label_path = self.label_dir / label_filename
        
        try:
            with open(label_path, 'w') as f:
                for detection in detections:
                    cls, center_x, center_y, width, height = detection['yolo_format']
                    f.write(f"{cls} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        except Exception as e:
            print(f"❌ 保存YOLO标签失败: {label_filename}, 错误: {e}")
            return False
        
        # 保存JSON格式标签
        json_filename = f"frame_{frame_idx:06d}.json"
        json_path = self.json_label_dir / json_filename
        
        # 构建JSON格式数据
        json_data = {
            "frame_idx": frame_idx,
            "frame_file": image_filename,
            "imagePath": image_filename,
            "detections": []
        }
        
        try:
            for detection in detections:
                # 获取检测结果信息
                bbox = detection['bbox']  # [x1, y1, x2, y2] 原始像素坐标
                conf = detection['conf']   # 置信度
                cls = detection['cls']     # 原始类别
                yolo_fmt = detection['yolo_format']  # YOLO格式 [cls, center_x, center_y, width, height]
                size = detection['size']   # [width, height] 目标尺寸
                
                # JSON格式检测数据
                detection_data = {
                    "class": 0,  # 强制为0类（小目标）
                    "original_class": int(cls),  # 原始检测类别
                    "confidence": float(conf),
                    "bbox_pixel": {
                        "x1": int(bbox[0]),
                        "y1": int(bbox[1]), 
                        "x2": int(bbox[2]),
                        "y2": int(bbox[3])
                    },
                    "bbox_normalized": {
                        "center_x": float(yolo_fmt[1]),
                        "center_y": float(yolo_fmt[2]),
                        "width": float(yolo_fmt[3]),
                        "height": float(yolo_fmt[4])
                    },
                    "size": {
                        "width": int(size[0]),
                        "height": int(size[1])
                    }
                }
                json_data["detections"].append(detection_data)
            
            # 保存JSON文件
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"❌ 保存JSON标签失败: {json_filename}, 错误: {e}")
            return False
        
        self.last_saved_frame = frame_idx
        
        # 统计目标大小分布
        sizes_info = [f"{det['size'][0]}x{det['size'][1]}" for det in detections]
        size_summary = f"[{', '.join(sizes_info[:3])}{'...' if len(sizes_info) > 3 else ''}]"
        
        print(f"💾 保存小目标数据: {image_filename} (检测数: {len(detections)}, 尺寸: {size_summary})")
        return True
    
    def process_video(self):
        """处理整个视频"""
        if not self.cap or not self.writer:
            raise RuntimeError("视频未初始化")
        
        print(f"\n🚀 开始处理视频...")
        start_time = time.time()
        
        frame_idx = 0
        saved_count = 0
        total_detections = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 检测当前帧
            vis_frame, detections = self.detect_frame(frame, frame_idx)
            
            # 保存检测数据
            if detections:
                total_detections += len(detections)
                if self.save_detection_data(frame, detections, frame_idx):
                    saved_count += 1
            
            # 写入输出视频
            self.writer.write(vis_frame)
            
            # 显示进度
            if (frame_idx + 1) % 100 == 0:
                progress = (frame_idx + 1) / self.video_info['total_frames'] * 100
                elapsed = time.time() - start_time
                fps = (frame_idx + 1) / elapsed
                print(f"📊 进度: {frame_idx + 1}/{self.video_info['total_frames']} "
                      f"({progress:.1f}%) | FPS: {fps:.1f} | 已保存: {saved_count}")
            
            frame_idx += 1
        
        # 统计信息
        total_time = time.time() - start_time
        avg_fps = frame_idx / total_time
        
        print(f"\n📊 小目标检测完成统计:")
        print(f"   ✅ 处理帧数: {frame_idx}")
        print(f"   🎯 总检测数: {total_detections}")
        print(f"   💾 保存帧数: {saved_count}")
        print(f"   ⏱️ 处理时间: {total_time:.2f}秒")
        print(f"   ⚡ 平均FPS: {avg_fps:.1f}")
        print(f"   📂 图像目录: {self.image_dir}")
        print(f"   📂 标签目录: {self.label_dir}")
        print(f"   🎥 输出视频: {self.output_video}")
    
    def close(self):
        """释放资源"""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
        print("✅ 资源已释放")

def main():
    """主函数"""
    try:
        # 创建检测器，使用小目标检测模型
        detector = BasicDetector(
            video_path="vedio/1s_60s_complex-background.mp4",
            output_video="vedio/output.mp4",
            image_dir="picture/images",
            label_dir="picture/labels",
            json_label_dir="picture/json-labels",
            model_weights="/home/mingxing/worksapce/ultralytics/small_target_detection/yolov8_small_aircraft/weights/best.pt"
        )
        
        # 处理视频
        detector.process_video()
        
        # 释放资源
        detector.close()
        
        print(f"\n🎉 小目标检测视频处理完成！")
        
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

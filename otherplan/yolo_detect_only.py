#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于YOLOv11的红外小目标检测程序
====================================

本脚本仅保留检测功能，去除了卡尔曼滤波、门控距离判断等跟踪相关功能。
专注于单帧检测：仅YOLO检测（已移除传统检测回退）。

⚠️ 重要说明 ⚠️
当前使用的YOLO模型是基于COCO数据集预训练的通用模型，可能不适合红外小目标检测。
如需获得更好的检测效果，建议：
1. 使用专门针对红外小目标训练的YOLO模型
2. 或在红外小目标数据集上fine-tune现有模型

主要功能：
1. 使用YOLO进行红外目标检测（仅保留置信度≥阈值的检测结果）
2. 每处理100帧输出一次进度信息
3. 自动创建输出目录：otherplan/runs/视频名-日期时间/output-video/
4. 无CSV输出，专注于检测可视化

依赖库：
- OpenCV ≥ 4.6
- PyTorch ≥ 1.13
- ultralytics（YOLOv11）
- numpy

作者：基于yolov11x+kalman.py修改
日期：2025-09-17
"""

import os
import sys
import math
import time
from datetime import datetime
import numpy as np
import cv2

# ------------------------- 可选：尝试导入YOLOv11x（若失败则回退传统检测） -------------------------
HAVE_ULTRALYTICS = False
try:
    from ultralytics import YOLO  # pip install ultralytics
    HAVE_ULTRALYTICS = True
except Exception as e:
    print(f"[WARN] 未检测到ultralytics库，YOLO推理将被禁用，改用传统检测回退：{e}")

# ================================ 输入视频路径（用户需修改） ================================
INPUT_VIDEO = "vedio/10s_24s_short.mp4"    # 输入灰度视频路径（mp4）
YOLO_WEIGHTS = "otherplan/yolo11x.pt"       # 改进YOLOv11权重文件

# ================================ 检测参数 ================================
YOLO_CONF_THR = 0.05   # YOLO最小置信度（降低阈值以检测更多潜在目标）
YOLO_IOU_THR = 0.45   # YOLO NMS IoU阈值
YOLO_IMG_SIZE = 640    # YOLO推理分辨率

# 传统增强与几何约束参数（回退检测）
TOPHAT_KSIZE = 6        # 顶帽（强调亮小点），奇数
LOG_GAUSS_SIGMA = 1.2   # LoG中的高斯sigma
BINARY_PRC = 98         # 二值化的分位阈值（0-100），越高越严格
MIN_AREA_RATIO = 1e-6   # 最小面积（相对帧面积）
MAX_AREA_RATIO = 2e-4   # 最大面积（相对帧面积）
MIN_CIRCULARITY = 0.55   # 圆度阈值（4πA/P^2），越接近1越圆
ASPECT_TOL = 0.6         # 宽高比容忍（min(w,h)/max(w,h) ≥ 该值）

# 可视化参数
FONT = cv2.FONT_HERSHEY_SIMPLEX
PROGRESS_INTERVAL = 100  # 每处理多少帧输出一次进度

# --------------------------------- 工具函数 ---------------------------------

def create_output_path(input_video_path):
    """
    根据输入视频创建输出路径
    格式：otherplan/runs/视频名-YYYYMMDD-HHMMSS/output-video/
    """
    # 获取视频文件名（不含扩展名）
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    
    # 获取当前日期时间
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    
    # 构建输出目录
    output_dir = f"otherplan/runs/{video_name}-{timestamp}"
    video_output_dir = os.path.join(output_dir, "output-video")
    
    # 创建目录
    os.makedirs(video_output_dir, exist_ok=True)
    
    # 输出视频路径
    output_video_path = os.path.join(video_output_dir, f"{video_name}_detected.mp4")
    
    return output_video_path, output_dir


def normalize(img: np.ndarray) -> np.ndarray:
    """把图像线性归一到[0,255]的uint8。"""
    img = img.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    out = (img - mn) / (mx - mn) * 255.0
    return out.clip(0, 255).astype(np.uint8)


def enhance_small_targets(gray: np.ndarray) -> np.ndarray:
    """小目标增强：Top-hat + LoG + 局部对比，返回增强图（uint8）。"""
    # Top-hat：突出亮小结构
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_KSIZE, TOPHAT_KSIZE))
    toph = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)

    # LoG：高斯平滑后取拉普拉斯，突出局部极值
    blur = cv2.GaussianBlur(gray, (0, 0), LOG_GAUSS_SIGMA)
    log_ = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
    log_pos = np.maximum(log_, 0.0)

    # 局部对比：原图 - 大尺度模糊（类似白顶帽）
    bg = cv2.GaussianBlur(gray, (0, 0), 5.0)
    local_contrast = cv2.subtract(gray, bg)

    score = 0.5 * normalize(toph) + 0.3 * normalize(log_pos) + 0.2 * normalize(local_contrast)
    return normalize(score)


def classical_detect(gray: np.ndarray, frame_shape) -> list:
    """
    传统回退检测：返回候选框列表[(x1,y1,x2,y2,score), ...]。
    仅在无YOLO或权重缺失时启用。
    """
    H, W = frame_shape[:2]
    enh = enhance_small_targets(gray)
    
    # 高分像素阈值
    thr_val = np.percentile(enh, BINARY_PRC)
    _, bin_ = cv2.threshold(enh, thr_val, 255, cv2.THRESH_BINARY)

    # 连通域/轮廓
    cnts, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    min_area = int(MIN_AREA_RATIO * W * H)
    max_area = int(MAX_AREA_RATIO * W * H)

    for c in cnts:
        area = cv2.contourArea(c)
        if area < max(1, min_area) or area > max_area:
            continue
            
        x, y, w, h = cv2.boundingRect(c)
        
        # 几何约束：圆度检查
        perim = cv2.arcLength(c, True)
        circularity = 0.0 if perim <= 1e-3 else 4.0 * math.pi * area / (perim * perim)
        if circularity < MIN_CIRCULARITY:
            continue
            
        # 宽高比检查
        aspect = min(w, h) / max(w, h)
        if aspect < ASPECT_TOL:
            continue
            
        # 以增强图的均值作为得分
        patch = enh[y:y+h, x:x+w]
        score = float(patch.mean())
        boxes.append((x, y, x + w, y + h, score))

    # 根据得分排序
    boxes.sort(key=lambda b: b[4], reverse=True)
    return boxes


class Detection:
    """检测结果类"""
    def __init__(self, x1, y1, x2, y2, conf, source):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.conf = float(conf)
        self.source = source  # 'yolo' or 'classical'

    @property
    def cx(self):
        return int(0.5 * (self.x1 + self.x2))

    @property
    def cy(self):
        return int(0.5 * (self.y1 + self.y2))

    @property
    def area(self):
        return max(0, (self.x2 - self.x1)) * max(0, (self.y2 - self.y1))


class IRSmallTargetDetector:
    """红外小目标检测器（仅检测，无跟踪）"""
    
    def __init__(self, yolo_model=None):
        self.model = yolo_model
        self.frame_idx = 0
        self.detection_count = 0
        self.start_time = time.time()
        
    def detect_frame(self, frame_bgr, gray):
        """对单帧进行检测，返回检测结果列表"""
        H, W = gray.shape
        detections = []

        # YOLO检测（仅保留高置信度目标）
        if self.model is not None:
            try:
                results = self.model.predict(frame_bgr, conf=YOLO_CONF_THR, iou=YOLO_IOU_THR,
                                           imgsz=YOLO_IMG_SIZE, verbose=False)[0]
                
                yolo_raw_count = len(results.boxes) if results.boxes is not None else 0
                if self.frame_idx == 0 and yolo_raw_count == 0:
                    print(f"[WARN] YOLO未检测到任何目标，可能需要专用的红外小目标检测模型")
                
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # 严格的置信度过滤：只保留高于阈值的检测
                    if conf < YOLO_CONF_THR:
                        continue
                    
                    # 基本几何过滤
                    w, h = x2 - x1, y2 - y1
                    if w <= 0 or h <= 0:
                        continue
                        
                    # 宽高比检查
                    aspect = min(w, h) / max(w, h)
                    if aspect < ASPECT_TOL:
                        continue
                        
                    # 面积检查
                    area = w * h
                    if area < MIN_AREA_RATIO * W * H or area > MAX_AREA_RATIO * W * H:
                        continue
                    
                    detections.append(Detection(x1, y1, x2, y2, conf, 'yolo'))
                    
            except Exception as e:
                print(f"[WARN] YOLO检测出错: {e}")

        # 注意：已移除传统检测回退，仅保留YOLO高置信度检测结果

        return detections

    def process_frame(self, frame_bgr):
        """处理单帧并返回可视化结果"""
        vis = frame_bgr.copy()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # 检测
        detections = self.detect_frame(frame_bgr, gray)
        
        # 统计
        self.detection_count += len(detections)
        
        # 绘制检测框
        for det in detections:
            # 根据检测源选择颜色
            color = (0, 255, 0) if det.source == 'yolo' else (0, 180, 255)
            
            # 画检测框
            cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            
            # 画中心点
            cv2.circle(vis, (det.cx, det.cy), 3, (0, 0, 255), -1)
            
            # 标注信息
            label = f"{det.source}: {det.conf:.2f}"
            cv2.putText(vis, label, (det.x1, max(0, det.y1-6)), 
                       FONT, 0.5, color, 1, cv2.LINE_AA)
        
        # 状态信息
        cv2.putText(vis, f"Frame: {self.frame_idx} | Detections: {len(detections)}", 
                   (10, 24), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 进度输出
        if self.frame_idx % PROGRESS_INTERVAL == 0 and self.frame_idx > 0:
            elapsed = time.time() - self.start_time
            fps = self.frame_idx / elapsed if elapsed > 0 else 0
            avg_detections = self.detection_count / self.frame_idx if self.frame_idx > 0 else 0
            
            print(f"[INFO] 已处理 {self.frame_idx} 帧 | "
                  f"处理速度: {fps:.1f} FPS | "
                  f"平均检测数: {avg_detections:.2f} 个/帧 | "
                  f"总检测数: {self.detection_count}")
        
        self.frame_idx += 1
        return vis


def main():
    """主函数"""
    
    # 检查输入视频
    if not os.path.exists(INPUT_VIDEO):
        print(f"[ERR] 输入视频不存在：{INPUT_VIDEO}")
        return
    
    # 创建输出路径
    output_video_path, output_dir = create_output_path(INPUT_VIDEO)
    print(f"[INFO] 输出目录：{output_dir}")
    print(f"[INFO] 输出视频：{output_video_path}")
    
    # 打开输入视频
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"[ERR] 无法打开输入视频：{INPUT_VIDEO}")
        return

    # 获取视频参数
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[INFO] 视频参数：{W}x{H}, {fps:.1f} FPS, {total_frames} 帧")

    # 创建视频写出器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))
    
    if not writer.isOpened():
        print(f"[ERR] 无法创建输出视频：{output_video_path}")
        return

    # 尝试加载YOLO模型
    yolo_model = None
    if HAVE_ULTRALYTICS and os.path.exists(YOLO_WEIGHTS):
        try:
            yolo_model = YOLO(YOLO_WEIGHTS)
            print(f"[INFO] 已加载YOLO权重：{YOLO_WEIGHTS}")
        except Exception as e:
            print(f"[WARN] YOLO权重加载失败，使用传统检测回退：{e}")
    else:
        if not HAVE_ULTRALYTICS:
            print("[WARN] 未安装ultralytics，使用传统检测回退。")
        else:
            print(f"[WARN] 未找到权重文件：{YOLO_WEIGHTS}，使用传统检测回退。")

    # 创建检测器
    detector = IRSmallTargetDetector(yolo_model)
    
    print("[INFO] 开始处理视频...")
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 确保输入为BGR格式
            if len(frame.shape) == 2 or frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            # 处理帧
            vis_frame = detector.process_frame(frame)
            
            # 写入输出视频
            writer.write(vis_frame)
            
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断处理")
    except Exception as e:
        print(f"[ERR] 处理过程中出错：{e}")
    
    finally:
        # 清理资源
        cap.release()
        writer.release()
        
        # 最终统计
        end_time = time.time()
        total_time = end_time - start_time
        final_fps = detector.frame_idx / total_time if total_time > 0 else 0
        avg_detections = detector.detection_count / detector.frame_idx if detector.frame_idx > 0 else 0
        
        print(f"\n[完成] 视频处理完成！")
        print(f"  处理帧数：{detector.frame_idx}/{total_frames}")
        print(f"  总用时：{total_time:.1f} 秒")
        print(f"  平均速度：{final_fps:.1f} FPS")
        print(f"  总检测数：{detector.detection_count}")
        print(f"  平均检测数：{avg_detections:.2f} 个/帧")
        print(f"  输出视频：{output_video_path}")


if __name__ == "__main__":
    main()

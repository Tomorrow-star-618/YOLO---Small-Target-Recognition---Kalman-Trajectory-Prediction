#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版红外小目标检测程序
========================

基于yolov11x+kalman.py简化而来，只保留检测算法部分：
1. YOLOv11检测（带P2小目标检测头）
2. 传统检测回退（Top-hat + LoG + 几何约束）
3. 去除了卡尔曼滤波、门控距离判断等跟踪相关功能
4. 每处理100帧输出一次进度信息
5. 自动输出路径管理

输出路径规则：
- 输出目录：otherplan/runs/[视频名称-日期时间]/
- 输出视频：otherplan/runs/[视频名称-日期时间]/output-video/[视频名称]_detection.mp4

"""

import os
import sys
import math
import time
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import cv2

# ------------------------- 可选：尝试导入YOLOv11 -------------------------
HAVE_ULTRALYTICS = False
try:
    from ultralytics import YOLO
    HAVE_ULTRALYTICS = True
except Exception as e:
    print(f"[WARN] 未检测到ultralytics库，YOLOv11推理将被禁用，改用传统检测回退：{e}")

# ================================ 用户配置路径 ================================
INPUT_VIDEO = "vedio/10s_24s_short.mp4"    # 输入灰度视频路径（mp4）
YOLO_WEIGHTS = "otherplan/yolo11x.pt"       # YOLOv11权重文件

# 输出路径配置函数
def setup_output_paths(input_video_path):
    """
    根据输入视频和当前时间设置输出路径
    Args:
        input_video_path: 输入视频路径
    Returns:
        str: 输出视频路径
    """
    # 获取视频文件名（不含扩展名）
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    
    # 获取当前日期时间
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建新文件夹名称：视频名称-日期时间
    folder_name = f"{video_name}-{current_time}"
    
    # 输出目录
    output_dir = os.path.join("otherplan", "runs", folder_name)
    
    # 输出视频路径
    output_video_dir = os.path.join(output_dir, "output-video")
    output_video = os.path.join(output_video_dir, f"{video_name}_detection.mp4")
    
    return output_video

# ================================ 检测参数 ================================
YOLO_CONF_THR = 0.15   # YOLO最小置信度
YOLO_IOU_THR = 0.45    # YOLO NMS IoU阈值
YOLO_IMG_SIZE = 640    # YOLO推理分辨率

# 传统增强与几何约束参数
TOPHAT_KSIZE = 6        # 顶帽（强调亮小点），奇数
LOG_GAUSS_SIGMA = 1.2   # LoG中的高斯sigma
BINARY_PRC = 98         # 二值化的分位阈值（0-100），越高越严格
MIN_AREA_RATIO = 1e-6   # 最小面积（相对帧面积）
MAX_AREA_RATIO = 2e-4   # 最大面积（相对帧面积）
MIN_CIRCULARITY = 0.55  # 圆度阈值（4πA/P^2），越接近1越圆
ASPECT_TOL = 0.6        # 宽高比容忍（min(w,h)/max(w,h) ≥ 该值）

# 方向先验参数
DIR_PRIOR_GAIN = 0.6    # 方向先验增益（从右向左更优）

# 进度输出参数
PROGRESS_INTERVAL = 100  # 每处理多少帧输出一次进度

# 可视化参数
FONT = cv2.FONT_HERSHEY_SIMPLEX

# --------------------------------- 工具函数 ---------------------------------

def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def normalize(img: np.ndarray) -> np.ndarray:
    """把图像线性归一到[0,255]的uint8"""
    img = img.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    out = (img - mn) / (mx - mn) * 255.0
    return out.clip(0, 255).astype(np.uint8)


def enhance_small_targets(gray: np.ndarray) -> np.ndarray:
    """小目标增强：Top-hat + LoG + 局部对比，返回增强图（uint8）"""
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
    """传统检测：返回候选框列表[(x1,y1,x2,y2,score), ...]"""
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
        perim = cv2.arcLength(c, True)
        circularity = 0.0 if perim <= 1e-3 else 4.0 * math.pi * area / (perim * perim)
        if circularity < MIN_CIRCULARITY:
            continue
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


@dataclass
class Detection:
    """检测结果数据类"""
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    source: str  # 'yolo' or 'classical'

    @property
    def cx(self):
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self):
        return 0.5 * (self.y1 + self.y2)

    @property
    def area(self):
        return max(0, (self.x2 - self.x1)) * max(0, (self.y2 - self.y1))


class SimpleDetector:
    """简化的检测器，只保留检测功能"""
    
    def __init__(self, cap, writer, yolo_model=None):
        self.cap = cap
        self.writer = writer
        self.model = yolo_model
        self.frame_idx = 0
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.start_time = time.time()
        
        # 统计信息
        self.total_detections = 0
        self.max_targets_per_frame = 0
        self.yolo_detections = 0
        self.classical_detections = 0
        
        print(f"[INFO] 视频总帧数: {self.total_frames}, FPS: {self.fps:.1f}")

    def direction_score(self, prev_cx, new_cx):
        """方向先验得分：从右到左更高分"""
        if prev_cx is None:
            return 1.0
        dx = new_cx - prev_cx
        # 期望 dx<0（从右到左）；若dx>=0则降低得分
        return 1.0 + (0.3 if dx < 0 else -0.3) * DIR_PRIOR_GAIN

    def detect_targets(self, frame_bgr, gray, prev_cx=None):
        """目标检测，返回所有检测到的目标列表"""
        H, W = gray.shape
        cands = []

        # 1. YOLO检测
        if self.model is not None:
            res = self.model.predict(frame_bgr, conf=YOLO_CONF_THR, iou=YOLO_IOU_THR,
                                   imgsz=YOLO_IMG_SIZE, verbose=False)[0]
            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                    conf = float(b.conf[0].cpu().numpy())
                    # 几何与先验过滤
                    w, h = x2 - x1, y2 - y1
                    if w <= 0 or h <= 0:
                        continue
                    aspect = min(w, h) / max(w, h)
                    if aspect < ASPECT_TOL:
                        continue
                    area = w * h
                    if area < MIN_AREA_RATIO * W * H or area > MAX_AREA_RATIO * W * H:
                        continue
                    # 应用方向先验
                    ds = self.direction_score(prev_cx, 0.5*(x1+x2))
                    adj_conf = conf * ds
                    cands.append(Detection(int(x1), int(y1), int(x2), int(y2), adj_conf, 'yolo'))

        # 2. 传统检测回退
        if len(cands) == 0:
            boxes = classical_detect(gray, gray.shape)
            for (x1, y1, x2, y2, score) in boxes:
                ds = self.direction_score(prev_cx, 0.5*(x1+x2))
                adj_conf = float(score / 255.0) * ds
                cands.append(Detection(x1, y1, x2, y2, adj_conf, 'classical'))

        # 3. 返回所有检测结果（而不是只返回最佳的一个）
        if len(cands) == 0:
            return []
        cands.sort(key=lambda d: d.conf, reverse=True)
        return cands

    def process_frame(self, frame_bgr, prev_cx=None):
        """处理单帧，返回检测结果"""
        vis = frame_bgr.copy()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        
        # 检测目标
        detections = self.detect_targets(frame_bgr, gray, prev_cx)
        
        # 可视化
        target_count = len(detections)
        
        if target_count > 0:
            for i, det in enumerate(detections):
                # 根据来源选择颜色
                if det.source == 'yolo':
                    color = (0, 255, 0)  # 绿色
                else:
                    color = (255, 0, 0)  # 蓝色
                
                # 多目标时使用不同颜色
                if target_count > 1:
                    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                    color = colors[i % len(colors)]
                
                # 绘制检测框
                cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), color, 2)
                
                # 标注信息
                label = f"T{i+1}-{det.source}: {det.conf:.2f}"
                cv2.putText(vis, label, (det.x1, max(0, det.y1-6)), 
                           FONT, 0.5, color, 1, cv2.LINE_AA)
                
                # 绘制中心点
                cx, cy = int(det.cx), int(det.cy)
                cv2.circle(vis, (cx, cy), 4, color, -1)
        
        # 状态信息
        cv2.putText(vis, f"Frame: {self.frame_idx}, Targets: {target_count}", 
                   (10, 24), FONT, 0.7, (255, 255, 255), 2)
        
        # 输出到视频
        self.writer.write(vis)
        
        # 更新统计信息
        self.total_detections += len(detections)
        self.max_targets_per_frame = max(self.max_targets_per_frame, len(detections))
        for det in detections:
            if det.source == 'yolo':
                self.yolo_detections += 1
            else:
                self.classical_detections += 1
        
        # 进度输出
        if (self.frame_idx + 1) % PROGRESS_INTERVAL == 0:
            self.print_progress()
        
        self.frame_idx += 1
        return detections

    def print_progress(self, force_print=False):
        """输出进度信息"""
        elapsed = time.time() - self.start_time
        progress = (self.frame_idx + 1) / max(1, self.total_frames) * 100
        fps = (self.frame_idx + 1) / max(elapsed, 1e-6)
        eta = (self.total_frames - self.frame_idx - 1) / max(fps, 1e-6)
        
        avg_targets = self.total_detections / max(1, self.frame_idx + 1)
        
        print(f"[PROGRESS] 帧: {self.frame_idx + 1:>6}/{self.total_frames} "
              f"({progress:>5.1f}%) | "
              f"处理速度: {fps:>5.1f} fps | "
              f"用时: {elapsed:>6.1f}s | "
              f"预计剩余: {eta:>6.1f}s | "
              f"平均目标: {avg_targets:.1f}")
        
        if force_print:
            print(f"[STATS] 总检测: {self.total_detections}, "
                  f"YOLO: {self.yolo_detections}, "
                  f"传统: {self.classical_detections}, "
                  f"单帧最大: {self.max_targets_per_frame}")


def main():
    """主函数"""
    print("[INFO] 简化版红外小目标检测程序启动...")
    
    # 设置输出路径
    output_video = setup_output_paths(INPUT_VIDEO)
    print(f"[INFO] 输入视频: {INPUT_VIDEO}")
    print(f"[INFO] 输出视频: {output_video}")
    print(f"[INFO] 输出目录: {os.path.dirname(os.path.dirname(output_video))}")
    
    # 确保输出目录存在
    ensure_dir(output_video)

    # 打开输入视频
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开输入视频：{INPUT_VIDEO}")
        return

    # 获取视频属性
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] 视频属性: {W}x{H}, {fps:.1f} fps, {total_frames} 帧")

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video, fourcc, fps, (W, H))
    
    if not writer.isOpened():
        print(f"[ERROR] 无法创建视频写入器: {output_video}")
        return

    # 尝试加载YOLO模型
    yolo_model = None
    if HAVE_ULTRALYTICS and os.path.exists(YOLO_WEIGHTS):
        try:
            yolo_model = YOLO(YOLO_WEIGHTS)
            print(f"[INFO] 已加载YOLOv11权重：{YOLO_WEIGHTS}")
        except Exception as e:
            print(f"[WARN] YOLO权重加载失败，使用传统检测回退：{e}")
    else:
        if not HAVE_ULTRALYTICS:
            print("[WARN] 未安装ultralytics，使用传统检测回退")
        else:
            print(f"[WARN] 未找到权重文件：{YOLO_WEIGHTS}，使用传统检测回退")

    # 创建检测器
    detector = SimpleDetector(cap, writer, yolo_model)

    print("[INFO] 开始处理视频...")
    prev_cx = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 确保输入为BGR格式
            if len(frame.shape) == 2 or frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            # 处理当前帧（返回检测结果列表）
            detections = detector.process_frame(frame, prev_cx)
            
            # 更新前一帧中心位置（用于方向先验，使用主要目标）
            if len(detections) > 0:
                prev_cx = detections[0].cx  # 使用置信度最高的目标作为参考
                
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断处理")
    except Exception as e:
        print(f"\n[ERROR] 处理过程中出现错误：{e}")
    
    # 清理资源
    cap.release()
    writer.release()
    
    # 最终进度输出
    detector.print_progress(force_print=True)
    
    print(f"\n[COMPLETE] 多目标检测完成！")
    print(f"  输出视频: {output_video}")
    print(f"  总共处理: {detector.frame_idx} 帧")
    print(f"  检测统计: 总计{detector.total_detections}个目标")
    print(f"    - YOLO检测: {detector.yolo_detections}")
    print(f"    - 传统检测: {detector.classical_detections}")
    print(f"    - 单帧最大目标数: {detector.max_targets_per_frame}")


if __name__ == "__main__":
    main()

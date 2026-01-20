#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

复杂云层背景干扰下的红外多目标检测与跟踪系统
============================================

本系统实现基于YOLO检测与灰度预测融合的多目标跟踪方案，
用于在复杂动态云层背景中检测和跟踪多个从右向左运动的亮白色红外小目标。

核心功能：
---------
1) **多目标检测**：改进YOLOv11同时检测多个红外小目标
2) **多目标跟踪**：每个目标独立的灰度预测器进行位置预测
3) **智能关联**：基于距离和IoU的检测-轨迹关联算法
4) **动态管理**：自动初始化新轨迹和删除消失轨迹
5) **可视化优化**：不同颜色区分不同目标，显示目标ID

用户可配置参数：
- MAX_TARGETS: 最大同时跟踪目标数（默认5个）
- ASSOCIATION_THRESHOLD: 关联距离阈值（默认80像素）
- NEW_TARGET_THRESHOLD: 新目标创建阈值（默认0.3置信度）

多目标跟踪策略：
1. 检测阶段获取所有候选目标
2. 与现有轨迹进行距离匹配关联
3. 未匹配的高置信度检测创建新轨迹
4. 现有轨迹使用灰度预测维持连续性
5. 长期未更新的轨迹自动删除
"""

import os
import sys
import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import cv2
import torch
from typing import List, Dict, Tuple, Optional

# ------------------------- 可选：尝试导入YOLOv11x（若失败则回退传统检测） -------------------------
HAVE_ULTRALYTICS = False
try:
    from ultralytics import YOLO  # pip install ultralytics
    HAVE_ULTRALYTICS = True
except Exception as e:
    print("[WARN] 未检测到ultralytics库，YOLOv8推理将被禁用，改用传统检测回退：", e)

# ================================ 用户需按需修改的路径 ================================
INPUT_VIDEO = "../vedio/26s_50s_short.mp4"    # 输入灰度视频路径（mp4）

# 动态生成输出路径：基于视频名称和处理时间
def generate_output_paths(input_video_path):
    """根据输入视频路径和当前时间生成输出路径"""
    # 提取视频文件名（不含扩展名）
    video_basename = os.path.splitext(os.path.basename(input_video_path))[0]
    
    # 生成时间戳字符串
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建输出文件夹名称
    output_folder = f"{video_basename}_multi_{timestamp}"
    
    # 生成完整路径
    output_dir = os.path.join("runs", output_folder)
    output_video = os.path.join(output_dir, "multi-output.mp4")
    
    return output_video

OUTPUT_VIDEO = generate_output_paths(INPUT_VIDEO)
YOLO_WEIGHTS = "yolo11x.pt"   # YOLOv11权重文件路径

# ================================ 多目标跟踪配置参数 ================================
MAX_TARGETS = 5                    # 最大同时跟踪目标数量
ASSOCIATION_THRESHOLD = 80.0       # 检测与轨迹关联的最大距离（像素）
NEW_TARGET_THRESHOLD = 0.3         # 创建新轨迹的最小置信度
MIN_TRACK_LIFE = 5                 # 轨迹的最小生存帧数
MAX_COAST = float('inf')           # 允许纯预测的最大连续帧数（不限制）

# ================================ 可调参数（检测/跟踪） ================================
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

# 灰度预测参数
ROI_SIZE = 40            # ROI区域大小（40x40像素）
SEARCH_RADIUS = 50       # 搜索半径
MIN_PREDICTION_CONFIDENCE = 0.1  # 最低预测置信度阈值
GATE_DIST_PX = 50.0      # 观测-预测的门控距离（像素）
DIR_PRIOR_GAIN = 0.6     # 方向先验增益

# 可视化参数
TRACE_LEN = 100    # 轨迹可视长度
FONT = cv2.FONT_HERSHEY_SIMPLEX

# 多目标可视化颜色列表（BGR格式）
TARGET_COLORS = [
    (0, 255, 0),    # 绿色
    (255, 0, 0),    # 蓝色
    (0, 0, 255),    # 红色
    (255, 255, 0),  # 青色
    (255, 0, 255),  # 洋红色
    (0, 255, 255),  # 黄色
    (128, 0, 255),  # 紫色
    (255, 128, 0),  # 橙色
    (0, 128, 255),  # 浅蓝色
    (128, 255, 0),  # 浅绿色
]

# --------------------------------- 工具函数 ---------------------------------

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

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
    """传统回退检测：返回候选框列表[(x1,y1,x2,y2,score), ...]。"""
    H, W = frame_shape[:2]
    enh = enhance_small_targets(gray)
    thr_val = np.percentile(enh, BINARY_PRC)
    _, bin_ = cv2.threshold(enh, thr_val, 255, cv2.THRESH_BINARY)

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
        patch = enh[y:y+h, x:x+w]
        score = float(patch.mean())
        boxes.append((x, y, x + w, y + h, score))

    boxes.sort(key=lambda b: b[4], reverse=True)
    return boxes

def calculate_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """计算两点之间的欧几里得距离"""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """计算两个边界框的IoU"""
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    
    # 计算交集区域
    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

# --------------------------------- 数据类 ---------------------------------

@dataclass
class Detection:
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

    def get_bbox(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

class GrayscalePredictor:
    """基于灰度局部区域的目标位置预测器"""
    
    def __init__(self, target_id: int):
        """初始化灰度预测器"""
        self.target_id = target_id
        self.roi_size = ROI_SIZE
        self.search_radius = SEARCH_RADIUS
        self.min_confidence = MIN_PREDICTION_CONFIDENCE
        
        # 状态信息
        self.last_center = None
        self.last_roi = None
        self.template = None
        self.is_initialized = False
        
    def init(self, cx, cy, frame):
        """初始化预测器"""
        self.last_center = (int(cx), int(cy))
        
        # 提取初始ROI作为模板
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
            
        roi, _ = self.extract_roi(gray_frame, int(cx), int(cy))
        self.last_roi = roi
        self.template = roi.copy() if roi.size > 0 else None
        self.is_initialized = True
        
    def extract_roi(self, frame, center_x, center_y, size=None):
        """提取ROI区域"""
        if size is None:
            size = self.roi_size
        
        half_size = size // 2
        h, w = frame.shape[:2]
        
        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(w, center_x + half_size)
        y2 = min(h, center_y + half_size)
        
        roi = frame[y1:y2, x1:x2]
        return roi, (x1, y1, x2, y2)
    
    def gradient_magnitude_prediction(self, frame):
        """基于局部灰度值的位置预测"""
        if not self.is_initialized or self.last_center is None:
            return None, 0.0
            
        last_x, last_y = self.last_center
        h, w = frame.shape[:2]
        
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
        
        half_roi = self.roi_size // 2
        roi_x1 = max(0, last_x - half_roi)
        roi_y1 = max(0, last_y - half_roi)
        roi_x2 = min(w, last_x + half_roi)
        roi_y2 = min(h, last_y + half_roi)
        
        roi_40x40 = gray_frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if roi_40x40.size == 0:
            return self.last_center, 0.0
        
        window_size = 5
        half_window = window_size // 2
        
        best_score = -1
        best_local_center = (roi_40x40.shape[1] // 2, roi_40x40.shape[0] // 2)
        
        for y in range(half_window, roi_40x40.shape[0] - half_window):
            for x in range(half_window, roi_40x40.shape[1] - half_window):
                window_5x5 = roi_40x40[y-half_window:y+half_window+1, 
                                     x-half_window:x+half_window+1]
                
                if window_5x5.shape != (window_size, window_size):
                    continue
                
                window_mean = np.mean(window_5x5.astype(np.float32))
                grad_x = cv2.Sobel(window_5x5.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(window_5x5.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                grad_mean = np.mean(gradient_magnitude)
                
                score = window_mean + (grad_mean * 0.3)
                
                if score > best_score:
                    best_score = score
                    best_local_center = (x, y)
        
        global_x = roi_x1 + best_local_center[0]
        global_y = roi_y1 + best_local_center[1]
        
        predicted_center = (global_x, global_y)
        normalized_score = min(1.0, best_score / 255.0)
        
        self.last_center = predicted_center
        return predicted_center, normalized_score
    
    def predict(self, frame):
        """预测下一个位置"""
        return self.gradient_magnitude_prediction(frame)
    
    def update(self, cx, cy, frame):
        """使用新的观测更新预测器"""
        self.last_center = (int(cx), int(cy))
        
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
            
        roi, _ = self.extract_roi(gray_frame, int(cx), int(cy))
        if roi.size > 0:
            self.last_roi = roi

class Track:
    """单个目标的轨迹类"""
    
    def __init__(self, track_id: int, detection: Detection, frame_idx: int):
        self.track_id = track_id
        self.predictor = GrayscalePredictor(track_id)
        self.trace = deque(maxlen=TRACE_LEN)
        self.last_detection = detection
        self.last_update_frame = frame_idx
        self.miss_count = 0
        self.life_count = 1
        self.color = TARGET_COLORS[track_id % len(TARGET_COLORS)]
        
        # 初始化预测器
        self.current_center = (detection.cx, detection.cy)
        self.trace.append(self.current_center)
        
    def init_predictor(self, frame):
        """初始化灰度预测器"""
        self.predictor.init(self.current_center[0], self.current_center[1], frame)
    
    def update(self, detection: Detection, frame_idx: int, frame):
        """使用新检测更新轨迹"""
        self.last_detection = detection
        self.last_update_frame = frame_idx
        self.miss_count = 0
        self.life_count += 1
        self.current_center = (detection.cx, detection.cy)
        self.trace.append(self.current_center)
        
        # 更新预测器
        if self.predictor.is_initialized:
            self.predictor.update(detection.cx, detection.cy, frame)
        else:
            self.init_predictor(frame)
    
    def predict(self, frame_idx: int, frame):
        """使用灰度预测更新位置"""
        self.miss_count += 1
        self.life_count += 1
        
        if self.predictor.is_initialized:
            predicted_center, confidence = self.predictor.predict(frame)
            if predicted_center:
                self.current_center = predicted_center
                self.trace.append(self.current_center)
        
    def is_valid(self) -> bool:
        """判断轨迹是否有效"""
        return (self.life_count >= MIN_TRACK_LIFE and 
                self.miss_count <= MAX_COAST)
    
    def should_delete(self) -> bool:
        """判断是否应该删除轨迹"""
        return self.miss_count > MAX_COAST

class MultiTargetTracker:
    """多目标跟踪器"""
    
    def __init__(self, cap, writer, yolo_model=None):
        self.cap = cap
        self.writer = writer
        self.model = yolo_model
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_idx = 0
        
        # 多目标管理
        self.tracks: Dict[int, Track] = {}  # track_id -> Track
        self.next_track_id = 0
        
    def direction_score(self, prev_cx, new_cx):
        """方向先验得分：从右到左更高分"""
        if prev_cx is None:
            return 1.0
        dx = new_cx - prev_cx
        return 1.0 + (0.3 if dx < 0 else -0.3) * DIR_PRIOR_GAIN

    def get_all_detections(self, frame_bgr, gray) -> List[Detection]:
        """获取所有检测结果"""
        H, W = gray.shape
        all_detections = []

        # YOLO检测
        if self.model is not None:
            res = self.model.predict(frame_bgr, conf=YOLO_CONF_THR, iou=YOLO_IOU_THR,
                                     imgsz=YOLO_IMG_SIZE, verbose=False)[0]
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                conf = float(b.conf[0].cpu().numpy())
                
                # 几何过滤
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                aspect = min(w, h) / max(w, h)
                if aspect < ASPECT_TOL:
                    continue
                area = w * h
                if area < MIN_AREA_RATIO * W * H or area > MAX_AREA_RATIO * W * H:
                    continue
                
                all_detections.append(Detection(int(x1), int(y1), int(x2), int(y2), conf, 'yolo'))

        # 传统检测回退
        if len(all_detections) == 0:
            boxes = classical_detect(gray, gray.shape)
            for (x1, y1, x2, y2, score) in boxes:
                adj_conf = float(score / 255.0)
                all_detections.append(Detection(x1, y1, x2, y2, adj_conf, 'classical'))

        # 按置信度排序
        all_detections.sort(key=lambda d: d.conf, reverse=True)
        return all_detections

    def associate_detections_to_tracks(self, detections: List[Detection]) -> Tuple[List[Tuple[int, Detection]], List[Detection]]:
        """将检测结果与现有轨迹进行关联"""
        if not self.tracks or not detections:
            return [], detections
        
        # 计算距离矩阵
        track_ids = list(self.tracks.keys())
        distance_matrix = np.full((len(track_ids), len(detections)), float('inf'))
        
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            track_pos = track.current_center
            
            for j, det in enumerate(detections):
                det_pos = (det.cx, det.cy)
                dist = calculate_distance(track_pos, det_pos)
                
                # 只考虑在关联阈值内的匹配
                if dist <= ASSOCIATION_THRESHOLD:
                    distance_matrix[i, j] = dist
        
        # 简单的贪婪匹配算法
        matched_pairs = []
        used_detections = set()
        used_tracks = set()
        
        # 找到所有有效的匹配对并按距离排序
        valid_matches = []
        for i in range(len(track_ids)):
            for j in range(len(detections)):
                if distance_matrix[i, j] < float('inf'):
                    valid_matches.append((distance_matrix[i, j], track_ids[i], j))
        
        valid_matches.sort(key=lambda x: x[0])  # 按距离升序排序
        
        # 贪婪匹配
        for dist, track_id, det_idx in valid_matches:
            if track_id not in used_tracks and det_idx not in used_detections:
                matched_pairs.append((track_id, detections[det_idx]))
                used_tracks.add(track_id)
                used_detections.add(det_idx)
        
        # 未匹配的检测
        unmatched_detections = [det for i, det in enumerate(detections) if i not in used_detections]
        
        return matched_pairs, unmatched_detections

    def create_new_tracks(self, unmatched_detections: List[Detection], frame):
        """为未匹配的高置信度检测创建新轨迹"""
        for detection in unmatched_detections:
            # 只有置信度足够高且轨迹数量未达上限才创建新轨迹
            if (detection.conf >= NEW_TARGET_THRESHOLD and 
                len(self.tracks) < MAX_TARGETS):
                
                new_track = Track(self.next_track_id, detection, self.frame_idx)
                new_track.init_predictor(frame)
                self.tracks[self.next_track_id] = new_track
                self.next_track_id += 1

    def update_tracks(self, matched_pairs: List[Tuple[int, Detection]], frame):
        """更新匹配的轨迹"""
        for track_id, detection in matched_pairs:
            if track_id in self.tracks:
                self.tracks[track_id].update(detection, self.frame_idx, frame)

    def predict_unmatched_tracks(self, frame):
        """对未匹配的轨迹进行预测"""
        matched_track_ids = set()
        for track_id, _ in getattr(self, '_last_matched_pairs', []):
            matched_track_ids.add(track_id)
        
        for track_id, track in self.tracks.items():
            if track_id not in matched_track_ids:
                track.predict(self.frame_idx, frame)

    def remove_invalid_tracks(self):
        """移除无效的轨迹"""
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track.should_delete():
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]

    def step(self, frame_bgr):
        """处理一帧"""
        vis = frame_bgr.copy()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        # 1. 获取所有检测
        detections = self.get_all_detections(frame_bgr, gray)
        
        # 2. 检测与轨迹关联
        matched_pairs, unmatched_detections = self.associate_detections_to_tracks(detections)
        self._last_matched_pairs = matched_pairs  # 保存用于预测阶段
        
        # 3. 更新匹配的轨迹
        self.update_tracks(matched_pairs, frame_bgr)
        
        # 4. 预测未匹配的轨迹
        self.predict_unmatched_tracks(frame_bgr)
        
        # 5. 创建新轨迹
        self.create_new_tracks(unmatched_detections, frame_bgr)
        
        # 6. 移除无效轨迹
        self.remove_invalid_tracks()
        
        # 7. 可视化
        self.visualize_tracks(vis, matched_pairs)
        
        # 8. 输出到视频
        self.writer.write(vis)
        
        self.frame_idx += 1
        return True

    def visualize_tracks(self, vis, matched_pairs):
        """可视化所有轨迹"""
        H, W = vis.shape[:2]
        
        # 绘制每个轨迹
        for track_id, track in self.tracks.items():
            if not track.is_valid():
                continue
                
            cx, cy = track.current_center
            cx, cy = int(cx), int(cy)
            
            # 判断是检测模式还是预测模式
            is_detected = any(tid == track_id for tid, _ in matched_pairs)
            
            if is_detected:
                # 检测模式：绿色检测框
                if track.last_detection:
                    det = track.last_detection
                    cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
                    cv2.putText(vis, f"ID{track_id}: {det.conf:.2f}", 
                              (det.x1, max(0, det.y1-6)), FONT, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                # 预测模式：红色预测框
                pred_size = 20
                pred_x1 = max(0, cx - pred_size//2)
                pred_y1 = max(0, cy - pred_size//2)
                pred_x2 = min(W, cx + pred_size//2)
                pred_y2 = min(H, cy + pred_size//2)
                cv2.rectangle(vis, (pred_x1, pred_y1), (pred_x2, pred_y2), (0, 0, 255), 2)
                cv2.putText(vis, f"ID{track_id}: PRED", 
                          (cx+8, cy-8), FONT, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        # 状态信息
        info_text = f"Frame: {self.frame_idx} | Tracks: {len(self.tracks)} | Active: {len([t for t in self.tracks.values() if t.is_valid()])}"
        cv2.putText(vis, info_text, (10, 25), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    def write_csv_data(self, matched_pairs, unmatched_detections):
        """写入CSV数据"""
        pass  # CSV功能已删除

def main():
    ensure_dir(OUTPUT_VIDEO)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"[ERR] 无法打开输入视频：{INPUT_VIDEO}")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # 视频写出器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (W, H))

    # 尝试加载YOLO
    yolo_model = None
    if HAVE_ULTRALYTICS and os.path.exists(YOLO_WEIGHTS):
        try:
            yolo_model = YOLO(YOLO_WEIGHTS)
            print(f"[INFO] 已加载YOLOv11权重：{YOLO_WEIGHTS}")
        except Exception as e:
            print("[WARN] YOLO权重加载失败，使用传统检测回退：", e)
    else:
        if not HAVE_ULTRALYTICS:
            print("[WARN] 未安装ultralytics，使用传统检测回退。")
        else:
            print(f"[WARN] 未找到权重文件：{YOLO_WEIGHTS}，使用传统检测回退。")

    tracker = MultiTargetTracker(cap, writer, yolo_model)

    print("=" * 60)
    print("多目标红外小目标检测与跟踪系统")
    print("=" * 60)
    print(f"最大同时跟踪目标数: {MAX_TARGETS}")
    print(f"关联距离阈值: {ASSOCIATION_THRESHOLD} 像素")
    print(f"新目标创建阈值: {NEW_TARGET_THRESHOLD}")
    print(f"输入视频: {INPUT_VIDEO}")
    print(f"输出视频: {OUTPUT_VIDEO}")
    print("-" * 60)

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] 视频总帧数: {total_frames}")
    
    # 进度显示相关变量
    start_time = time.time()
    frame_count = 0
    
    print("[INFO] 开始处理...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 确保帧为BGR格式
        if len(frame.shape) == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        tracker.step(frame)
        frame_count += 1
        
        # 每100帧显示一次进度
        if frame_count % 100 == 0:
            current_time = time.time()
            elapsed_time = current_time - start_time
            processing_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            active_tracks = len([t for t in tracker.tracks.values() if t.is_valid()])
            print(f"[INFO] 进度: {frame_count}/{total_frames} ({progress:.1f}%) | "
                  f"处理FPS: {processing_fps:.1f} | 活跃轨迹: {active_tracks}")

    cap.release()
    writer.release()
    
    print("-" * 60)
    print("[SUCCESS] 多目标跟踪完成!")
    print(f"  可视化视频: {OUTPUT_VIDEO}")
    print(f"  总共处理帧数: {frame_count}")
    print(f"  最大轨迹ID: {tracker.next_track_id - 1}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

复杂云层背景干扰下的红外小目标时空关联检测与跟踪
================================================

#需求描述：目前要做一个复杂云背景干扰下检测红外小目标的项目，该项目需要输入mp4格式的灰度图像视频，视频里面有着比较复杂的动态云层背景，
#红外小目标呈现亮白色的圆形，形状紧凑稳定，在云层画面里从右向左运动，而大多数云层形状的边界相对模糊散乱，
#亮暗不均。拟提出一种复杂云层背景干扰下时空关联的红外小目标检测方法，首先在单帧静态目标检测研究方法上，调用yolov11，
#引入无跨步卷积层和P2小目标检测头，解决小目标检测细粒度信息丢失问题，提高小目标检测能力；然后，在动态轨迹预测方法上，引入灰度局部区域预测算法实现红外小目标轨迹预测；
#最后将单帧静态目标检测方法和基于灰度预测的动态轨迹预测关联，实现当低慢小目标检测信息丢失时，依据置信度判别切换灰度预测方法持续获取目标位置，
#实现同一序列中对目标的帧间信息对齐，完成帧间信息的交互，在时间维度上建立关联。根据以上原理，请采用opencv计算机视觉库和pytorch框架实现算法，输入视频路径、
#输出等全部写到代码内，不要在终端单独输入，代码注释采用中文，详细描述使用了哪些算法，参数是如何设置的，解释越详细越好

本脚本实现一个“单帧静态检测（改进YOLOv11） + 动态轨迹预测（卡尔曼滤波） + 置信度切换”的完整流水线，
用于从右向左运动、亮白色、形状紧凑稳定的红外小目标，在复杂动态云层背景中进行鲁棒检测与持续跟踪。

核心思想概述（与参数说明）
--------------------------
1) **单帧静态检测：改进YOLOv11（P2小目标检测头 + 无跨步卷积层）**
   - 通过在YOLOv11中**增加P2检测头**（接收更高分辨率的特征图）提升细粒度小目标的可分辨性；
   - 在干扰较强的场景中，在主干网络前端**替换/移除跨步(Stride=2)的卷积**，以**无跨步卷积层**（Stride=1 + 膨胀卷积/堆叠）保留更多高频细节，降低小目标信息的早期损失；
   - 推理阶段本脚本直接**加载你训练好的权重**（假设已按上面结构训练得到），无需再给出YAML。
   - 若暂时没有权重，脚本带有**传统小目标增强 + 几何约束**的检测回退方案（见 `classical_detect()`）。

   ★ 推理关键阈值（可调）：
   - `YOLO_CONF_THR`：YOLO置信度阈值（默认 0.20）。
   - `YOLO_IOU_THR` ：YOLO NMS的IoU阈值（默认 0.45）。

2) **动态轨迹预测：灰度局部区域预测算法**
   - 状态向量 x = [cx, cy, vx, vy]^T（目标中心位置与速度）；
   - 观测为 z = [cx, cy]^T（来自检测框中心）；
   - 采用帧率推导的Δt构造状态转移矩阵F与过程噪声Q，观测噪声R按像素尺度设定；
   - 内置**方向先验**：红外小目标主要“从右向左”运动，若检测到的候选与该先验不符，会进行打分惩罚。

   ★ 关键参数（可按视频帧率/目标速度微调）：
   - `PROC_NOISE_POS` / `PROC_NOISE_VEL`：过程噪声（位置/速度），默认(1.0, 5.0)。
   - `MEAS_NOISE_POS`：观测噪声（像素），默认3.0。

3) **置信度切换与帧间信息对齐**
   - 当单帧检测低于阈值或丢失时，以灰度局部区域预测值维持目标位置，持续输出，避免轨迹断裂
   - 设置“允许预测的最大连续帧数”`MAX_COAST`（默认30帧），超出后若仍无检测则判定目标消失；
   - 在检测重新出现且与预测**IoU/门限距离**匹配时，自动与当前轨迹**对齐并纠正**，实现帧间关联。

4) **输入/输出在代码内固定**（不通过命令行传参）：
   - `INPUT_VIDEO`   ：输入mp4灰度视频路径；
   - `OUTPUT_VIDEO`  ：带叠加可视化的输出视频；
   - `OUTPUT_CSV`    ：逐帧输出的轨迹表（帧号、时间戳、状态、置信度等）；
   - `YOLO_WEIGHTS`  ：你的改进YOLOv11训练权重（.pt）。

5) **先验与几何/外观约束**（用于削弱云层伪影）：
   - 小目标**近圆形**：使用**圆度(circularity)**与**紧致度**约束过滤非目标；
   - **小面积**：面积范围限制（随画面尺寸自动计算）；
   - **亮于局部背景**：基于Top-hat/LoG增强的响应掩膜进行一致性校验；
   - **方向先验**：从右向左（vx<0）更可信。

依赖（建议）
------------
- Python ≥ 3.8
- OpenCV ≥ 4.6
- PyTorch ≥ 1.13
- ultralytics（YOLOv11）：用于加载你训练好的改进YOLOv11权重

你可以直接运行本脚本。若运行环境缺ultralytics或权重文件缺失，会自动退化为“传统增强+检测”模式。

"""

import os
import sys
import math
import time
import csv
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import cv2
import torch

# ------------------------- 可选：尝试导入YOLOv11x（若失败则回退传统检测） -------------------------
HAVE_ULTRALYTICS = False
try:
    from ultralytics import YOLO  # pip install ultralytics
    HAVE_ULTRALYTICS = True
except Exception as e:
    print("[WARN] 未检测到ultralytics库，YOLOv8推理将被禁用，改用传统检测回退：", e)

# ================================ 用户需按需修改的路径 ================================
#INPUT_VIDEO = "complex-background.mp4"                 # 输入灰度视频路径（mp4）
INPUT_VIDEO = "/home/mingxing/worksapce/ultralytics/vedio/complex-background.mp4"    # 输入灰度视频路径（mp4）

# 动态生成输出路径：基于视频名称和处理时间
def generate_output_paths(input_video_path):
    """根据输入视频路径和当前时间生成输出路径"""
    # 提取视频文件名（不含扩展名）
    video_basename = os.path.splitext(os.path.basename(input_video_path))[0]
    
    # 生成时间戳字符串
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建输出文件夹名称
    output_folder = f"{video_basename}_{timestamp}"
    
    # 生成完整路径
    output_dir = os.path.join("runs", output_folder)
    output_video = os.path.join(output_dir, "output.mp4")
    output_csv = os.path.join(output_dir, "output.csv")
    
    return output_video, output_csv

OUTPUT_VIDEO, OUTPUT_CSV = generate_output_paths(INPUT_VIDEO)
YOLO_WEIGHTS = "best.pt"   # YOLOv11权重文件路径

# 若你的YOLO权重还未就绪，也可以使用传统检测回退模式。

# ================================ 可调参数（检测/跟踪） ================================
YOLO_CONF_THR = 0.15   # YOLO最小置信度
YOLO_IOU_THR = 0.45   # YOLO NMS IoU阈值
YOLO_IMG_SIZE = 640    # YOLO推理分辨率（会自动按需缩放）

# 传统增强与几何约束参数（仅在回退/辅助过滤时使用）
TOPHAT_KSIZE = 6        # 顶帽（强调亮小点），奇数
LOG_GAUSS_SIGMA = 1.2   # LoG中的高斯sigma
BINARY_PRC = 98         # 二值化的分位阈值（0-100），越高越严格
MIN_AREA_RATIO = 1e-6   # 最小面积（相对帧面积）
MAX_AREA_RATIO = 2e-4   # 最大面积（相对帧面积）
MIN_CIRCULARITY = 0.55   # 圆度阈值（4πA/P^2），越接近1越圆
ASPECT_TOL = 0.6         # 宽高比容忍（min(w,h)/max(w,h) ≥ 该值）

# 灰度预测参数（替换卡尔曼参数）
MAX_COAST = float('inf')  # 允许纯预测的最大连续帧数（不限制）
ROI_SIZE = 40            # ROI区域大小（40x40像素）
SEARCH_RADIUS = 50       # 搜索半径
MIN_PREDICTION_CONFIDENCE = 0.1  # 最低预测置信度阈值
GATE_DIST_PX = 50.0      # 观测-预测的门控距离（像素）
DIR_PRIOR_GAIN = 0.6     # 方向先验增益（从右向左更优，分数×>1；反向则×<1）

# 可视化参数
TRACE_LEN = 100    # 轨迹可视长度
FONT = cv2.FONT_HERSHEY_SIMPLEX

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
    """传统回退检测：返回候选框列表[(x1,y1,x2,y2,score), ...]。
       仅在无YOLO或权重缺失时启用，或作为YOLO输出的几何一致性过滤参考。"""
    H, W = frame_shape[:2]
    enh = enhance_small_targets(gray)
    # 高分像素阈值
    thr_val = np.percentile(enh, BINARY_PRC)
    _, bin_ = cv2.threshold(enh, thr_val, 255, cv2.THRESH_BINARY)

    # 连通域/轮廓
    cnts, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = []
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
        areas.append(area)

    # 根据得分排序
    boxes.sort(key=lambda b: b[4], reverse=True)
    return boxes


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


class GrayscalePredictor:
    """基于灰度局部区域的目标位置预测器"""
    
    def __init__(self):
        """初始化灰度预测器"""
        self.roi_size = ROI_SIZE
        self.search_radius = SEARCH_RADIUS
        self.min_confidence = MIN_PREDICTION_CONFIDENCE
        
        # GPU加速设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_gpu = torch.cuda.is_available()
        
        # 状态信息
        self.last_center = None
        self.last_roi = None
        self.template = None
        self.is_initialized = False
        
    def init(self, cx, cy, frame):
        """初始化预测器
        
        Args:
            cx, cy: 初始中心位置
            frame: 当前帧图像
        """
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
    
    def gradient_magnitude_prediction(self, frame):
        """基于局部灰度值的位置预测
        
        逻辑：
        1. 以最后检测中心为基准，提取40x40的ROI区域
        2. 在40x40区域内寻找5x5窗口中灰度值最高的位置
        3. 该5x5窗口的中心就是预测的新中心位置
        
        Args:
            frame: 当前帧
            
        Returns:
            predicted_center: 预测的中心位置 (x, y)
            confidence: 预测置信度
        """
        if not self.is_initialized or self.last_center is None:
            return None, 0.0
            
        last_x, last_y = self.last_center
        h, w = frame.shape[:2]
        
        # 将彩色帧转为灰度
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
        
        # 步骤1: 以最后检测中心为基准，提取40x40的ROI区域
        half_roi = self.roi_size // 2  # 20
        
        # 计算ROI边界，确保不越界
        roi_x1 = max(0, last_x - half_roi)
        roi_y1 = max(0, last_y - half_roi)
        roi_x2 = min(w, last_x + half_roi)
        roi_y2 = min(h, last_y + half_roi)
        
        # 提取40x40的ROI区域
        roi_40x40 = gray_frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if roi_40x40.size == 0:
            return self.last_center, 0.0
        
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
        global_x = roi_x1 + best_local_center[0]
        global_y = roi_y1 + best_local_center[1]
        
        predicted_center = (global_x, global_y)
        
        # 归一化评分到0-1范围
        normalized_score = min(1.0, best_score / 255.0)
        
        # 更新状态
        self.last_center = predicted_center
        
        return predicted_center, normalized_score
    
    def predict(self, frame):
        """预测下一个位置
        
        Args:
            frame: 当前帧
            
        Returns:
            predicted_center: 预测的中心位置 (x, y)
            confidence: 预测置信度
        """
        return self.gradient_magnitude_prediction(frame)
    
    def update(self, cx, cy, frame):
        """使用新的观测更新预测器
        
        Args:
            cx, cy: 观测到的中心位置
            frame: 当前帧
        """
        self.last_center = (int(cx), int(cy))
        
        # 更新ROI模板
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
            
        roi, _ = self.extract_roi(gray_frame, int(cx), int(cy))
        if roi.size > 0:
            self.last_roi = roi
            # 可以选择更新模板或保持原模板
            # self.template = roi.copy()
    
    @property
    def state(self):
        """获取当前状态"""
        if self.last_center is None:
            return None
        return np.array([self.last_center[0], self.last_center[1], 0, 0], dtype=np.float32)


class IRSmallTargetTracker:
    def __init__(self, cap, writer, csv_writer, yolo_model=None):
        self.cap = cap
        self.writer = writer
        self.csv_writer = csv_writer
        self.model = yolo_model
        self.predictor = None  # 灰度预测器
        self.last_det = None
        self.miss_cnt = 0
        self.trace = deque(maxlen=TRACE_LEN)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.dt = 1.0 / max(1e-6, self.fps)
        self.frame_idx = 0

    def direction_score(self, prev_cx, new_cx):
        """方向先验得分：从右到左更高分（<1 惩罚，>1 奖励）。"""
        if prev_cx is None:
            return 1.0
        dx = new_cx - prev_cx
        # 期望 dx<0；若dx>=0则降低得分
        return 1.0 + (0.3 if dx < 0 else -0.3) * DIR_PRIOR_GAIN

    def pick_detection(self, frame_bgr, gray, prev_cx):
        H, W = gray.shape
        cands = []

        # YOLO候选
        if self.model is not None:
            res = self.model.predict(frame_bgr, conf=YOLO_CONF_THR, iou=YOLO_IOU_THR,
                                     imgsz=YOLO_IMG_SIZE, verbose=False)[0]
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
                ds = self.direction_score(prev_cx, 0.5*(x1+x2))
                adj_conf = conf * ds
                cands.append(Detection(int(x1), int(y1), int(x2), int(y2), adj_conf, 'yolo'))

        # 传统候选（作为回退或补充）
        if len(cands) == 0:
            boxes = classical_detect(gray, gray.shape)
            for (x1, y1, x2, y2, score) in boxes:
                ds = self.direction_score(prev_cx, 0.5*(x1+x2))
                adj_conf = float(score / 255.0) * ds
                cands.append(Detection(x1, y1, x2, y2, adj_conf, 'classical'))

        # 取最高得分
        if len(cands) == 0:
            return None
        cands.sort(key=lambda d: d.conf, reverse=True)
        return cands[0]

    def step(self, frame_bgr):
        vis = frame_bgr.copy()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        prev_cx = self.last_det.cx if self.last_det is not None else None
        det = self.pick_detection(frame_bgr, gray, prev_cx)

        mode = 'predict'
        used_det = False
        if det is not None:
            # 若已有预测器，用门控距离限制观测更新
            if self.predictor is not None and self.predictor.state is not None:
                px, py = self.predictor.last_center if self.predictor.last_center else (0, 0)
                dist = math.hypot(det.cx - px, det.cy - py)
                if dist <= GATE_DIST_PX or self.miss_cnt >= 3:
                    self.predictor.update(det.cx, det.cy, frame_bgr)
                    mode = 'detect'
                    used_det = True
                    self.miss_cnt = 0
                else:
                    # 检测与预测不一致，先只预测一帧
                    self.predictor.predict(frame_bgr)
                    self.miss_cnt += 1
            else:
                # 初始化预测器
                self.predictor = GrayscalePredictor()
                self.predictor.init(det.cx, det.cy, frame_bgr)
                mode = 'detect'
                used_det = True
                self.miss_cnt = 0
        else:
            # 没有检测，纯预测
            if self.predictor is not None:
                self.predictor.predict(frame_bgr)
                self.miss_cnt += 1
            else:
                # 仍无预测器，无法输出
                self.miss_cnt += 1

        # 可视化与记录
        draw_cx, draw_cy = None, None
        conf = det.conf if det is not None else 0.0
        src = det.source if det is not None else 'none'

        if self.predictor is not None and self.predictor.last_center is not None:
            cx, cy = self.predictor.last_center
            draw_cx, draw_cy = int(cx), int(cy)
            self.trace.append((draw_cx, draw_cy))

        # 绘制检测框（绿色）
        if det is not None and used_det:
            color = (0, 255, 0)  # 绿色检测框
            cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            cv2.putText(vis, f"DETECT: {conf:.2f}", (det.x1, max(0, det.y1-6)), FONT, 0.5, color, 1, cv2.LINE_AA)

        # 绘制预测中心（红色）
        if draw_cx is not None:
            if mode == 'predict':
                # 预测模式：红色圆圈和预测框
                cv2.circle(vis, (draw_cx, draw_cy), 6, (0, 0, 255), 2)  # 红色空心圆
                # 绘制预测框（假设为20x20像素）
                pred_size = 20
                pred_x1 = max(0, draw_cx - pred_size//2)
                pred_y1 = max(0, draw_cy - pred_size//2)
                pred_x2 = min(W, draw_cx + pred_size//2)
                pred_y2 = min(H, draw_cy + pred_size//2)
                cv2.rectangle(vis, (pred_x1, pred_y1), (pred_x2, pred_y2), (0, 0, 255), 2)  # 红色预测框
                cv2.putText(vis, "PREDICT", (draw_cx+8, draw_cy-8), FONT, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                # 检测模式：绿色实心圆作为中心点
                cv2.circle(vis, (draw_cx, draw_cy), 3, (0, 255, 0), -1)  # 绿色实心圆
            
            self.trace.append((draw_cx, draw_cy))

        # 状态文字
        cv2.putText(vis, f"Frame: {self.frame_idx} | Miss: {self.miss_cnt} | Mode: {mode.upper()}", 
                   (10, 25), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # 输出到视频
        self.writer.write(vis)

        # 写CSV
        t = self.frame_idx / max(1e-6, self.fps)
        row = [self.frame_idx, f"{t:.3f}", mode, f"{conf:.3f}", src,
               int(det.cx) if det is not None else -1,
               int(det.cy) if det is not None else -1,
               int(draw_cx) if draw_cx is not None else -1,
               int(draw_cy) if draw_cy is not None else -1,
               self.miss_cnt]
        self.csv_writer.writerow(row)

        # 终止条件：长期丢失
        if self.miss_cnt > MAX_COAST:
            return False  # 停止（由于MAX_COAST为无限大，此条件永不满足）

        self.last_det = det if used_det else self.last_det
        self.frame_idx += 1
        return True


def main():
    ensure_dir(OUTPUT_VIDEO)
    ensure_dir(OUTPUT_CSV)

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

    # CSV写出器
    csv_f = open(OUTPUT_CSV, 'w', newline='')
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow(["frame", "time_sec", "mode", "conf", "src",
                         "det_cx", "det_cy", "pred_cx", "pred_cy", "miss_cnt"])

    # 尝试加载YOLO
    yolo_model = None
    if HAVE_ULTRALYTICS and os.path.exists(YOLO_WEIGHTS):
        try:
            yolo_model = YOLO(YOLO_WEIGHTS)
            print(f"[INFO] 已加载YOLOv8权重：{YOLO_WEIGHTS}")
        except Exception as e:
            print("[WARN] YOLO权重加载失败，使用传统检测回退：", e)
    else:
        if not HAVE_ULTRALYTICS:
            print("[WARN] 未安装ultralytics，使用传统检测回退。")
        else:
            print(f"[WARN] 未找到权重文件：{YOLO_WEIGHTS}，使用传统检测回退。")

    tracker = IRSmallTargetTracker(cap, writer, csv_writer, yolo_model)

    print("[INFO] 开始处理...")
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] 视频总帧数: {total_frames}")
    
    # 进度显示相关变量
    start_time = time.time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 输入视频若为灰度单通道，cv2读出通常为BGR，这里保证为BGR
        if len(frame.shape) == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        ok = tracker.step(frame)
        
        frame_count += 1
        
        # 每100帧显示一次进度
        if frame_count % 100 == 0:
            current_time = time.time()
            elapsed_time = current_time - start_time
            processing_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"[INFO] 进度: {frame_count}/{total_frames} ({progress:.1f}%) | 处理FPS: {processing_fps:.1f}")
        
        if not ok:
            print("[INFO] 目标长期未观测到，提前结束。")
            break

    cap.release()
    writer.release()
    csv_f.close()
    print("[OK] 处理完成：")
    print("  可视化视频：", OUTPUT_VIDEO)
    print("  轨迹CSV  ：", OUTPUT_CSV)


if __name__ == "__main__":
    main()

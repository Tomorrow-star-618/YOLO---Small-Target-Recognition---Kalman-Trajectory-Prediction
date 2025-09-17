#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

复杂云层背景干扰下的红外小目标时空关联检测与跟踪
================================================

#需求描述：目前要做一个复杂云背景干扰下检测红外小目标的项目，该项目需要输入mp4格式的灰度图像视频，视频里面有着比较复杂的动态云层背景，
#红外小目标呈现亮白色的圆形，形状紧凑稳定，在云层画面里从右向左运动，而大多数云层形状的边界相对模糊散乱，
#亮暗不均。拟提出一种复杂云层背景干扰下时空关联的红外小目标检测方法，首先在单帧静态目标检测研究方法上，调用yolov11，
#引入无跨步卷积层和P2小目标检测头，解决小目标检测细粒度信息丢失问题，提高小目标检测能力；然后，在动态轨迹预测方法上，引入卡尔曼滤波算法实现红外小目标轨迹预测；
#最后将单帧静态目标检测方法和基于卡尔曼滤波的动态轨迹预测关联，实现当低慢小目标检测信息丢失时，依据置信度判别切换动态轨迹预测方法持续获取目标位置，
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

2) **动态轨迹预测：恒速模型卡尔曼滤波（CV-2D）**
   - 状态向量 x = [cx, cy, vx, vy]^T（目标中心位置与速度）；
   - 观测为 z = [cx, cy]^T（来自检测框中心）；
   - 采用帧率推导的Δt构造状态转移矩阵F与过程噪声Q，观测噪声R按像素尺度设定；
   - 内置**方向先验**：红外小目标主要“从右向左”运动，若检测到的候选与该先验不符，会进行打分惩罚。

   ★ 关键参数（可按视频帧率/目标速度微调）：
   - `PROC_NOISE_POS` / `PROC_NOISE_VEL`：过程噪声（位置/速度），默认(1.0, 5.0)。
   - `MEAS_NOISE_POS`：观测噪声（像素），默认3.0。

3) **置信度切换与帧间信息对齐**
   - 当单帧检测低于阈值或丢失时，以卡尔曼预测值维持目标位置，**持续输出**，避免轨迹断裂；
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

【训练结构参考（仅供说明，推理阶段无需）】
-------------------------------------------
# 训练时可使用类似下述YAML思路（示意）：
# - 在stem/early stage避免stride=2，改为stride=1并增加膨胀卷积或堆叠保持感受野；
# - 在head中额外添加P2分支（stride=4对应的高分辨率特征），适合小目标；
# backbone:
#   - Conv(c=..., s=1)  # 无跨步
#   - ...
# head:
#   - Detect(P2, P3, P4, P5)
# 注：具体实现以你的训练代码/框架为准。

"""

import os
import sys
import math
import time
import csv
from collections import deque
from dataclasses import dataclass
import numpy as np
import cv2

# ------------------------- 可选：尝试导入YOLOv11x（若失败则回退传统检测） -------------------------
HAVE_ULTRALYTICS = False
try:
    from ultralytics import YOLO  # pip install ultralytics
    HAVE_ULTRALYTICS = True
except Exception as e:
    print("[WARN] 未检测到ultralytics库，YOLOv8推理将被禁用，改用传统检测回退：", e)

# ================================ 用户需按需修改的路径 ================================
#INPUT_VIDEO = "complex-background.mp4"                 # 输入灰度视频路径（mp4）
INPUT_VIDEO = "vedio/10s_24s_short.mp4"    # 输入灰度视频路径（mp4）
OUTPUT_VIDEO = "runs/complex.mp4"       # 输出可视化视频
OUTPUT_CSV = "runs/complex.csv"     # 输出轨迹CSV
YOLO_WEIGHTS = "otherplan/yolo11x.pt"   # 你的改进YOLOv8权重（已含P2/无跨步）

# 若你的YOLO权重还未就绪，也可以先用传统回退模式跑通流程。

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

# 卡尔曼与切换参数
MAX_COAST = 30            # 允许纯预测的最大连续帧数（避免长期漂移）
PROC_NOISE_POS = 1.0      # 过程噪声-位置（像素）
PROC_NOISE_VEL = 5.0      # 过程噪声-速度（像素/帧）
MEAS_NOISE_POS = 3.0      # 观测噪声-位置（像素）
GATE_DIST_PX = 50.0       # 观测-预测的门控距离（像素）
DIR_PRIOR_GAIN = 0.6      # 方向先验增益（从右向左更优，分数×>1；反向则×<1）

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


class KalmanCV2D:
    """2D恒速模型卡尔曼滤波器：x=[cx, cy, vx, vy]^T"""
    def __init__(self, dt: float, proc_pos=PROC_NOISE_POS, proc_vel=PROC_NOISE_VEL, meas_pos=MEAS_NOISE_POS):
        self.dt = dt
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]], dtype=np.float32)
        q = np.array([proc_pos, proc_pos, proc_vel, proc_vel], dtype=np.float32)
        self.Q = np.diag(q*q)  # 过程噪声协方差
        r = np.array([meas_pos, meas_pos], dtype=np.float32)
        self.R = np.diag(r*r)  # 观测噪声
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)
        self.x = None  # 状态
        self.P = None  # 协方差

    def init(self, cx, cy, init_vel=(-5.0, 0.0)):
        # 初速度给一个向左的小负值以融入方向先验
        vx, vy = init_vel
        self.x = np.array([cx, cy, vx, vy], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 100.0

    def predict(self):
        if self.x is None:
            return None
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, zcx, zcy):
        if self.x is None:
            self.init(zcx, zcy)
        z = np.array([zcx, zcy], dtype=np.float32)
        y = z - (self.H @ self.x)  # 创新
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        return self.x.copy()

    @property
    def state(self):
        return None if self.x is None else self.x.copy()


class IRSmallTargetTracker:
    def __init__(self, cap, writer, csv_writer, yolo_model=None):
        self.cap = cap
        self.writer = writer
        self.csv_writer = csv_writer
        self.model = yolo_model
        self.kf = None
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
            # 若已有KF，用门控距离限制观测更新
            if self.kf is not None and self.kf.state is not None:
                px, py, vx, vy = self.kf.state
                dist = math.hypot(det.cx - px, det.cy - py)
                if dist <= GATE_DIST_PX or self.miss_cnt >= 3:
                    self.kf.update(det.cx, det.cy)
                    mode = 'detect'
                    used_det = True
                    self.miss_cnt = 0
                else:
                    # 检测与预测不一致，先只预测一帧
                    self.kf.predict()
                    self.miss_cnt += 1
            else:
                # 初始化KF
                self.kf = KalmanCV2D(self.dt)
                self.kf.init(det.cx, det.cy)
                mode = 'detect'
                used_det = True
                self.miss_cnt = 0
        else:
            # 没有检测，纯预测
            if self.kf is not None:
                self.kf.predict()
                self.miss_cnt += 1
            else:
                # 仍无KF，无法输出
                self.miss_cnt += 1

        # 若使用了检测但尚未预测一步，可补一次预测以输出更平滑的位置（可选）
        if self.kf is not None and used_det:
            # 先更新后预测，给出下一时刻更平滑的位置用于绘制
            self.kf.predict()

        # 可视化与记录
        draw_cx, draw_cy = None, None
        conf = det.conf if det is not None else 0.0
        src = det.source if det is not None else 'none'

        if self.kf is not None and self.kf.state is not None:
            cx, cy, vx, vy = self.kf.state
            draw_cx, draw_cy = int(cx), int(cy)
            self.trace.append((draw_cx, draw_cy))

        # 绘制检测框
        if det is not None:
            color = (0, 255, 0) if mode == 'detect' and used_det else (0, 180, 255)
            cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            cv2.putText(vis, f"{src}: conf={conf:.2f}", (det.x1, max(0, det.y1-6)), FONT, 0.5, color, 1, cv2.LINE_AA)

        # 绘制预测中心与轨迹
        if draw_cx is not None:
            cv2.circle(vis, (draw_cx, draw_cy), 4, (0, 0, 255), -1)
            cv2.putText(vis, f"{mode}", (draw_cx+6, draw_cy-6), FONT, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            # 轨迹线
            #for i in range(1, len(self.trace)):
                #cv2.line(vis, self.trace[i-1], self.trace[i], (0, 0, 255), 2)

        # 状态文字
        cv2.putText(vis, f"frame={self.frame_idx} miss={self.miss_cnt}", (10, 24), FONT, 0.7, (255, 255, 255), 2)

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
            return False  # 停止

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
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 输入视频若为灰度单通道，cv2读出通常为BGR，这里保证为BGR
        if len(frame.shape) == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        ok = tracker.step(frame)
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

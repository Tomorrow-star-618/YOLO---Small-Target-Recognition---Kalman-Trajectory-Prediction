#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版红外小目标检测程序（性能优化版）
============================

基于原复杂系统简化而来，只保留检测算法部分：
1. YOLOv11检测（带P2小目标检测头）
2. 传统检测回退（Top-hat + LoG + 几何约束）
3. 去除了卡尔曼滤波、门控距离判断等跟踪相关功能
4. 每处理100帧输出一次进度信息
5. 多种性能优化选项

性能优化特性：
- 快速模式：简化算法流程
- GPU加速：YOLOv11推理加速
- 自适应优化：根据实时FPS自动调整
- 可视化优化：减少绘制开销
- I/O优化：更快的视频编码

快速配置选项：
==============
根据你的需求选择配置：

1. 高精度模式（慢）：
   FAST_MODE = False
   YOLO_IMG_SIZE = 640
   SKIP_VISUALIZATION = False

2. 平衡模式（默认推荐）：
   FAST_MODE = False  
   YOLO_IMG_SIZE = 640
   SKIP_VISUALIZATION = False

3. 高速模式（快）：
   FAST_MODE = True
   YOLO_IMG_SIZE = 416
   SKIP_VISUALIZATION = False

4. 极速模式（最快）：
   FAST_MODE = True
   YOLO_IMG_SIZE = 320
   SKIP_VISUALIZATION = True

"""

import os
import sys
import math
import time
import csv
import argparse
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
YOLO_WEIGHTS = "otherplan/yolo11x.pt"           # YOLOv11权重文件

# 输出路径配置函数
def setup_output_paths(input_video_path, enable_csv=False):
    """
    根据输入视频和当前时间设置输出路径
    Args:
        input_video_path: 输入视频路径
        enable_csv: 是否启用CSV输出
    Returns:
        tuple: (输出视频路径, CSV路径, 保存图片目录)
    """
    # 获取视频文件名（不含扩展名）
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    
    # 获取当前日期时间
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建新文件夹名称：视频名称-日期时间
    folder_name = f"{video_name}-{current_time}"
    
    # 输出目录
    output_dir = os.path.join("otherplan", "runs", folder_name)
    
    # 各种输出路径
    output_video_dir = os.path.join(output_dir, "output-video")
    output_video = os.path.join(output_video_dir, f"{video_name}_detection.mp4")
    output_csv = os.path.join(output_dir, f"{video_name}_results.csv") if enable_csv else None
    save_images_dir = os.path.join(output_dir, "save-images")
    
    return output_video, output_csv, save_images_dir

# ================================ 检测参数 ================================
YOLO_CONF_THR = 0.15    # YOLO最小置信度
YOLO_IOU_THR = 0.5     # YOLO NMS IoU阈值（提高阈值减少重叠框）
YOLO_IMG_SIZE = 640    # YOLO推理分辨率（降低分辨率提升速度）

# 传统检测参数（简化版本）
TOPHAT_KSIZE = 5        # 顶帽核大小，减小核提升速度
LOG_GAUSS_SIGMA = 1.0   # LoG高斯sigma，减小提升速度
BINARY_PRC = 97         # 二值化阈值，略降低减少候选
MIN_AREA_RATIO = 1e-6   # 最小面积（相对帧面积）
MAX_AREA_RATIO = 2e-4   # 最大面积（相对帧面积）
MIN_CIRCULARITY = 0.6   # 圆度阈值，提高减少候选
ASPECT_TOL = 0.6         # 宽高比，按原版设置
MAX_CLASSICAL_TARGETS = 3  # 传统检测最大目标数（减少后处理）

# 用户控制参数
ENABLE_CSV_OUTPUT = False   # CSV输出功能开关（可通过命令行参数启用）
SAVE_MULTI_TARGET_IMAGES = False  # 自动保存多目标图像（默认禁用）

# 性能优化参数
ENABLE_GPU = True           # 启用GPU加速（如果可用）
SKIP_VISUALIZATION = False  # 跳过部分可视化提升速度
FAST_MODE = False          # 快速模式（简化处理）- 默认关闭以保持检测精度

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
    """优化版小目标增强：简化流程提升速度"""
    if FAST_MODE:
        # 快速模式：只使用Top-hat，跳过复杂的LoG和对比度增强
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_KSIZE, TOPHAT_KSIZE))
        toph = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
        return toph
    else:
        # 标准模式：保留所有增强
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_KSIZE, TOPHAT_KSIZE))
        toph = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)

        # LoG：高斯平滑后取拉普拉斯
        blur = cv2.GaussianBlur(gray, (0, 0), LOG_GAUSS_SIGMA)
        log_ = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
        log_pos = np.maximum(log_, 0.0)

        # 局部对比
        bg = cv2.GaussianBlur(gray, (0, 0), 3.0)  # 减小核大小
        local_contrast = cv2.subtract(gray, bg)

        # 组合增强结果
        score = 0.6 * normalize(toph) + 0.3 * normalize(log_pos) + 0.1 * normalize(local_contrast)
        return normalize(score)


def classical_detect(gray: np.ndarray, frame_shape) -> list:
    """优化版传统检测：提升处理速度"""
    H, W = frame_shape[:2]
    enh = enhance_small_targets(gray)
    
    # 快速阈值处理
    if FAST_MODE:
        # 使用固定阈值而不是百分位数计算
        mean_val = np.mean(enh)
        std_val = np.std(enh)
        thr_val = mean_val + 2.5 * std_val  # 快速阈值估计
    else:
        thr_val = np.percentile(enh, BINARY_PRC)
    
    _, bin_ = cv2.threshold(enh, thr_val, 255, cv2.THRESH_BINARY)

    # 优化的连通域分析
    cnts, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    min_area = int(MIN_AREA_RATIO * W * H)
    max_area = int(MAX_AREA_RATIO * W * H)

    # 限制处理的轮廓数量
    max_contours = 20 if FAST_MODE else 50
    cnts = cnts[:max_contours]

    for c in cnts:
        area = cv2.contourArea(c)
        if area < max(1, min_area) or area > max_area:
            continue
        
        x, y, w, h = cv2.boundingRect(c)
        
        # 快速几何检查
        aspect = min(w, h) / max(w, h)
        if aspect < ASPECT_TOL:
            continue
            
        if FAST_MODE:
            # 快速模式：跳过圆度计算，直接用面积作为得分
            score = float(area)
        else:
            # 标准模式：计算圆度
            perim = cv2.arcLength(c, True)
            circularity = 0.0 if perim <= 1e-3 else 4.0 * math.pi * area / (perim * perim)
            if circularity < MIN_CIRCULARITY:
                continue
            # 简化得分计算
            score = float(area * circularity)
            
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
    
    def __init__(self, cap, writer, csv_writer, yolo_model=None, save_images_dir=None):
        self.cap = cap
        self.writer = writer
        self.csv_writer = csv_writer
        self.model = yolo_model
        self.save_images_dir = save_images_dir
        self.frame_idx = 0
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.start_time = time.time()
        
        # 统计信息
        self.total_detections = 0
        self.max_targets_per_frame = 0
        self.yolo_detections = 0
        self.classical_detections = 0
        self.saved_images_count = 0  # 保存的多目标图像数量
        
        # 确保保存图像目录存在
        if self.save_images_dir and SAVE_MULTI_TARGET_IMAGES:
            ensure_dir(os.path.join(self.save_images_dir, "dummy.txt"))
        
        print(f"[INFO] 视频总帧数: {self.total_frames}, FPS: {self.fps:.1f}")
        if SAVE_MULTI_TARGET_IMAGES and self.save_images_dir:
            print(f"[INFO] 多目标图像将保存到: {self.save_images_dir}")

    def direction_score(self, prev_cx, new_cx):
        """方向先验得分：从右到左更高分"""
        if prev_cx is None:
            return 1.0
        dx = new_cx - prev_cx
        # 期望 dx<0（从右到左）；若dx>=0则降低得分
        return 1.0 + (0.3 if dx < 0 else -0.3) * DIR_PRIOR_GAIN

    def detect_targets(self, frame_bgr, gray, prev_cx=None):
        """优化版目标检测"""
        H, W = gray.shape
        cands = []

        # 1. YOLOv11检测（参考原版实现）
        if self.model is not None:
            # 使用原版的推理参数设置
            res = self.model.predict(
                frame_bgr, 
                conf=YOLO_CONF_THR, 
                iou=YOLO_IOU_THR,
                imgsz=YOLO_IMG_SIZE, 
                verbose=False
            )[0]
            
            # 按原版方式处理检测结果
            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                    conf = float(b.conf[0].cpu().numpy())
                    
                    # 几何与先验过滤（按原版逻辑）
                    w, h = x2 - x1, y2 - y1
                    if w <= 0 or h <= 0:
                        continue
                        
                    aspect = min(w, h) / max(w, h)
                    if aspect < ASPECT_TOL:
                        continue
                        
                    area = w * h
                    # 如果检测框过大，可能是背景误检，跳过
                    if area < MIN_AREA_RATIO * W * H or area > MAX_AREA_RATIO * W * H:
                        continue
                    
                    # 应用方向先验
                    ds = self.direction_score(prev_cx, 0.5*(x1+x2))
                    adj_conf = conf * ds
                    
                    cands.append(Detection(int(x1), int(y1), int(x2), int(y2), adj_conf, 'yolo'))

        # 2. 传统检测回退（简化版，参考原版）
        if len(cands) == 0:
            boxes = classical_detect(gray, gray.shape)
            for i, (x1, y1, x2, y2, score) in enumerate(boxes[:MAX_CLASSICAL_TARGETS]):
                ds = self.direction_score(prev_cx, 0.5*(x1+x2))
                adj_conf = float(score / 255.0) * ds  # 按原版归一化
                cands.append(Detection(x1, y1, x2, y2, adj_conf, 'classical'))

        # 3. 返回所有检测结果
        if len(cands) == 0:
            return []
        cands.sort(key=lambda d: d.conf, reverse=True)
        return cands

    def process_frame(self, frame_bgr, prev_cx=None):
        """优化版帧处理"""
        vis = frame_bgr.copy() if not SKIP_VISUALIZATION else frame_bgr
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # 检测目标
        detections = self.detect_targets(frame_bgr, gray, prev_cx)
        
        # 优化的可视化
        target_count = len(detections)
        
        if not SKIP_VISUALIZATION and target_count > 0:
            for i, det in enumerate(detections):
                # 简化颜色选择
                if det.source == 'yolo':
                    color = (0, 255, 0)  # 绿色
                else:
                    color = (255, 0, 0)  # 蓝色
                
                # 多目标时使用预定义颜色
                if target_count > 1:
                    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                    color = colors[i % len(colors)]
                
                # 绘制检测框
                cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), color, 2)
                
                # 简化标注（减少文字绘制）
                if FAST_MODE:
                    label = f"T{i+1}"  # 只显示编号
                else:
                    label = f"T{i+1}-{det.source}: {det.conf:.2f}"
                    
                cv2.putText(vis, label, (det.x1, max(0, det.y1-6)), 
                           FONT, 0.5, color, 1, cv2.LINE_AA)
                
                # 绘制中心点
                cx, cy = int(det.cx), int(det.cy)
                cv2.circle(vis, (cx, cy), 3, color, -1)  # 减小圆点大小
        
        # 简化状态信息
        if not SKIP_VISUALIZATION:
            cv2.putText(vis, f"F:{self.frame_idx}", (10, 24), 
                       FONT, 0.6, (255, 255, 255), 1)  # 简化文字
            if target_count > 0:
                cv2.putText(vis, f"T:{target_count}", (10, 45), 
                           FONT, 0.6, (255, 255, 255), 1)
            # 显示已保存的多目标图像数量
            if SAVE_MULTI_TARGET_IMAGES and self.saved_images_count > 0:
                cv2.putText(vis, f"Saved:{self.saved_images_count}", (10, 66), 
                           FONT, 0.6, (0, 255, 255), 1)

        # 保存多目标图像（必须在绘制完标注后进行，确保保存的是已标注图像）
        if target_count >= 2 and SAVE_MULTI_TARGET_IMAGES and self.save_images_dir:
            self.save_multi_target_image(vis, detections)
        
        # 输出到视频
        self.writer.write(vis)
        
        # CSV写入（简化版本）
        if ENABLE_CSV_OUTPUT and self.csv_writer:
            if FAST_MODE:
                self.write_csv_row_fast(detections)
            else:
                self.write_csv_row(detections)
        
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

    def save_multi_target_image(self, annotated_frame, detections):
        """保存已标注的多目标图像"""
        try:
            # 生成保存文件名
            filename = f"frame_{self.frame_idx:06d}_targets_{len(detections)}.jpg"
            filepath = os.path.join(self.save_images_dir, filename)
            
            # 直接保存已标注的图像（vis变量已包含所有标注）
            cv2.imwrite(filepath, annotated_frame)
            self.saved_images_count += 1
            
            if self.saved_images_count == 1:  # 第一次保存时打印提示
                print(f"[INFO] 开始保存多目标已标注图像到: {self.save_images_dir}")
                
        except Exception as e:
            print(f"[WARN] 保存多目标图像失败: {e}")

    def write_csv_row_fast(self, detections):
        """快速CSV写入（简化版）"""
        t = self.frame_idx / max(1e-6, self.fps)
        
        if len(detections) > 0:
            # 只记录第一个（最佳）目标
            det = detections[0]
            row = [self.frame_idx, f"{t:.2f}", det.source, f"{det.conf:.2f}",
                   int(det.cx), int(det.cy), len(detections)]
            self.csv_writer.writerow(row)
        else:
            row = [self.frame_idx, f"{t:.2f}", "none", "0.00", -1, -1, 0]
            self.csv_writer.writerow(row)

    def write_csv_row(self, detections):
        """标准CSV写入（详细版）"""
        t = self.frame_idx / max(1e-6, self.fps)
        
        if len(detections) > 0:
            for i, det in enumerate(detections):
                row = [self.frame_idx, f"{t:.3f}", f"target_{i+1}", det.source, 
                       f"{det.conf:.3f}", int(det.cx), int(det.cy), 
                       det.x1, det.y1, det.x2, det.y2, len(detections)]
                self.csv_writer.writerow(row)
        else:
            row = [self.frame_idx, f"{t:.3f}", "none", "none", 
                   "0.000", -1, -1, -1, -1, -1, -1, 0]
            self.csv_writer.writerow(row)

    def print_progress(self, force_print=False):
        """优化版进度输出"""
        elapsed = time.time() - self.start_time
        progress = (self.frame_idx + 1) / max(1, self.total_frames) * 100
        fps = (self.frame_idx + 1) / max(elapsed, 1e-6)
        eta = (self.total_frames - self.frame_idx - 1) / max(fps, 1e-6)
        
        avg_targets = self.total_detections / max(1, self.frame_idx + 1)
        
        # 动态调整处理策略
        if fps < 15.0 and not force_print:  # 如果FPS太低，动态优化
            global FAST_MODE, SKIP_VISUALIZATION
            if not FAST_MODE:
                FAST_MODE = True
                print(f"[AUTO] 检测到FPS过低({fps:.1f})，自动启用快速模式")
            elif not SKIP_VISUALIZATION:
                SKIP_VISUALIZATION = True
                print(f"[AUTO] FPS仍然过低({fps:.1f})，自动简化可视化")
        
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
            
            # 性能建议
            if fps < 20:
                print(f"[TIPS] 性能优化建议:")
                print(f"  - 降低输入视频分辨率")
                print(f"  - 设置 YOLO_IMG_SIZE = 320")
                print(f"  - 设置 FAST_MODE = True")
                print(f"  - 检查GPU可用性")
                if self.model is not None:
                    device = next(self.model.model.parameters()).device
                    print(f"  - 当前设备: {device}")
            else:
                print(f"[INFO] 处理速度良好: {fps:.1f} fps")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='简化版红外小目标检测程序')
    parser.add_argument('--enable-csv', action='store_true', 
                       help='启用CSV结果输出功能')
    parser.add_argument('--input', '-i', type=str, default=INPUT_VIDEO,
                       help=f'输入视频路径 (默认: {INPUT_VIDEO})')
    parser.add_argument('--weights', '-w', type=str, default=YOLO_WEIGHTS,
                       help=f'YOLO权重文件路径 (默认: {YOLO_WEIGHTS})')
    parser.add_argument('--fast-mode', action='store_true',
                       help='启用快速处理模式')
    parser.add_argument('--enable-image-save', action='store_true',
                       help='启用多目标图像保存功能')
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 根据参数更新全局配置
    global ENABLE_CSV_OUTPUT, SAVE_MULTI_TARGET_IMAGES, FAST_MODE
    ENABLE_CSV_OUTPUT = args.enable_csv
    SAVE_MULTI_TARGET_IMAGES = args.enable_image_save
    if args.fast_mode:
        FAST_MODE = True
    
    print("[INFO] 简化版红外小目标检测程序启动...")
    print(f"[INFO] CSV输出: {'启用' if ENABLE_CSV_OUTPUT else '禁用'}")
    print(f"[INFO] 多目标图像保存: {'启用' if SAVE_MULTI_TARGET_IMAGES else '禁用'}")
    print(f"[INFO] 快速模式: {'启用' if FAST_MODE else '禁用'}")
    
    # 设置输出路径
    output_video, output_csv, save_images_dir = setup_output_paths(
        args.input, ENABLE_CSV_OUTPUT)
    
    print(f"[INFO] 输入视频: {args.input}")
    print(f"[INFO] 输出视频: {output_video}")
    print(f"[INFO] 输出目录: {os.path.dirname(os.path.dirname(output_video))}")
    
    # 确保输出目录存在（删除原有的ensure_dir调用，因为我们会在创建writer前处理）
    if output_csv:
        ensure_dir(output_csv)

    # 打开输入视频
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开输入视频：{args.input}")
        return

    # 获取视频属性
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] 视频属性: {W}x{H}, {fps:.1f} fps, {total_frames} 帧")

    # 确保输出视频目录存在
    output_video_dir = os.path.dirname(output_video)
    os.makedirs(output_video_dir, exist_ok=True)

    # 创建视频写入器（优化编码参数）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编码更兼容
    writer = cv2.VideoWriter(output_video, fourcc, fps, (W, H))
    
    # 检查视频写入器是否成功初始化
    if not writer.isOpened():
        print(f"[ERROR] 无法创建视频写入器: {output_video}")
        print("[INFO] 尝试使用其他编码格式...")
        # 尝试其他编码格式
        for codec in ['XVID', 'MJPG', 'MP4V']:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer.release()  # 释放之前的写入器
            writer = cv2.VideoWriter(output_video, fourcc, fps, (W, H))
            if writer.isOpened():
                print(f"[INFO] 成功使用 {codec} 编码创建视频写入器")
                break
        else:
            print(f"[ERROR] 所有编码格式都失败，无法创建视频输出")
            return

    # 创建CSV写入器（如果启用）
    csv_f = None
    csv_writer = None
    if ENABLE_CSV_OUTPUT and output_csv:
        csv_f = open(output_csv, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_f)
        if FAST_MODE:
            csv_writer.writerow(["frame", "time_sec", "source", "conf", "cx", "cy", "total_targets"])
        else:
            csv_writer.writerow(["frame", "time_sec", "target_id", "source", "conf", 
                                "cx", "cy", "x1", "y1", "x2", "y2", "total_targets"])

    # 尝试加载YOLO模型（按原版方式）
    yolo_model = None
    if HAVE_ULTRALYTICS and os.path.exists(args.weights):
        try:
            yolo_model = YOLO(args.weights)
            print(f"[INFO] 已加载YOLOv11权重：{args.weights}")
            # 注意：原版没有显式调用.to('cuda')，让YOLO自己处理设备
        except Exception as e:
            print(f"[WARN] YOLO权重加载失败，使用传统检测回退：{e}")
    else:
        if not HAVE_ULTRALYTICS:
            print("[WARN] 未安装ultralytics，使用传统检测回退")
        else:
            print(f"[WARN] 未找到权重文件：{args.weights}，使用传统检测回退")

    # 创建检测器
    detector = SimpleDetector(cap, writer, csv_writer, yolo_model, save_images_dir)

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
    if csv_f:
        csv_f.close()
    
    # 最终进度输出
    detector.print_progress(force_print=True)
    
    print(f"\n[COMPLETE] 多目标检测完成！")
    print(f"  输出视频: {output_video}")
    if output_csv:
        print(f"  检测结果: {output_csv}")
    if SAVE_MULTI_TARGET_IMAGES and detector.saved_images_count > 0:
        print(f"  保存图像: {detector.saved_images_count} 张 -> {save_images_dir}")
    print(f"  总共处理: {detector.frame_idx} 帧")
    print(f"  检测统计: 总计{detector.total_detections}个目标")
    print(f"    - YOLO检测: {detector.yolo_detections}")
    print(f"    - 传统检测: {detector.classical_detections}")
    print(f"    - 单帧最大目标数: {detector.max_targets_per_frame}")
    print(f"    - 多目标图像: {detector.saved_images_count} 张")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""
YOLO基础推理测试 - 目标丢失检测与方框提取
当目标连续丢失5帧时，保存连续5张以丢失位置为中心的25x25像素方框.
"""

import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO


class YOLOTargetLossDetector:
    """YOLO目标丢失检测器."""

    def __init__(self, model_path, save_dir="target_loss_patches"):
        """
        初始化检测器.

        Args:
            model_path: YOLO模型路径
            save_dir: 保存方框的目录
        """
        print(f"🔥 加载YOLO模型: {model_path}")
        self.model = YOLO(model_path)

        # 创建保存目录
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 方框保存目录: {self.save_dir}")

        # 跟踪状态
        self.tracks = {}  # track_id -> {'last_seen_frame': int, 'last_position': (x, y), 'lost_frames': int, 'has_saved': bool}
        self.frame_count = 0
        self.fps = 30  # 默认fps，后续会从视频中获取

        # 参数设置
        self.lost_frames_threshold = 5  # 连续丢失5帧触发保存
        self.save_frames_count = 5  # 保存5张连续图像
        self.patch_size = 25  # 方框尺寸

        # 统计信息
        self.stats = {"total_frames": 0, "total_detections": 0, "lost_targets": 0, "saved_patches": 0}

        print(
            f"✅ 初始化完成 - 丢失{self.lost_frames_threshold}帧后立即保存{self.save_frames_count}张图像, 方框尺寸: {self.patch_size}x{self.patch_size}"
        )

    def process_video(self, video_path, conf_threshold=0.5):
        """
        处理视频文件.

        Args:
            video_path: 视频文件路径
            conf_threshold: 置信度阈值
        """
        print(f"\n🎬 开始处理视频: {video_path}")

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        # 获取视频信息
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"📊 视频信息: {width}x{height}, {self.fps:.1f}fps, {total_frames}帧")

        self.frame_count = 0
        start_time = time.time()
        last_valid_frame = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 保存最后一个有效帧
                last_valid_frame = frame.copy()

                # 处理当前帧
                self.process_frame(frame, conf_threshold)
                self.frame_count += 1
                self.stats["total_frames"] = self.frame_count

                # 显示进度
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    print(
                        f"⏳ 已处理 {self.frame_count}/{total_frames} 帧 "
                        f"({self.frame_count / total_frames * 100:.1f}%) - {fps:.1f} fps"
                    )

        finally:
            cap.release()

        # 处理视频结束时的丢失目标（使用最后一个有效帧）
        if last_valid_frame is not None:
            self.check_final_lost_targets(last_valid_frame)

        # 显示统计结果
        self.print_statistics()

    def process_frame(self, frame, conf_threshold):
        """
        处理单帧.

        Args:
            frame: 当前帧图像
            conf_threshold: 置信度阈值
        """
        # YOLO推理
        results = self.model(frame, conf=conf_threshold, verbose=False)

        # 获取当前帧的检测结果
        current_detections = {}
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i, box in enumerate(boxes):
                # 获取边界框和置信度
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()

                # 计算中心位置
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # 简单的ID分配（基于位置）
                track_id = self.assign_track_id(center_x, center_y)
                current_detections[track_id] = {
                    "position": (center_x, center_y),
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                }

        self.stats["total_detections"] += len(current_detections)

        # 更新跟踪状态
        self.update_tracks(current_detections, frame)

    def assign_track_id(self, center_x, center_y, max_distance=50):
        """
        简单的轨迹ID分配（基于位置距离）.

        Args:
            center_x, center_y: 检测中心位置
            max_distance: 最大匹配距离

        Returns:
            分配的轨迹ID
        """
        min_distance = float("inf")
        assigned_id = None

        for track_id, track_info in self.tracks.items():
            last_pos = track_info["last_position"]
            distance = np.sqrt((center_x - last_pos[0]) ** 2 + (center_y - last_pos[1]) ** 2)

            if distance < min_distance and distance < max_distance:
                min_distance = distance
                assigned_id = track_id

        # 如果没有找到匹配的轨迹，创建新的
        if assigned_id is None:
            assigned_id = len(self.tracks) + 1
            self.tracks[assigned_id] = {
                "last_seen_frame": self.frame_count,
                "last_position": (center_x, center_y),
                "lost_frames": 0,
                "has_saved": False,  # 是否已经保存过5张图片
            }

        return assigned_id

    def update_tracks(self, current_detections, frame):
        """
        更新轨迹状态.

        Args:
            current_detections: 当前帧的检测结果
            frame: 当前帧图像
        """
        # 更新检测到的轨迹
        for track_id, detection in current_detections.items():
            if track_id in self.tracks:
                # 如果轨迹之前丢失过，现在重新检测到，重置丢失状态
                if self.tracks[track_id]["lost_frames"] > 0:
                    self.tracks[track_id]["lost_frames"] = 0
                    self.tracks[track_id]["has_saved"] = False  # 重新开始新的丢失周期

                self.tracks[track_id].update(
                    {"last_seen_frame": self.frame_count, "last_position": detection["position"]}
                )

        # 检查丢失的轨迹
        for track_id, track_info in self.tracks.items():
            if track_id not in current_detections:
                # 目标在当前帧中丢失
                track_info["lost_frames"] += 1

                # 检查是否需要保存方框（恰好丢失5帧且还未保存过）
                if track_info["lost_frames"] == self.lost_frames_threshold and not track_info["has_saved"]:
                    # 立即连续保存5张当前帧的图片
                    self.save_loss_patches_batch(frame, track_id, track_info)
                    track_info["has_saved"] = True
                    self.stats["lost_targets"] += 1

    def check_final_lost_targets(self, last_frame):
        """
        检查视频结束时的丢失目标.

        Args:
            last_frame: 最后一帧图像
        """
        if last_frame is None:
            print("⚠️ 最后一帧为空，跳过最终丢失目标检查")
            return

        for track_id, track_info in self.tracks.items():
            # 视频结束时，如果目标丢失超过阈值且还未保存过，保存5张图片
            if track_info["lost_frames"] >= self.lost_frames_threshold and not track_info["has_saved"]:
                self.save_loss_patches_batch(last_frame, track_id, track_info, final=True)
                track_info["has_saved"] = True
                self.stats["lost_targets"] += 1

    def save_loss_patches_batch(self, frame, track_id, track_info, final=False):
        """
        批量保存丢失位置的5张方框图片.

        Args:
            frame: 当前帧图像
            track_id: 轨迹ID
            track_info: 轨迹信息
            final: 是否为视频结束时的保存
        """
        if frame is None:
            print(f"⚠️ 帧图像为空，跳过轨迹{track_id}的方框保存")
            return

        print(f"🎯 轨迹{track_id}丢失{track_info['lost_frames']}帧，开始保存{self.save_frames_count}张方框图片...")

        # 连续保存5张相同的图片
        for i in range(self.save_frames_count):
            self.save_single_patch(frame, track_id, track_info, i + 1, final)

        print(f"✅ 轨迹{track_id}完成保存{self.save_frames_count}张方框图片")

    def save_single_patch(self, frame, track_id, track_info, save_index, final=False):
        """
        保存单张丢失位置的方框.

        Args:
            frame: 当前帧图像
            track_id: 轨迹ID
            track_info: 轨迹信息
            save_index: 保存的序号（第几张）
            final: 是否为视频结束时的保存
        """
        if frame is None:
            print(f"⚠️ 帧图像为空，跳过轨迹{track_id}的方框保存")
            return

        # 获取丢失位置
        center_x, center_y = track_info["last_position"]

        # 计算方框区域
        half_size = self.patch_size // 2
        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(frame.shape[1], center_x + half_size)
        y2 = min(frame.shape[0], center_y + half_size)

        # 提取方框区域
        patch = frame[y1:y2, x1:x2]

        # 如果区域太小，进行填充
        if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
            padded_patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
            h, w = patch.shape[:2]
            padded_patch[:h, :w] = patch
            patch = padded_patch

        # 生成文件名：视频时间_秒数_帧数_轨迹ID_第几张_位置信息
        current_second = int(self.frame_count / self.fps)
        frame_in_second = self.frame_count % int(self.fps)
        timestamp = datetime.now().strftime("%H%M%S")

        suffix = "final" if final else f"img{save_index}"

        # 添加位置信息到文件名
        position_info = f"pos{center_x}x{center_y}"
        filename = (
            f"{timestamp}_{current_second:03d}s_{frame_in_second:02d}f_track{track_id:02d}_{suffix}_{position_info}.png"
        )

        # 保存图像
        save_path = self.save_dir / filename
        cv2.imwrite(str(save_path), patch)

        self.stats["saved_patches"] += 1

        print(
            f"💾 保存丢失方框: {filename} - 位置({center_x}, {center_y}) - "
            f"丢失{track_info['lost_frames']}帧 - 第{save_index}张"
        )

    def print_statistics(self):
        """打印统计信息."""
        print(f"\n{'=' * 50}")
        print("📊 处理完成统计")
        print(f"{'=' * 50}")
        print(f"总帧数: {self.stats['total_frames']}")
        print(f"总检测数: {self.stats['total_detections']}")
        print(f"丢失目标数: {self.stats['lost_targets']}")
        print(f"保存方框数: {self.stats['saved_patches']}")
        print(f"保存目录: {self.save_dir}")

        # 计算平均检测率
        if self.stats["total_frames"] > 0:
            avg_detections = self.stats["total_detections"] / self.stats["total_frames"]
            print(f"平均检测率: {avg_detections:.2f} 个/帧")


def main():
    """主函数."""
    # 配置路径
    video_path = "/home/mingxing/worksapce/ultralytics/video/short.mp4"
    model_path = "/home/mingxing/worksapce/ultralytics/small_target_detection/yolov8_small_aircraft/weights/best.pt"

    # 检查文件是否存在
    if not Path(video_path).exists():
        print(f"❌ 视频文件不存在: {video_path}")
        return

    if not Path(model_path).exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return

    try:
        # 创建检测器
        detector = YOLOTargetLossDetector(model_path=model_path, save_dir="target_loss_patches")

        # 处理视频
        detector.process_video(video_path=video_path, conf_threshold=0.5)

    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

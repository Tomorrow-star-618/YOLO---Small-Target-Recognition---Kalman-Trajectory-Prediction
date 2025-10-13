"""集成相机运动补偿的多目标跟踪器 结合全局运动检测和个体重置机制."""

import os
import sys
import time
from collections import deque

import numpy as np

# 添加主项目路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from camera_motion_compensation.global_motion_detector import GlobalMotionDetector
from camera_motion_compensation.motion_reset_kalman_tracker import MotionResetKalmanTracker
from kalman.enhanced_multi_target_tracker import EnhancedMultiTargetTracker


class MotionCompensatedMultiTracker(EnhancedMultiTargetTracker):
    """
    相机运动补偿多目标跟踪器.

    核心功能：
    1. 全局相机运动检测
    2. 智能卡尔曼重置策略
    3. 个体目标运动分析
    4. 自适应跟踪参数调整
    5. 详细的性能统计
    """

    def __init__(self, max_lost_frames=150, min_hits=1, iou_threshold=0.1, motion_detection_method="optical_flow"):
        """
        初始化运动补偿多目标跟踪器.

        Args:
            max_lost_frames: 最大丢失帧数
            min_hits: 最少命中次数
            iou_threshold: IoU匹配阈值
            motion_detection_method: 运动检测方法
        """
        super().__init__(max_lost_frames, min_hits, iou_threshold)

        # 全局运动检测器
        self.motion_detector = GlobalMotionDetector(method=motion_detection_method)

        # 运动补偿参数
        self.global_motion_compensation = True
        self.individual_reset_enabled = True
        self.adaptive_thresholds = True

        # 历史状态记录
        self.global_motion_history = deque(maxlen=20)
        self.detection_stability_history = deque(maxlen=10)

        # 性能统计
        self.stats = {
            "total_frames": 0,
            "global_motion_events": 0,
            "global_resets": 0,
            "individual_resets": 0,
            "tracking_recoveries": 0,
            "processing_times": deque(maxlen=100),
            "motion_compensation_effects": [],
        }

        # 当前帧状态
        self.current_frame = None
        self.frame_motion_info = None

        print("🚀 运动补偿多目标跟踪器初始化完成")
        print(f"    全局运动检测: {motion_detection_method}")
        print(f"    最大丢失帧数: {max_lost_frames}")
        print(f"    IoU阈值: {iou_threshold}")

    def update(self, detections, frame=None):
        """
        更新跟踪器，集成全局和个体运动补偿.

        Args:
            detections: 检测结果列表 [[x1, y1, x2, y2, conf], ...]
            frame: 当前帧图像（用于运动检测）

        Returns:
            list: 跟踪结果列表
        """
        time.time()

        self.frame_count += 1
        self.stats["total_frames"] += 1
        self.current_frame = frame

        # 1. 全局运动检测
        global_motion_detected = False
        if frame is not None and self.global_motion_compensation:
            motion_result = self.motion_detector.detect_motion(frame)
            is_motion, motion_magnitude, motion_vector, should_reset = motion_result

            self.frame_motion_info = {
                "is_motion": is_motion,
                "magnitude": motion_magnitude,
                "vector": motion_vector.tolist() if hasattr(motion_vector, "tolist") else motion_vector,
                "should_reset": should_reset,
            }

            self.global_motion_history.append(motion_magnitude)

            if should_reset:
                global_motion_detected = True
                self.stats["global_motion_events"] += 1
                print(f"🌍 帧{self.frame_count}: 检测到全局运动 ({motion_magnitude:.1f}px)")

        # 2. 检测稳定性分析
        detection_count = len(detections)
        self.detection_stability_history.append(detection_count)

        # 3. 全局重置策略
        if global_motion_detected and self._should_global_reset():
            return self._perform_global_reset(detections)

        # 4. 标准跟踪流程 + 个体运动补偿
        return self._perform_standard_tracking_with_compensation(detections)

    def _should_global_reset(self):
        """判断是否应该执行全局重置."""
        if not self.frame_motion_info:
            return False

        # 基本条件：全局运动检测器建议重置
        if not self.frame_motion_info["should_reset"]:
            return False

        # 考虑检测稳定性
        if len(self.detection_stability_history) >= 5:
            recent_detections = list(self.detection_stability_history)[-5:]
            detection_stability = np.std(recent_detections) / (np.mean(recent_detections) + 1)

            # 如果检测结果非常不稳定，更容易触发全局重置
            if detection_stability > 0.5:
                return True

        # 考虑运动历史一致性
        if len(self.global_motion_history) >= 3:
            recent_motions = list(self.global_motion_history)[-3:]
            if np.mean(recent_motions) > 30.0:  # 连续的大幅运动
                return True

        # 基于当前运动强度
        return self.frame_motion_info["magnitude"] > 60.0

    def _perform_global_reset(self, detections):
        """执行全局重置."""
        print(f"🔄 帧{self.frame_count}: 执行全局重置 - 清空{len(self.trackers)}个跟踪器")

        # 记录统计信息
        self.stats["global_resets"] += 1

        # 清空所有现有跟踪器
        old_tracker_count = len(self.trackers)
        self.trackers.clear()

        # 为所有检测创建新的跟踪器
        for detection in detections:
            bbox = detection[:4]
            new_tracker = MotionResetKalmanTracker(bbox, max_lost_frames=self.max_lost_frames)
            self.trackers.append(new_tracker)

        print(f"✅ 全局重置完成: {old_tracker_count} → {len(self.trackers)}个跟踪器")

        return self._get_enhanced_track_results()

    def _perform_standard_tracking_with_compensation(self, detections):
        """执行标准跟踪流程，包含个体运动补偿."""
        # 预测阶段
        predicted_bboxes = []
        for tracker in self.trackers:
            predicted_bbox = tracker.predict()
            predicted_bboxes.append(predicted_bbox)

        # 数据关联
        if len(detections) > 0 and len(self.trackers) > 0:
            matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(
                detections, predicted_bboxes, self.iou_threshold
            )
        else:
            matched = []
            unmatched_dets = list(range(len(detections)))
            unmatched_trks = list(range(len(self.trackers)))

        # 更新匹配的跟踪器（包含个体重置逻辑）
        individual_resets = 0
        for match in matched:
            det_idx, trk_idx = match
            detection_bbox = detections[det_idx][:4]

            # 记录重置前状态
            pre_reset_count = getattr(self.trackers[trk_idx], "reset_count", 0)

            # 更新跟踪器（可能触发个体重置）
            self.trackers[trk_idx].update(detection_bbox)

            # 检查是否发生了重置
            post_reset_count = getattr(self.trackers[trk_idx], "reset_count", 0)
            if post_reset_count > pre_reset_count:
                individual_resets += 1

        # 更新统计信息
        if individual_resets > 0:
            self.stats["individual_resets"] += individual_resets
            print(f"📊 帧{self.frame_count}: {individual_resets}个个体重置")

        # 处理未匹配的跟踪器
        for trk_idx in unmatched_trks:
            self.trackers[trk_idx].mark_as_lost()

        # 为未匹配的检测创建新跟踪器
        for det_idx in unmatched_dets:
            bbox = detections[det_idx][:4]
            new_tracker = MotionResetKalmanTracker(bbox, max_lost_frames=self.max_lost_frames)
            self.trackers.append(new_tracker)

        # 移除无效的跟踪器
        valid_trackers = []
        for tracker in self.trackers:
            if tracker.should_delete(self.max_lost_frames):
                # 可能的跟踪恢复统计
                if hasattr(tracker, "reset_count") and tracker.reset_count > 0:
                    self.stats["tracking_recoveries"] += 1
            else:
                valid_trackers.append(tracker)

        self.trackers = valid_trackers

        return self._get_enhanced_track_results()

    def associate_detections_to_trackers(self, detections, predicted_bboxes, iou_threshold):
        """数据关联：将检测结果匹配到跟踪器 使用IoU距离进行匹配."""
        if len(detections) == 0:
            return [], [], list(range(len(predicted_bboxes)))

        if len(predicted_bboxes) == 0:
            return [], list(range(len(detections))), []

        # 计算IoU矩阵
        iou_matrix = np.zeros((len(detections), len(predicted_bboxes)))

        for d, det in enumerate(detections):
            for t, pred in enumerate(predicted_bboxes):
                iou_matrix[d, t] = self._calculate_iou(det[:4], pred)

        # 使用匈牙利算法或简单的贪心算法进行匹配
        matched_indices = []

        # 简单贪心匹配
        used_detections = set()
        used_trackers = set()

        # 按IoU从大到小排序
        matches = []
        for d in range(len(detections)):
            for t in range(len(predicted_bboxes)):
                if iou_matrix[d, t] > iou_threshold:
                    matches.append((iou_matrix[d, t], d, t))

        matches.sort(reverse=True)  # 按IoU降序

        for iou_value, d, t in matches:
            if d not in used_detections and t not in used_trackers:
                matched_indices.append([d, t])
                used_detections.add(d)
                used_trackers.add(t)

        # 未匹配的检测和跟踪器
        unmatched_detections = [d for d in range(len(detections)) if d not in used_detections]
        unmatched_trackers = [t for t in range(len(predicted_bboxes)) if t not in used_trackers]

        return matched_indices, unmatched_detections, unmatched_trackers

    def _calculate_iou(self, box1, box2):
        """计算两个边界框的IoU."""
        # box format: [x1, y1, x2, y2]
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # 计算交集
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def get_comprehensive_stats(self):
        """获取综合统计信息."""
        # 基础统计
        base_stats = {
            "total_frames": self.stats["total_frames"],
            "global_motion_events": self.stats["global_motion_events"],
            "global_resets": self.stats["global_resets"],
            "individual_resets": self.stats["individual_resets"],
            "tracking_recoveries": self.stats["tracking_recoveries"],
        }

        # 运动检测器统计
        motion_stats = self.motion_detector.get_stats()

        # 性能统计
        perf_stats = {}
        if self.stats["processing_times"]:
            perf_stats = {
                "avg_processing_time": f"{np.mean(self.stats['processing_times']):.2f}ms",
                "max_processing_time": f"{np.max(self.stats['processing_times']):.2f}ms",
                "min_processing_time": f"{np.min(self.stats['processing_times']):.2f}ms",
            }

        # 当前跟踪器状态
        tracker_stats = {
            "active_trackers": len(self.trackers),
            "total_resets_by_tracker": sum(getattr(t, "reset_count", 0) for t in self.trackers),
        }

        return {
            "basic": base_stats,
            "motion_detection": motion_stats,
            "performance": perf_stats,
            "trackers": tracker_stats,
            "motion_history_avg": np.mean(self.global_motion_history) if self.global_motion_history else 0.0,
        }

    def enable_adaptive_mode(self, enabled=True):
        """启用/禁用自适应模式."""
        self.adaptive_thresholds = enabled
        for tracker in self.trackers:
            if hasattr(tracker, "adaptive_enabled"):
                tracker.adaptive_enabled = enabled
        print(f"{'✅' if enabled else '❌'} 自适应模式: {'启用' if enabled else '禁用'}")

    def set_global_motion_sensitivity(self, sensitivity):
        """设置全局运动敏感度 (0.5-2.0)."""
        if 0.5 <= sensitivity <= 2.0:
            self.motion_detector.global_motion_threshold /= sensitivity
            self.motion_detector.reset_motion_threshold /= sensitivity
            print(f"🎯 全局运动敏感度设置为: {sensitivity}")
        else:
            print(f"❌ 敏感度应在0.5-2.0之间，当前值: {sensitivity}")

    def reset_all_statistics(self):
        """重置所有统计信息."""
        self.stats = {
            "total_frames": 0,
            "global_motion_events": 0,
            "global_resets": 0,
            "individual_resets": 0,
            "tracking_recoveries": 0,
            "processing_times": deque(maxlen=100),
            "motion_compensation_effects": [],
        }
        self.motion_detector.reset_stats()
        print("📊 所有统计信息已重置")

    def _get_enhanced_track_results(self):
        """获取增强的跟踪结果."""
        tracks = []

        for tracker in self.trackers:
            track_info = tracker.get_track_info()

            # 添加运动补偿相关信息
            if hasattr(self, "frame_motion_info") and self.frame_motion_info:
                track_info["global_motion"] = self.frame_motion_info

            # 添加个体运动统计
            if hasattr(tracker, "get_reset_statistics"):
                reset_stats = tracker.get_reset_statistics()
                track_info["reset_statistics"] = reset_stats

            tracks.append(track_info)

        return tracks

"""
带运动重置功能的增强卡尔曼跟踪器
结合个体运动检测和全局运动补偿
"""

import numpy as np
import uuid
from collections import deque
import sys
import os

# 添加主项目路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kalman.enhanced_aircraft_kalman_tracker import AircraftKalmanTracker

class MotionResetKalmanTracker(AircraftKalmanTracker):
    """
    运动重置卡尔曼跟踪器
    
    核心功能：
    1. 检测个体目标的位置跳跃
    2. 检测相机运动导致的整体位移
    3. 智能重置卡尔曼滤波器状态
    4. 保持轨迹ID的连续性
    5. 提供详细的重置统计信息
    """
    
    def __init__(self, initial_bbox, track_id=None, max_lost_frames=150):
        """
        初始化运动重置跟踪器
        
        Args:
            initial_bbox: [x1, y1, x2, y2] 初始边界框
            track_id: 跟踪ID
            max_lost_frames: 最大丢失帧数
        """
        super().__init__(initial_bbox, track_id, max_lost_frames)
        
        # 运动检测参数
        self.position_history = deque(maxlen=8)  # 位置历史
        self.velocity_smoothing = deque(maxlen=5)  # 速度平滑
        self.bbox_history = deque(maxlen=5)  # 边界框历史
        
        # 运动检测阈值
        self.jump_threshold = 40.0       # 位置跳跃阈值（像素）
        self.velocity_threshold = 60.0    # 速度突变阈值（像素/帧）
        self.size_change_threshold = 0.3  # 尺寸变化阈值（比例）
        self.reset_cooldown = 15         # 重置冷却期（帧）
        
        # 重置相关统计
        self.reset_count = 0
        self.last_reset_frame = -999
        self.reset_reasons = []  # 记录重置原因
        self.motion_scores = deque(maxlen=10)  # 运动评分历史
        
        # 自适应参数
        self.adaptive_enabled = True
        self.confidence_factor = 1.0
        self.motion_consistency = 0.0
        
        # 记录初始状态
        center = self._get_bbox_center(initial_bbox)
        self.position_history.append(center)
        self.bbox_history.append(initial_bbox)
        
        print(f"🎯 运动重置跟踪器初始化: {self.track_id}")
    
    def _get_bbox_center(self, bbox):
        """获取边界框中心点"""
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
    
    def _get_bbox_size(self, bbox):
        """获取边界框尺寸"""
        x1, y1, x2, y2 = bbox
        return np.array([x2 - x1, y2 - y1])
    
    def _detect_position_jump(self, new_center):
        """检测位置跳跃"""
        if len(self.position_history) < 2:
            return False, 0.0, "insufficient_history"
        
        # 计算最近几帧的平均位置
        recent_positions = list(self.position_history)[-3:]
        avg_position = np.mean(recent_positions, axis=0)
        
        # 计算当前位置与平均位置的距离
        distance = np.linalg.norm(new_center - avg_position)
        
        # 检测是否为跳跃
        is_jump = distance > self.jump_threshold
        
        # 计算运动评分
        motion_score = min(distance / self.jump_threshold, 3.0)
        self.motion_scores.append(motion_score)
        
        reason = f"position_jump_{distance:.1f}px" if is_jump else "normal_motion"
        
        return is_jump, distance, reason
    
    def _detect_velocity_change(self, new_center):
        """检测速度突变"""
        if len(self.position_history) < 3:
            return False, 0.0, "insufficient_velocity_history"
        
        # 计算最近的速度
        positions = list(self.position_history)[-3:] + [new_center]
        velocities = []
        
        for i in range(1, len(positions)):
            vel = np.linalg.norm(positions[i] - positions[i-1])
            velocities.append(vel)
        
        if len(velocities) < 2:
            return False, 0.0, "insufficient_velocity_data"
        
        # 检测速度突变
        current_velocity = velocities[-1]
        avg_velocity = np.mean(velocities[:-1])
        
        velocity_change = abs(current_velocity - avg_velocity)
        is_sudden_change = velocity_change > self.velocity_threshold
        
        reason = f"velocity_change_{velocity_change:.1f}px/f" if is_sudden_change else "normal_velocity"
        
        return is_sudden_change, velocity_change, reason
    
    def _detect_size_change(self, new_bbox):
        """检测尺寸突变"""
        if len(self.bbox_history) < 2:
            return False, 0.0, "insufficient_size_history"
        
        # 计算尺寸变化
        current_size = self._get_bbox_size(new_bbox)
        prev_size = self._get_bbox_size(self.bbox_history[-1])
        
        # 避免除零
        prev_size = np.maximum(prev_size, 1.0)
        size_ratio = current_size / prev_size
        
        # 检测显著的尺寸变化
        max_ratio_change = max(abs(size_ratio[0] - 1.0), abs(size_ratio[1] - 1.0))
        is_size_jump = max_ratio_change > self.size_change_threshold
        
        reason = f"size_change_{max_ratio_change:.2f}" if is_size_jump else "normal_size"
        
        return is_size_jump, max_ratio_change, reason
    
    def _calculate_motion_consistency(self):
        """计算运动一致性"""
        if len(self.motion_scores) < 3:
            return 0.0
        
        scores = list(self.motion_scores)
        # 一致性 = 1 - 方差/均值（归一化）
        mean_score = np.mean(scores)
        if mean_score > 0:
            variance = np.var(scores)
            consistency = max(0.0, 1.0 - variance / (mean_score + 0.1))
        else:
            consistency = 1.0
        
        return consistency
    
    def _should_reset_kalman(self, new_bbox):
        """
        综合判断是否应该重置卡尔曼滤波器
        
        Returns:
            tuple: (should_reset, reset_reasons, confidence)
        """
        # 检查重置冷却期
        frames_since_reset = self.age - self.last_reset_frame
        if frames_since_reset < self.reset_cooldown:
            return False, ["in_cooldown"], 0.0
        
        new_center = self._get_bbox_center(new_bbox)
        reset_reasons = []
        confidence_factors = []
        
        # 1. 位置跳跃检测
        is_jump, jump_distance, jump_reason = self._detect_position_jump(new_center)
        if is_jump:
            reset_reasons.append(jump_reason)
            confidence_factors.append(min(jump_distance / self.jump_threshold, 2.0))
        
        # 2. 速度突变检测
        is_vel_change, vel_change, vel_reason = self._detect_velocity_change(new_center)
        if is_vel_change:
            reset_reasons.append(vel_reason)
            confidence_factors.append(min(vel_change / self.velocity_threshold, 2.0))
        
        # 3. 尺寸突变检测
        is_size_change, size_change, size_reason = self._detect_size_change(new_bbox)
        if is_size_change:
            reset_reasons.append(size_reason)
            confidence_factors.append(size_change / self.size_change_threshold)
        
        # 4. 计算综合重置置信度
        if confidence_factors:
            reset_confidence = np.mean(confidence_factors)
            
            # 考虑运动一致性
            self.motion_consistency = self._calculate_motion_consistency()
            if self.motion_consistency < 0.3:  # 运动不一致时更容易触发重置
                reset_confidence *= 1.5
            
            # 自适应阈值
            if self.adaptive_enabled:
                # 如果最近重置较频繁，提高阈值
                if self.reset_count > 0 and frames_since_reset < 50:
                    reset_confidence *= 0.8
            
            should_reset = reset_confidence > 1.0
        else:
            should_reset = False
            reset_confidence = 0.0
        
        return should_reset, reset_reasons, reset_confidence
    
    def _reset_kalman_filter(self, new_bbox, reasons, confidence):
        """重置卡尔曼滤波器"""
        print(f"🔄 [{self.track_id}] 卡尔曼重置 - 置信度: {confidence:.2f}")
        print(f"    重置原因: {', '.join(reasons)}")
        
        # 保存重置信息
        self.reset_count += 1
        self.last_reset_frame = self.age
        self.reset_reasons.append({
            'frame': self.age,
            'reasons': reasons,
            'confidence': confidence,
            'motion_consistency': self.motion_consistency
        })
        
        # 重置卡尔曼滤波器状态
        bbox_state = self.bbox_to_state(new_bbox)
        self.x[:4] = bbox_state
        self.x[4:] = 0  # 速度归零
        
        # 调整协方差矩阵
        self.P[4:, 4:] *= 100.0  # 增加速度不确定性
        self.P[:4, :4] *= 5.0    # 适度增加位置不确定性
        
        # 清空相关历史
        center = self._get_bbox_center(new_bbox)
        self.trajectory_history.clear()
        self.trajectory_history.append((center[0], center[1]))
        
        if hasattr(self, 'velocity_history'):
            self.velocity_history.clear()
        if hasattr(self, 'position_history'):
            self.position_history.clear()
            self.position_history.append(center)
        
        self.motion_scores.clear()
        
        # 更新轨迹状态
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        
        print(f"✅ [{self.track_id}] 重置完成 (第{self.reset_count}次)")
    
    def update(self, bbox):
        """
        更新跟踪器，包含智能重置逻辑
        
        Args:
            bbox: [x1, y1, x2, y2] 检测边界框
        """
        # 检查是否需要重置
        should_reset, reasons, confidence = self._should_reset_kalman(bbox)
        
        if should_reset:
            # 重置卡尔曼滤波器
            self._reset_kalman_filter(bbox, reasons, confidence)
        else:
            # 正常更新
            super().update(bbox)
        
        # 更新历史记录
        new_center = self._get_bbox_center(bbox)
        self.position_history.append(new_center)
        self.bbox_history.append(bbox)
    
    def predict(self):
        """预测，考虑重置后的状态调整"""
        predicted_bbox = super().predict()
        
        # 重置后短期内降低预测权重
        frames_since_reset = self.age - self.last_reset_frame
        if frames_since_reset < 10:  # 重置后10帧内
            # 轻微调整预测结果，使其更保守
            if len(self.position_history) > 0:
                last_center = self.position_history[-1]
                pred_center = self._get_bbox_center(predicted_bbox)
                
                # 混合上一帧位置和预测位置
                blend_factor = min(frames_since_reset / 10.0, 1.0)
                adjusted_center = (1 - blend_factor) * last_center + blend_factor * pred_center
                
                # 重构边界框
                size = self._get_bbox_size(predicted_bbox)
                predicted_bbox = [
                    adjusted_center[0] - size[0]/2,
                    adjusted_center[1] - size[1]/2,
                    adjusted_center[0] + size[0]/2,
                    adjusted_center[1] + size[1]/2
                ]
        
        return predicted_bbox
    
    def get_track_info(self):
        """获取跟踪信息，包含重置统计"""
        track_info = super().get_track_info()
        
        # 添加重置相关信息
        track_info['reset_count'] = self.reset_count
        frames_since_reset = self.age - self.last_reset_frame
        track_info['frames_since_reset'] = frames_since_reset
        track_info['motion_consistency'] = f"{self.motion_consistency:.2f}"
        
        # 状态后缀显示
        if self.reset_count > 0:
            if frames_since_reset < 20:
                track_info['status_suffix'] = f" | 重置({frames_since_reset}f前)"
            elif self.reset_count == 1:
                track_info['status_suffix'] = f" | 已重置1次"
            else:
                track_info['status_suffix'] = f" | 已重置{self.reset_count}次"
        else:
            track_info['status_suffix'] = ""
        
        return track_info
    
    def get_reset_statistics(self):
        """获取详细的重置统计信息"""
        if not self.reset_reasons:
            return {"total_resets": 0, "details": []}
        
        # 按原因分类统计
        reason_counts = {}
        for reset_info in self.reset_reasons:
            for reason in reset_info['reasons']:
                key = reason.split('_')[0]  # 取主要原因类型
                reason_counts[key] = reason_counts.get(key, 0) + 1
        
        return {
            "total_resets": self.reset_count,
            "reason_distribution": reason_counts,
            "avg_confidence": np.mean([r['confidence'] for r in self.reset_reasons]),
            "avg_motion_consistency": np.mean([r['motion_consistency'] for r in self.reset_reasons]),
            "details": self.reset_reasons[-5:]  # 最近5次重置详情
        }

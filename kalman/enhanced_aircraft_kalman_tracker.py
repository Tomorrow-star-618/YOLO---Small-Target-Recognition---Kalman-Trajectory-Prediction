import numpy as np
import cv2
from collections import deque
import uuid
import math

class AircraftKalmanTracker:
    """
    优化版飞机目标卡尔曼滤波跟踪器
    
    专门针对飞机直线运动特性和云层遮挡设计，支持长时间预测（5-15秒）
    
    核心优化：
    - 长期预测：支持300-450帧（10-15秒@30fps）
    - 轨迹记忆：基于历史运动模式预测
    - 自适应噪声：根据运动稳定性动态调整
    - 智能预测：结合历史轨迹和运动趋势
    
    状态向量: [center_x, center_y, width, height, vx, vy, vw, vh]
    观测向量: [center_x, center_y, width, height]
    """
    
    def __init__(self, initial_bbox, track_id=None, max_lost_frames=450):
        """
        初始化优化版卡尔曼滤波器
        
        Args:
            initial_bbox: [x1, y1, x2, y2] 初始边界框
            track_id: 轨迹ID
            max_lost_frames: 最大丢失帧数（默认450帧=15秒@30fps）
        """
        self.track_id = track_id or str(uuid.uuid4())[:8]
        self.age = 0
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0
        
        # 卡尔曼滤波器状态
        self.state_dim = 8  # [x, y, w, h, vx, vy, vw, vh]
        self.measure_dim = 4  # [x, y, w, h]
        self.x = np.zeros(self.state_dim, dtype=float)
        
        # 协方差矩阵 - 针对飞机运动优化
        self.P = np.eye(self.state_dim)
        self.P[:4, :4] *= 50.0     # 位置初始不确定性
        self.P[4:6, 4:6] *= 100.0  # 速度初始不确定性
        self.P[6:, 6:] *= 1.0      # 尺寸变化不确定性
        
        # 状态转移矩阵 (匀速直线运动模型)
        self.F = np.eye(self.state_dim)
        self.F[0, 4] = 1  # x = x + vx * dt
        self.F[1, 5] = 1  # y = y + vy * dt
        self.F[2, 6] = 1  # w = w + vw * dt
        self.F[3, 7] = 1  # h = h + vh * dt
        
        # 观测矩阵
        self.H = np.zeros((self.measure_dim, self.state_dim))
        self.H[0, 0] = 1  # 观测center_x
        self.H[1, 1] = 1  # 观测center_y  
        self.H[2, 2] = 1  # 观测width
        self.H[3, 3] = 1  # 观测height
        
        # 过程噪声 - 飞机运动相对平稳
        self.Q = np.eye(self.state_dim)
        self.Q[:2, :2] *= 0.1      # 位置过程噪声小
        self.Q[2:4, 2:4] *= 0.01   # 尺寸变化很小
        self.Q[4:6, 4:6] *= 0.1    # 速度变化噪声
        self.Q[6:, 6:] *= 0.001    # 尺寸速度噪声
        
        # 观测噪声
        self.R = np.eye(self.measure_dim) * 10.0
        
        # 初始化状态
        bbox_state = self.bbox_to_state(initial_bbox)
        self.x[:4] = bbox_state
        
        # 增强轨迹记忆系统
        self.trajectory_history = deque(maxlen=150)  # 更长的历史记录
        self.velocity_history = deque(maxlen=50)     # 速度历史
        self.position_history = deque(maxlen=100)    # 位置历史
        
        # 运动分析数据
        self.motion_analysis = {
            'velocity_avg': np.array([0.0, 0.0]),
            'velocity_std': np.array([0.0, 0.0]),
            'direction': 0.0,
            'speed': 0.0,
            'stability_score': 0.0,
            'prediction_confidence': 0.0
        }
        
        # 预测状态管理
        self.is_lost = False
        self.lost_frames = 0
        self.max_lost_frames = max_lost_frames
        self.lost_start_state = None
        self.lost_start_time = None
        
        # 记录初始位置
        self.trajectory_history.append((bbox_state[0], bbox_state[1]))
        self.position_history.append(bbox_state[:2])
        
    def bbox_to_state(self, bbox):
        """
        边界框转换为状态向量
        
        Args:
            bbox: [x1, y1, x2, y2] 边界框
            
        Returns:
            state: [center_x, center_y, width, height] 状态向量前4位
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        return np.array([center_x, center_y, width, height])
    
    def state_to_bbox(self, state):
        """
        状态向量转换为边界框
        
        Args:
            state: 状态向量
            
        Returns:
            bbox: [x1, y1, x2, y2] 边界框
        """
        center_x, center_y, width, height = state[:4]
        x1 = center_x - width / 2.0
        y1 = center_y - height / 2.0  
        x2 = center_x + width / 2.0
        y2 = center_y + height / 2.0
        return np.array([x1, y1, x2, y2])
    
    def analyze_motion_pattern(self):
        """
        深度分析运动模式，为长期预测提供依据
        """
        if len(self.velocity_history) < 5:
            return
        
        # 速度统计分析
        velocities = np.array(list(self.velocity_history))
        self.motion_analysis['velocity_avg'] = np.mean(velocities, axis=0)
        self.motion_analysis['velocity_std'] = np.std(velocities, axis=0)
        
        # 速度大小和方向
        avg_vx, avg_vy = self.motion_analysis['velocity_avg']
        self.motion_analysis['speed'] = np.sqrt(avg_vx**2 + avg_vy**2)
        self.motion_analysis['direction'] = np.arctan2(avg_vy, avg_vx)
        
        # 运动稳定性评分
        speed_stability = 1.0 / (1.0 + np.mean(self.motion_analysis['velocity_std']))
        direction_consistency = self._calculate_direction_consistency()
        self.motion_analysis['stability_score'] = (speed_stability + direction_consistency) / 2.0
        
        # 预测置信度（基于历史数据量和稳定性）
        data_confidence = min(len(self.velocity_history) / 30.0, 1.0)
        self.motion_analysis['prediction_confidence'] = (
            self.motion_analysis['stability_score'] * data_confidence
        )
    
    def _calculate_direction_consistency(self):
        """计算方向一致性"""
        if len(self.velocity_history) < 3:
            return 0.0
        
        velocities = np.array(list(self.velocity_history))
        directions = np.arctan2(velocities[:, 1], velocities[:, 0])
        
        # 计算方向变化的标准差
        direction_changes = np.diff(directions)
        # 处理角度跳跃
        direction_changes = np.array([
            change if abs(change) < np.pi else change - 2*np.pi*np.sign(change)
            for change in direction_changes
        ])
        
        direction_std = np.std(direction_changes)
        return 1.0 / (1.0 + direction_std * 10)
    
    def predict(self):
        """
        标准卡尔曼预测步骤
        
        Returns:
            predicted_bbox: [x1, y1, x2, y2] 预测边界框
        """
        # 预测步骤
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # 更新计数器
        self.age += 1
        self.time_since_update += 1
        
        # 记录预测位置
        pred_center = (self.x[0], self.x[1])
        self.trajectory_history.append(pred_center)
        
        return self.state_to_bbox(self.x)
    
    def enhanced_long_term_predict(self, frames_ahead=1):
        """
        增强版长期预测，专门用于目标丢失后的预测
        
        Args:
            frames_ahead: 预测未来多少帧
            
        Returns:
            predicted_bbox: [x1, y1, x2, y2] 预测边界框
            confidence: 预测置信度 [0, 1]
        """
        if frames_ahead <= 1:
            return self.predict(), 1.0
        
        # 分析当前运动模式
        self.analyze_motion_pattern()
        
        # 基于运动分析的长期预测
        if self.motion_analysis['prediction_confidence'] > 0.3:
            # 高置信度：使用历史运动模式
            pred_state = self.x.copy()
            
            # 使用平均速度进行外推
            avg_velocity = self.motion_analysis['velocity_avg']
            pred_state[0] += avg_velocity[0] * frames_ahead
            pred_state[1] += avg_velocity[1] * frames_ahead
            
            # 尺寸保持相对稳定
            pred_state[2:4] = self.x[2:4]
            
            # 置信度随时间衰减
            time_decay = max(0.1, 1.0 - frames_ahead / self.max_lost_frames)
            confidence = self.motion_analysis['prediction_confidence'] * time_decay
            
        else:
            # 低置信度：使用标准卡尔曼预测
            pred_state = self.x.copy()
            for _ in range(frames_ahead):
                pred_state = self.F @ pred_state
            
            confidence = max(0.1, 1.0 - frames_ahead / (self.max_lost_frames * 0.5))
        
        return self.state_to_bbox(pred_state), confidence
    
    def update(self, bbox):
        """
        卡尔曼滤波更新步骤，增强版本
        
        Args:
            bbox: [x1, y1, x2, y2] 观测边界框
        """
        # 记录更新前的速度
        prev_velocity = self.x[4:6].copy()
        
        # 重置丢失状态
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        
        if self.is_lost:
            # 从丢失状态恢复
            lost_time = self.lost_frames  # 保存丢失时间用于日志
            self.is_lost = False
            self.lost_frames = 0
            self.lost_start_state = None
            self.lost_start_time = None
            print(f"目标 {self.track_id} 重新检测到，丢失了 {lost_time} 帧")
        
        # 转换观测
        z = self.bbox_to_state(bbox)
        
        # 卡尔曼更新
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        I_KH = np.eye(self.state_dim) - K @ self.H
        self.P = I_KH @ self.P
        
        # 记录运动数据
        current_velocity = self.x[4:6]
        self.velocity_history.append(current_velocity.copy())
        
        current_position = self.x[:2]
        self.position_history.append(current_position.copy())
        
        # 更新轨迹
        center = (self.x[0], self.x[1])
        self.trajectory_history.append(center)
        
        # 分析运动模式
        self.analyze_motion_pattern()
    
    def mark_as_lost(self):
        """
        标记目标为丢失状态，记录丢失时的详细信息
        """
        if not self.is_lost:
            self.is_lost = True
            self.lost_frames = 0
            self.lost_start_state = self.x.copy()
            self.lost_start_time = self.age
            
            # 打印丢失信息
            pos = self.lost_start_state[:2]
            vel = self.lost_start_state[4:6]
            confidence = self.motion_analysis.get('prediction_confidence', 0.0)
            print(f"目标 {self.track_id} 丢失 - 位置: [{pos[0]:.1f}, {pos[1]:.1f}], "
                  f"速度: [{vel[0]:.2f}, {vel[1]:.2f}], 运动置信度: {confidence:.2f}")
        
        self.lost_frames += 1
        self.hit_streak = 0
    
    def get_lost_prediction(self):
        """
        获取目标丢失期间的预测位置（核心优化功能）
        
        Returns:
            predicted_bbox: [x1, y1, x2, y2] 预测位置
            confidence: 预测置信度 [0, 1]
        """
        if not self.is_lost:
            return self.state_to_bbox(self.x), 1.0
        
        # 使用增强长期预测
        pred_bbox, confidence = self.enhanced_long_term_predict(frames_ahead=self.lost_frames)
        
        return pred_bbox, confidence
    
    def get_track_info(self):
        """
        获取完整的跟踪信息
        
        核心逻辑：基于time_since_update判断状态
        - time_since_update == 0: 刚刚检测到，显示绿色检测框
        - time_since_update > 0: 检测丢失，显示橙色预测框
        
        Returns:
            track_info: 包含位置、状态、置信度等信息的字典
        """
        # 关键变更：基于time_since_update判断状态，实现真正的交替显示
        is_predicted_state = self.time_since_update > 0
        
        if is_predicted_state:
            # 预测状态：显示橙色预测框
            if self.is_lost:
                # 长期丢失，使用增强预测
                pred_bbox, confidence = self.get_lost_prediction()
            else:
                # 短期丢失，使用标准预测
                pred_bbox = self.state_to_bbox(self.x)
                time_decay = max(0.3, 1.0 - self.time_since_update / 60.0)  # 2秒内保持较高置信度
                confidence = time_decay
            status = 'predicted'
        else:
            # 检测状态：显示绿色检测框
            pred_bbox = self.state_to_bbox(self.x)
            confidence = 1.0
            status = 'detected'
        
        return {
            'track_id': self.track_id,
            'bbox': pred_bbox,
            'confidence': confidence,
            'status': status,
            'age': self.age,
            'hits': self.hits,
            'hit_streak': self.hit_streak,
            'time_since_update': self.time_since_update,
            'lost_frames': self.time_since_update,  # 使用time_since_update作为丢失帧数
            'is_lost': is_predicted_state,  # 向后兼容
            'trajectory': list(self.trajectory_history)[-30:],  # 最近30个点
            'velocity': self.x[4:6],
            'motion_confidence': self.motion_analysis.get('prediction_confidence', 0.0),
            'is_stable_motion': self.motion_analysis.get('stability_score', 0.0) > 0.5,
            'speed': self.motion_analysis.get('speed', 0.0),
            'direction': self.motion_analysis.get('direction', 0.0)
        }
    
    def should_delete(self, max_lost_frames):
        """
        判断是否应该删除此跟踪器
        
        Args:
            max_lost_frames: 最大允许丢失帧数
            
        Returns:
            bool: 是否应该删除
        """
        # 超过最大丢失时间
        if self.time_since_update > max_lost_frames:
            return True
        
        # 新轨迹稍微宽松一些删除条件，让预测效果更明显
        if self.age < 5 and self.hit_streak == 0 and self.time_since_update > 15:
            return True
        elif self.age < 10 and self.hit_streak <= 1 and self.time_since_update > 30:
            return True
            
        return False

# 为了兼容性，提供别名
EnhancedAircraftKalmanTracker = AircraftKalmanTracker

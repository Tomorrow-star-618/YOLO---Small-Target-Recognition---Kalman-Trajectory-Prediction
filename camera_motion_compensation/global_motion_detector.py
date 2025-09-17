"""
全局相机运动检测器
用于检测画面整体的相机运动，为卡尔曼滤波器重置提供依据
"""

import cv2
import numpy as np
from collections import deque
import time

class GlobalMotionDetector:
    """
    全局相机运动检测器
    
    功能：
    1. 基于特征点匹配检测相机运动
    2. 基于光流场检测全局运动
    3. 提供多种运动检测策略
    4. 输出运动向量和强度
    """
    
    def __init__(self, method='optical_flow'):
        """
        初始化全局运动检测器
        
        Args:
            method: 检测方法 ('optical_flow', 'feature_matching', 'hybrid')
        """
        self.method = method
        self.prev_frame = None
        self.prev_gray = None
        
        # 运动历史记录
        self.motion_history = deque(maxlen=10)
        self.motion_vectors = deque(maxlen=5)
        
        # 阈值设置
        self.global_motion_threshold = 30.0  # 全局运动检测阈值
        self.reset_motion_threshold = 50.0   # 触发重置的阈值
        self.consistency_threshold = 0.7     # 运动一致性阈值
        
        # 光流参数
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # 特征点检测参数
        self.feature_params = dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=15,
            blockSize=7
        )
        
        # 统计信息
        self.stats = {
            'total_detections': 0,
            'motion_events': 0,
            'reset_triggers': 0,
            'avg_motion_magnitude': 0.0
        }
        
        print(f"✅ 全局运动检测器初始化完成 - 方法: {method}")
    
    def detect_motion(self, frame):
        """
        检测全局运动
        
        Args:
            frame: 当前帧图像
            
        Returns:
            tuple: (is_motion, motion_magnitude, motion_vector, should_reset)
        """
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return False, 0.0, np.array([0.0, 0.0]), False
        
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 根据选择的方法进行运动检测
        if self.method == 'optical_flow':
            result = self._detect_by_optical_flow(curr_gray)
        elif self.method == 'feature_matching':
            result = self._detect_by_feature_matching(curr_gray)
        else:  # hybrid
            result = self._detect_by_hybrid_method(curr_gray)
        
        # 更新历史
        self.prev_frame = frame.copy()
        self.prev_gray = curr_gray.copy()
        
        # 更新统计信息
        self.stats['total_detections'] += 1
        is_motion, motion_magnitude, motion_vector, should_reset = result
        
        if is_motion:
            self.stats['motion_events'] += 1
        if should_reset:
            self.stats['reset_triggers'] += 1
        
        # 更新平均运动强度
        self.stats['avg_motion_magnitude'] = (
            (self.stats['avg_motion_magnitude'] * (self.stats['total_detections'] - 1) + motion_magnitude) /
            self.stats['total_detections']
        )
        
        return result
    
    def _detect_by_optical_flow(self, curr_gray):
        """基于光流的运动检测"""
        # 检测角点
        corners = cv2.goodFeaturesToTrack(self.prev_gray, **self.feature_params)
        
        if corners is None or len(corners) < 20:
            return False, 0.0, np.array([0.0, 0.0]), False
        
        # 计算光流
        next_corners, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, corners, None, **self.lk_params
        )
        
        # 筛选有效匹配
        if status is not None:
            good_matches = status.flatten() == 1
        else:
            return False, 0.0, np.array([0.0, 0.0]), False
            
        if np.sum(good_matches) < 10:
            return False, 0.0, np.array([0.0, 0.0]), False
        
        # 确保形状匹配
        prev_points = corners[good_matches].reshape(-1, 2)
        next_points = next_corners[good_matches].reshape(-1, 2)
        
        # 计算运动向量
        motion_vectors = next_points - prev_points
        
        # 使用RANSAC去除异常值
        if len(motion_vectors) > 8:
            median_motion = np.median(motion_vectors, axis=0)
            distances = np.linalg.norm(motion_vectors - median_motion, axis=1)
            inliers = distances < np.percentile(distances, 75)
            
            if np.sum(inliers) > 5:
                global_motion_vector = np.mean(motion_vectors[inliers], axis=0)
                motion_magnitude = np.linalg.norm(global_motion_vector)
                
                # 更新历史
                self.motion_history.append(motion_magnitude)
                self.motion_vectors.append(global_motion_vector)
                
                # 判断是否为显著运动
                is_motion = motion_magnitude > self.global_motion_threshold
                should_reset = motion_magnitude > self.reset_motion_threshold
                
                # 考虑运动一致性
                if len(self.motion_vectors) >= 3:
                    recent_vectors = list(self.motion_vectors)[-3:]
                    consistency = self._calculate_motion_consistency(recent_vectors)
                    if consistency > self.consistency_threshold and is_motion:
                        should_reset = should_reset or motion_magnitude > self.global_motion_threshold * 1.5
                
                return is_motion, motion_magnitude, global_motion_vector, should_reset
        
        return False, 0.0, np.array([0.0, 0.0]), False
    
    def _detect_by_feature_matching(self, curr_gray):
        """基于特征匹配的运动检测"""
        # 使用ORB特征检测器
        orb = cv2.ORB_create(nfeatures=500)
        
        # 检测关键点和描述符
        kp1, des1 = orb.detectAndCompute(self.prev_gray, None)
        kp2, des2 = orb.detectAndCompute(curr_gray, None)
        
        if des1 is None or des2 is None or len(des1) < 20 or len(des2) < 20:
            return False, 0.0, np.array([0.0, 0.0]), False
        
        # 特征匹配
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < 20:
            return False, 0.0, np.array([0.0, 0.0]), False
        
        # 提取匹配点
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # 使用RANSAC估计单应性矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, 
                                   cv2.RANSAC, 5.0, maxIters=2000)
        
        if H is not None and mask is not None:
            inlier_ratio = np.sum(mask) / len(mask)
            
            if inlier_ratio > 0.3:  # 至少30%的内点
                # 从单应性矩阵提取平移分量
                translation = H[:2, 2]
                motion_magnitude = np.linalg.norm(translation)
                
                self.motion_history.append(motion_magnitude)
                
                is_motion = motion_magnitude > self.global_motion_threshold
                should_reset = motion_magnitude > self.reset_motion_threshold
                
                return is_motion, motion_magnitude, translation, should_reset
        
        return False, 0.0, np.array([0.0, 0.0]), False
    
    def _detect_by_hybrid_method(self, curr_gray):
        """混合方法：结合光流和特征匹配"""
        # 获取两种方法的结果
        flow_result = self._detect_by_optical_flow(curr_gray)
        feature_result = self._detect_by_feature_matching(curr_gray)
        
        # 如果其中一种方法检测到显著运动，则认为有运动
        is_motion = flow_result[0] or feature_result[0]
        
        # 运动强度取两者平均值（如果都有效）
        if flow_result[1] > 0 and feature_result[1] > 0:
            motion_magnitude = (flow_result[1] + feature_result[1]) / 2.0
            motion_vector = (flow_result[2] + feature_result[2]) / 2.0
        elif flow_result[1] > 0:
            motion_magnitude = flow_result[1]
            motion_vector = flow_result[2]
        else:
            motion_magnitude = feature_result[1]
            motion_vector = feature_result[2]
        
        # 重置判断：两种方法都认为需要重置，或者运动强度很大
        should_reset = (flow_result[3] and feature_result[3]) or motion_magnitude > self.reset_motion_threshold * 1.2
        
        return is_motion, motion_magnitude, motion_vector, should_reset
    
    def _calculate_motion_consistency(self, vectors):
        """计算运动向量的一致性"""
        if len(vectors) < 2:
            return 0.0
        
        # 计算向量间的角度差异
        angles = [np.arctan2(v[1], v[0]) for v in vectors]
        angle_diffs = []
        
        for i in range(1, len(angles)):
            diff = abs(angles[i] - angles[i-1])
            # 处理角度跳跃
            if diff > np.pi:
                diff = 2 * np.pi - diff
            angle_diffs.append(diff)
        
        # 一致性 = 1 - 平均角度差异/π
        avg_angle_diff = np.mean(angle_diffs)
        consistency = max(0.0, 1.0 - avg_angle_diff / np.pi)
        
        return consistency
    
    def get_stats(self):
        """获取检测统计信息"""
        if self.stats['total_detections'] > 0:
            motion_rate = self.stats['motion_events'] / self.stats['total_detections']
            reset_rate = self.stats['reset_triggers'] / self.stats['total_detections']
        else:
            motion_rate = reset_rate = 0.0
        
        return {
            'total_detections': self.stats['total_detections'],
            'motion_events': self.stats['motion_events'],
            'reset_triggers': self.stats['reset_triggers'],
            'motion_detection_rate': f"{motion_rate:.1%}",
            'reset_trigger_rate': f"{reset_rate:.1%}",
            'avg_motion_magnitude': f"{self.stats['avg_motion_magnitude']:.2f}px"
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_detections': 0,
            'motion_events': 0,
            'reset_triggers': 0,
            'avg_motion_magnitude': 0.0
        }
        print("📊 全局运动检测器统计信息已重置")

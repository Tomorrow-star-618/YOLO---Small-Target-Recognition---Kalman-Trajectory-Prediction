import numpy as np
from .enhanced_aircraft_kalman_tracker import AircraftKalmanTracker

class EnhancedMultiTargetTracker:
    """
    增强版多目标卡尔曼滤波跟踪管理器
    
    针对云层遮挡优化的多目标跟踪系统，支持：
    - 长时间目标跟踪（10-15秒丢失容忍）
    - 智能数据关联
    - 轨迹生命周期管理
    - 预测置信度评估
    """
    
    def __init__(self, max_lost_frames=450, min_hits=3, iou_threshold=0.3):
        """
        初始化增强版多目标跟踪器
        
        Args:
            max_lost_frames: 最大允许丢失帧数（默认450帧=15秒@30fps）
            min_hits: 确认新轨迹所需的最小命中次数
            iou_threshold: IoU匹配阈值
        """
        self.trackers = []  # 活跃的跟踪器列表
        self.max_lost_frames = max_lost_frames
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.frame_count = 0
        self.next_track_id = 1
        
        # 跟踪统计
        self.stats = {
            'total_tracks_created': 0,
            'total_tracks_terminated': 0,
            'current_active_tracks': 0,
            'long_term_predictions': 0,
            'successful_recoveries': 0  # 丢失后重新检测到的次数
        }
        
        print(f"增强版多目标跟踪器初始化完成 - 最大丢失容忍: {max_lost_frames}帧 ({max_lost_frames/30:.1f}秒)")
    
    def update(self, detections):
        """
        更新跟踪器状态，增强版本
        
        Args:
            detections: list of [x1, y1, x2, y2, conf] 检测结果
            
        Returns:
            tracks: list of track information dictionaries
        """
        self.frame_count += 1
        
        # Step 1: 所有跟踪器进行预测
        predicted_boxes = []
        for tracker in self.trackers:
            pred_bbox = tracker.predict()
            predicted_boxes.append(pred_bbox)
        
        # Step 2: 数据关联 - 匹配检测与跟踪器
        if len(detections) > 0 and len(self.trackers) > 0:
            matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
                detections, predicted_boxes, self.iou_threshold
            )
        else:
            matched = []
            unmatched_dets = list(range(len(detections)))
            unmatched_trks = list(range(len(self.trackers)))
        
        # Step 3: 更新匹配的跟踪器
        for det_idx, trk_idx in matched:
            tracker = self.trackers[trk_idx]
            was_lost = tracker.is_lost
            tracker.update(detections[det_idx][:4])
            
            # 统计恢复次数
            if was_lost:
                self.stats['successful_recoveries'] += 1
                print(f"跟踪器 {tracker.track_id} 重新检测到，切换回检测模式")
        
        # Step 4: 处理未匹配的跟踪器（标记为丢失）
        for trk_idx in unmatched_trks:
            tracker = self.trackers[trk_idx]
            was_lost = tracker.is_lost
            tracker.mark_as_lost()
            
            # 打印状态转换信息
            if not was_lost:  # 刚从检测转为丢失
                print(f"跟踪器 {tracker.track_id} 丢失检测，切换到预测模式")
        
        # Step 5: 为未匹配的检测创建新跟踪器
        for det_idx in unmatched_dets:
            new_tracker = AircraftKalmanTracker(
                detections[det_idx][:4],
                track_id=f"T{self.next_track_id:03d}",
                max_lost_frames=self.max_lost_frames
            )
            self.trackers.append(new_tracker)
            self.next_track_id += 1
            self.stats['total_tracks_created'] += 1
            print(f"创建新跟踪器: {new_tracker.track_id}")
        
        # Step 6: 移除需要删除的跟踪器
        valid_trackers = []
        for tracker in self.trackers:
            if not tracker.should_delete(self.max_lost_frames):
                valid_trackers.append(tracker)
            else:
                print(f"删除跟踪器 {tracker.track_id} - 丢失时间: {tracker.time_since_update}帧")
                self.stats['total_tracks_terminated'] += 1
        
        self.trackers = valid_trackers
        self.stats['current_active_tracks'] = len(self.trackers)
        
        # Step 7: 返回确认的跟踪结果
        confirmed_tracks = []
        for tracker in self.trackers:
            # 返回所有跟踪器（包括丢失状态的预测）
            # 对于丢失状态的目标，我们也要显示预测，不管hit_streak
            if tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits or tracker.is_lost:
                track_info = tracker.get_track_info()
                confirmed_tracks.append(track_info)
                
                # 统计长期预测
                if track_info['status'] == 'predicted' and track_info['lost_frames'] > 30:
                    self.stats['long_term_predictions'] += 1
        
        # 每100帧打印一次统计信息
        if self.frame_count % 100 == 0:
            self._print_statistics()
        
        return confirmed_tracks
    
    def _associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
        """
        增强版数据关联算法
        
        Args:
            detections: 当前帧检测结果
            trackers: 预测的跟踪器位置
            iou_threshold: IoU匹配阈值
            
        Returns:
            matched: [(det_idx, trk_idx), ...] 匹配对
            unmatched_detections: [det_idx, ...] 未匹配的检测
            unmatched_trackers: [trk_idx, ...] 未匹配的跟踪器
        """
        if len(trackers) == 0:
            return [], list(range(len(detections))), []
        
        # 计算IoU矩阵
        iou_matrix = self._calculate_iou_matrix(detections, trackers)
        
        # 使用匈牙利算法或贪心算法进行匹配
        matched_indices = self._solve_assignment_problem(iou_matrix, iou_threshold)
        
        # 整理匹配结果
        matched = []
        unmatched_detections = []
        unmatched_trackers = []
        
        for det_idx in range(len(detections)):
            if det_idx not in [m[0] for m in matched_indices]:
                unmatched_detections.append(det_idx)
        
        for trk_idx in range(len(trackers)):
            if trk_idx not in [m[1] for m in matched_indices]:
                unmatched_trackers.append(trk_idx)
        
        # 过滤低IoU匹配
        for det_idx, trk_idx in matched_indices:
            if iou_matrix[det_idx, trk_idx] >= iou_threshold:
                matched.append((det_idx, trk_idx))
            else:
                unmatched_detections.append(det_idx)
                unmatched_trackers.append(trk_idx)
        
        return matched, unmatched_detections, unmatched_trackers
    
    def _calculate_iou_matrix(self, detections, trackers):
        """
        计算检测与跟踪器之间的IoU矩阵
        
        Args:
            detections: 检测结果列表
            trackers: 跟踪器预测位置列表
            
        Returns:
            iou_matrix: IoU相似度矩阵
        """
        iou_matrix = np.zeros((len(detections), len(trackers)))
        
        for det_idx, detection in enumerate(detections):
            det_bbox = detection[:4]
            for trk_idx, tracker_bbox in enumerate(trackers):
                iou_matrix[det_idx, trk_idx] = self._calculate_iou(det_bbox, tracker_bbox)
        
        return iou_matrix
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        计算两个边界框的IoU
        
        Args:
            bbox1, bbox2: [x1, y1, x2, y2] 边界框
            
        Returns:
            iou: IoU值 [0, 1]
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
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
    
    def _solve_assignment_problem(self, iou_matrix, threshold):
        """
        解决分配问题（简化版匈牙利算法）
        
        Args:
            iou_matrix: IoU相似度矩阵
            threshold: 最小IoU阈值
            
        Returns:
            matched_indices: [(det_idx, trk_idx), ...] 匹配对
        """
        if iou_matrix.size == 0:
            return []
        
        # 简单贪心策略：选择最大IoU的匹配
        matched_indices = []
        used_dets = set()
        used_trks = set()
        
        # 按IoU值从大到小排序
        det_indices, trk_indices = np.where(iou_matrix >= threshold)
        if len(det_indices) == 0:
            return []
        
        iou_values = iou_matrix[det_indices, trk_indices]
        sorted_indices = np.argsort(-iou_values)  # 降序排列
        
        for idx in sorted_indices:
            det_idx = det_indices[idx]
            trk_idx = trk_indices[idx]
            
            if det_idx not in used_dets and trk_idx not in used_trks:
                matched_indices.append((det_idx, trk_idx))
                used_dets.add(det_idx)
                used_trks.add(trk_idx)
        
        return matched_indices
    
    def _print_statistics(self):
        """打印跟踪统计信息"""
        print(f"\n=== 跟踪统计 (帧 {self.frame_count}) ===")
        print(f"当前活跃轨迹: {self.stats['current_active_tracks']}")
        print(f"总创建轨迹: {self.stats['total_tracks_created']}")
        print(f"总终止轨迹: {self.stats['total_tracks_terminated']}")
        print(f"成功恢复次数: {self.stats['successful_recoveries']}")
        print(f"长期预测次数: {self.stats['long_term_predictions']}")
        
        # 打印每个跟踪器的状态
        for tracker in self.trackers:
            status = "丢失" if tracker.is_lost else "正常"
            confidence = tracker.motion_analysis.get('prediction_confidence', 0.0)
            print(f"  {tracker.track_id}: {status}, 年龄:{tracker.age}, "
                  f"命中:{tracker.hits}, 丢失:{tracker.lost_frames}, 置信度:{confidence:.2f}")
    
    def get_statistics(self):
        """获取详细统计信息"""
        return {
            **self.stats,
            'frame_count': self.frame_count,
            'tracker_details': [
                {
                    'track_id': t.track_id,
                    'age': t.age,
                    'hits': t.hits,
                    'lost_frames': t.lost_frames,
                    'is_lost': t.is_lost,
                    'confidence': t.motion_analysis.get('prediction_confidence', 0.0)
                }
                for t in self.trackers
            ]
        }

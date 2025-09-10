import cv2
import numpy as np
from collections import defaultdict

class TrajectoryVisualizer:
    """
    优化版轨迹可视化器，专注于检测/预测状态的清晰显示
    """
    
    def __init__(self, colors=None):
        """初始化可视化器"""
        self.colors = colors or {
            'detected': (0, 255, 0),      # 绿色 - 正常检测
            'predicted': (0, 165, 255),   # 橙色 - 卡尔曼预测  
            'lost': (0, 100, 255),        # 深橙色 - 长时间丢失
            'trajectory': (255, 255, 0),  # 黄色 - 轨迹线
            'velocity': (255, 0, 255),    # 品红色 - 速度向量
            'text': (255, 255, 255),      # 白色 - 文本
            'background': (0, 0, 0)       # 黑色 - 背景
        }
        
        self.trajectory_length = 20
        self.velocity_scale = 5.0
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.4  # 减小字体大小，避免遮挡小目标
        self.font_thickness = 1  # 减小字体粗细
        self.frame_counter = 0
        
    def draw_tracks(self, image, tracks, detections=None, frame_info=None):
        """在图像上绘制跟踪结果"""
        vis_image = image.copy()
        self.frame_counter += 1
        
        if detections:
            self._draw_detections(vis_image, detections)
        
        for track in tracks:
            self._draw_single_track(vis_image, track)
        
        if frame_info:
            self._draw_frame_info(vis_image, frame_info, tracks, detections)
        
        self._draw_legend(vis_image)
        return vis_image
    
    def _draw_detections(self, image, detections):
        """绘制原始检测结果"""
        for det in detections:
            if len(det) >= 5:
                x1, y1, x2, y2, conf = det[:5]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(image, (x1, y1), (x2, y2), self.colors['detected'], 1)
                cv2.putText(image, f"Det: {conf:.2f}", (x1, y1 - 5), 
                           self.font, 0.3, self.colors['detected'], 1)  # 更小字体
    
    def _draw_single_track(self, image, track):
        """绘制单个跟踪目标，强化状态区分"""
        bbox = track['bbox']
        track_id = str(track['track_id'])
        status = track.get('status', 'detected')
        time_since_update = int(track.get('time_since_update', 0))
        confidence = float(track.get('confidence', 1.0))
        trajectory = track.get('trajectory', [])
        velocity = track.get('velocity', (0, 0))
        
        x1, y1, x2, y2 = [int(float(coord)) for coord in bbox[:4]]
        
        if status == 'predicted':
            # 预测状态：醒目的橙色框（细线条）
            if (self.frame_counter // 6) % 2 == 0:
                flash_color = (0, 220, 255)  # 亮橙色
                thickness = 2  # 减少线条粗细
            else:
                flash_color = self.colors['predicted']
                thickness = 1  # 减少线条粗细
            
            cv2.rectangle(image, (x1, y1), (x2, y2), flash_color, thickness)
            
            # 半透明填充
            overlay = image.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), flash_color, -1)
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
            
            # ID标签
            label = f"ID:{track_id} PRED({time_since_update})"
            self._draw_label(image, label, x1, y1, x2, y2, flash_color)
            
            # "AI PREDICTION"提示
            pred_text = "⚠️ AI PREDICTION"
            self._draw_status_text(image, pred_text, x2, y1, flash_color)
            
        else:
            # 检测状态：绿色框（细线条）
            color = self.colors['detected']
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)  # 减少线条粗细
            
            # ID标签
            label = f"ID:{track_id} TRACKING"
            self._draw_label(image, label, x1, y1, x2, y2, color)
            
            # "DETECTED"提示
            det_text = "✅ DETECTED"
            self._draw_status_text(image, det_text, x2, y1, color)
        
        # 置信度 - 调整字体大小和位置
        conf_text = f"Conf: {confidence:.2f}"
        cv2.putText(image, conf_text, (x2 + 10, y2 + 10), 
                   self.font, 0.3, self.colors['text'], 1)  # 更小字体，移到右下角
        
        # 轨迹和速度
        color = self.colors['predicted'] if status == 'predicted' else self.colors['detected']
        self._draw_trajectory(image, trajectory, color)
        
        vx, vy = velocity
        speed = np.sqrt(vx**2 + vy**2)
        if speed > 1.0:
            self._draw_velocity_vector(image, bbox, velocity)
    
    def _draw_label(self, image, label, x1, y1, x2, y2, color):
        """绘制ID标签 - 移到方框右上角一定距离"""
        label_size = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)[0]
        
        # 标签位置：方框右上角偏移15像素
        label_x = x2 + 15  # 在方框右侧15像素处
        label_y = y1 - 5   # 在方框上方5像素处
        
        # 背景框
        bg_x1 = label_x - 2
        bg_y1 = label_y - label_size[1] - 2
        bg_x2 = label_x + label_size[0] + 2
        bg_y2 = label_y + 2
        
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.putText(image, label, (label_x, label_y), 
                   self.font, self.font_scale, self.colors['text'], self.font_thickness)
    
    def _draw_status_text(self, image, text, x2, y1, color):
        """绘制状态提示文字 - 调整字体大小和位置"""
        # 使用更小的字体
        status_font_scale = 0.35
        status_font_thickness = 1
        
        text_size = cv2.getTextSize(text, self.font, status_font_scale, status_font_thickness)[0]
        text_x = x2 + 20  # 增加距离，避免与标签重叠
        text_y = y1 + 15  # 稍微下移
        
        # 确保文字在图像范围内
        h, w = image.shape[:2]
        if text_x + text_size[0] > w:
            text_x = x2 - text_size[0] - 20
        if text_y > h:
            text_y = y1 - 10
        
        # 背景
        cv2.rectangle(image, (text_x - 2, text_y - text_size[1] - 2), 
                     (text_x + text_size[0] + 2, text_y + 2), color, -1)
        cv2.putText(image, text, (text_x, text_y), 
                   self.font, status_font_scale, (255, 255, 255), status_font_thickness)
    
    def _draw_trajectory(self, image, trajectory, color):
        """绘制轨迹"""
        if len(trajectory) < 2:
            return
        
        recent_trajectory = trajectory[-self.trajectory_length:]
        points = np.array(recent_trajectory, dtype=np.int32)
        
        for i in range(1, len(points)):
            alpha = i / len(points)
            thickness = max(1, int(3 * alpha))
            cv2.line(image, tuple(points[i-1]), tuple(points[i]), 
                    self.colors['trajectory'], thickness)
    
    def _draw_velocity_vector(self, image, bbox, velocity):
        """绘制速度向量"""
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)
        
        vx, vy = velocity
        end_x = int(center_x + vx * self.velocity_scale)
        end_y = int(center_y + vy * self.velocity_scale)
        
        cv2.arrowedLine(image, (center_x, center_y), (end_x, end_y), 
                       self.colors['velocity'], 2, tipLength=0.3)
    
    def _draw_frame_info(self, image, frame_info, tracks, detections):
        """绘制帧信息"""
        info_y = 30
        line_height = 25
        
        # 状态统计
        detected_count = sum(1 for t in tracks if t.get('status') == 'detected')
        predicted_count = sum(1 for t in tracks if t.get('status') == 'predicted')
        
        info_texts = [
            f"Frame: {frame_info.get('frame_number', 0)}",
            f"Detections: {len(detections) if detections else 0}",
            f"Tracking (Green): {detected_count}",
            f"Predicting (Orange): {predicted_count}",
        ]
        
        if 'state_changes' in frame_info:
            info_texts.append(f"State Changes: {frame_info['state_changes']}")
        
        for i, text in enumerate(info_texts):
            y_pos = info_y + i * line_height
            cv2.putText(image, text, (10, y_pos), 
                       self.font, 0.6, self.colors['text'], 2)
    
    def _draw_legend(self, image):
        """绘制图例"""
        h, w = image.shape[:2]
        legend_x = w - 220
        legend_y = h - 100
        
        cv2.rectangle(image, (legend_x - 10, legend_y - 10), 
                     (w - 10, h - 10), self.colors['background'], -1)
        cv2.rectangle(image, (legend_x - 10, legend_y - 10), 
                     (w - 10, h - 10), self.colors['text'], 2)
        
        cv2.putText(image, "Status Legend", (legend_x, legend_y - 5), 
                   self.font, 0.6, self.colors['text'], 2)
        
        legends = [
            ("Green = Detection", self.colors['detected']),
            ("Orange = Prediction", self.colors['predicted']),
            ("Yellow = Trail", self.colors['trajectory'])
        ]
        
        for i, (label, color) in enumerate(legends):
            y = legend_y + 15 + i * 20
            cv2.rectangle(image, (legend_x, y), (legend_x + 15, y + 15), color, -1)
            cv2.putText(image, label, (legend_x + 25, y + 12), 
                       self.font, 0.45, self.colors['text'], 1)

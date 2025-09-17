#!/usr/bin/env python3
"""
红外小目标飞机检测与卡尔曼跟踪系统

完整的检测+跟踪解决方案，包含：
1. YOLOv8小目标检测模型
2. 增强版卡尔曼滤波跟踪器  
3. 长期预测(5秒)云层遮挡容忍
4. 优化的可视化效果(细线条+小字体)

功能特点：
- 绿色细框: 正常检测状态
- 橙色闪烁框: 卡尔曼预测状态
- 右上角小字体: 避免遮挡小目标
- 150帧(5秒)预测容忍: 适应云层遮挡
"""

import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os

# 添加kalman模块路径
sys.path.append('/home/mingxing/worksapce/ultralytics')
from kalman.enhanced_multi_target_tracker import EnhancedMultiTargetTracker
from kalman.trajectory_visualizer import TrajectoryVisualizer

def aircraft_detection_tracking():
    """红外小目标飞机检测与跟踪系统主程序"""
    
    # 输入视频路径
    video_path = "/home/mingxing/worksapce/ultralytics/vedio/short.mp4"
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        return
    
    # 输出路径 - 最终版本
    output_path = "/home/mingxing/worksapce/ultralytics/tracking_results/aircraft_detection_tracking_result.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 初始化小目标检测模型
    model_path = "/home/mingxing/worksapce/ultralytics/small_target_detection/yolov8_small_aircraft/weights/best.pt"
    print(f"🚀 初始化小目标检测YOLO模型: {model_path}")
    model = YOLO(model_path)
    
    print("🎯 初始化增强版多目标跟踪器（小目标优化）...")
    tracker = EnhancedMultiTargetTracker(
        max_lost_frames=150,  # 增加到5秒，适应小目标长时间遮挡
        min_hits=1,           # 立即开始跟踪
        iou_threshold=0.1     # 降低IoU阈值适应小目标
    )
    
    print("🎨 初始化增强版可视化器...")
    visualizer = TrajectoryVisualizer()
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return
    
    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📺 视频信息: {width}x{height}, {fps}fps, {total_frames}帧")
    
    # 设置视频写入
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    detection_frames = 0
    prediction_frames = 0
    state_changes = 0
    last_states = {}
    
    print("🎬 开始处理视频...")
    print("🔍 关注效果: 绿色检测框 ↔ 橙色预测框 (小目标优化)")
    print(f"📹 视频源: {video_path}")
    print(f"🤖 模型: 小目标检测专用模型")
    print(f"📊 检测阈值: 0.1 (降低以适应小目标)")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # YOLO检测
            results = model(frame, verbose=False)
            
            # 提取检测结果
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()
                
                for box, score in zip(boxes, scores):
                    if score > 0.1:  # 降低置信度阈值以适应小目标检测
                        detections.append([box[0], box[1], box[2], box[3], score])
            
            # 多目标跟踪
            tracks = tracker.update(detections)
            
            # 统计状态切换
            current_states = {}
            for track in tracks:
                track_id = track['track_id']
                current_status = track['status']
                current_states[track_id] = current_status
                
                # 检测状态切换
                if track_id in last_states:
                    if last_states[track_id] != current_status:
                        state_changes += 1
                        change_msg = f"🔄 帧{frame_count}: 目标{track_id} {last_states[track_id]} → {current_status}"
                        print(change_msg)
                
                # 统计状态帧数
                if current_status == 'detected':
                    detection_frames += 1
                elif current_status == 'predicted':
                    prediction_frames += 1
            
            last_states = current_states.copy()
            
            # 使用增强版可视化器绘制
            frame_info = {
                'frame_number': frame_count,
                'detections': len(detections),
                'tracks': len(tracks),
                'detection_frames': detection_frames,
                'prediction_frames': prediction_frames,
                'state_changes': state_changes
            }
            
            vis_frame = visualizer.draw_tracks(frame, tracks, detections, frame_info)
            
            # 添加大标题显示当前状态
            title_y = 30
            if any(t['status'] == 'predicted' for t in tracks):
                title = "🟠 AI PREDICTION MODE - Orange Boxes"
                title_color = (0, 165, 255)  # 橙色
            elif any(t['status'] == 'detected' for t in tracks):
                title = "🟢 DETECTION MODE - Green Boxes"
                title_color = (0, 255, 0)  # 绿色
            else:
                title = "⚪ NO TARGETS"
                title_color = (255, 255, 255)  # 白色
            
            cv2.putText(vis_frame, title, (10, title_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, title_color, 3)
            
            # 写入输出视频
            out.write(vis_frame)
            
            # 每50帧显示进度
            if frame_count % 50 == 0:
                progress = frame_count / total_frames * 100
                print(f"⏳ 处理进度: {progress:.1f}% ({frame_count}/{total_frames})")
                print(f"   📊 检测帧: {detection_frames}, 预测帧: {prediction_frames}")
                print(f"   🔄 状态切换: {state_changes}次")
    
    except Exception as e:
        print(f"❌ 处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    # 最终统计
    print("\n" + "="*60)
    print("🎉 小目标检测交替显示测试结果")
    print("="*60)
    print(f"📈 总帧数: {frame_count}")
    print(f"🟢 检测状态帧数: {detection_frames}")
    print(f"🟠 预测状态帧数: {prediction_frames}")
    print(f"🔄 状态切换次数: {state_changes}")
    print(f"🤖 使用模型: 小目标检测专用模型")
    
    if detection_frames + prediction_frames > 0:
        pred_ratio = prediction_frames / (detection_frames + prediction_frames) * 100
        print(f"📊 预测帧占比: {pred_ratio:.1f}%")
    
    print(f"🎬 输出视频: {output_path}")
    
    # 效果评估
    print("\n" + "-"*60)
    print("📝 效果评估:")
    
    if state_changes >= 3:
        print("✅ 状态切换次数充足，用户能看到绿橙交替！")
    else:
        print("⚠️  状态切换较少，可能目标被遮挡时间不够长")
    
    if prediction_frames > detection_frames * 0.5:
        print("✅ 预测帧数量充足，橙色预测框非常明显！")
    elif prediction_frames > 0:
        print("✅ 有预测帧出现，橙色预测框应该可见")
    else:
        print("⚠️  没有预测帧，可能检测过于稳定")
    
    if detection_frames > 0:
        print("✅ 有检测帧，绿色检测框正常显示")
    
    print("\n🎯 核心验证点:")
    print("1. 绿色框 = 检测状态，应显示 '✅ DETECTED'")
    print("2. 橙色框 = 预测状态，应显示 '⚠️ AI PREDICTION'")
    print("3. 状态切换时颜色立即改变")
    print("4. 橙色框应该有闪烁和半透明填充效果")

if __name__ == "__main__":
    aircraft_detection_tracking()

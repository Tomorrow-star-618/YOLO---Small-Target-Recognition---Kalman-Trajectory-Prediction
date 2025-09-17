"""
相机运动补偿跟踪系统测试程序
完整的测试和性能评估
"""

import cv2
import numpy as np
import os
import sys
import time
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ultralytics import YOLO
from camera_motion_compensation.motion_compensated_multi_tracker import MotionCompensatedMultiTracker
from kalman.trajectory_visualizer import TrajectoryVisualizer

class CameraMotionCompensationTestSystem:
    """
    相机运动补偿测试系统
    
    测试功能：
    1. 对比原始跟踪vs运动补偿跟踪
    2. 性能评估和统计分析  
    3. 可视化效果对比
    4. 详细的测试报告生成
    """
    
    def __init__(self, model_path, conf_threshold=0.1):
        """初始化测试系统"""
        print("🚀 初始化相机运动补偿测试系统...")
        
        # YOLO模型
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # 可视化器
        self.visualizer = TrajectoryVisualizer()
        
        # 测试配置
        self.test_methods = [
            'optical_flow',
            'feature_matching', 
            'hybrid'
        ]
        
        # 测试结果
        self.test_results = {}
        
        print("✅ 系统初始化完成")
    
    def run_single_test(self, video_path, output_dir, method='optical_flow', 
                       test_name="default"):
        """运行单个测试"""
        print(f"\n{'='*60}")
        print(f"🧪 开始测试: {test_name} - {method}")
        print(f"{'='*60}")
        
        # 创建输出目录
        method_output_dir = os.path.join(output_dir, method)
        os.makedirs(method_output_dir, exist_ok=True)
        
        # 初始化跟踪器
        tracker = MotionCompensatedMultiTracker(
            max_lost_frames=150,
            min_hits=1,
            iou_threshold=0.1,
            motion_detection_method=method
        )
        
        # 视频处理
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频: {video_path}")
            return None
        
        # 视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📺 视频信息: {width}x{height}, {fps}fps, {total_frames}帧")
        
        # 输出视频
        output_video_path = os.path.join(method_output_dir, f"{test_name}_result.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # 处理统计
        frame_count = 0
        detection_count = 0
        tracking_stats = []
        processing_times = []
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            
            # YOLO检测
            results = self.model(frame, verbose=False, conf=self.conf_threshold)
            
            # 提取检测结果
            detections = []
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()
                
                for box, score in zip(boxes, scores):
                    if score > self.conf_threshold:
                        detections.append([
                            float(box[0]), float(box[1]), 
                            float(box[2]), float(box[3]), 
                            float(score)
                        ])
            
            detection_count += len(detections)
            
            # 跟踪更新
            tracks = tracker.update(detections, frame)
            
            # 可视化
            frame_info = {
                'frame_number': frame_count,
                'detections': len(detections),
                'tracks': len(tracks),
                'method': method
            }
            
            vis_frame = self.visualizer.draw_tracks(frame, tracks, detections, frame_info)
            
            # 添加测试信息
            self._add_test_info_to_frame(vis_frame, method, frame_count, tracker)
            
            # 写入视频
            out.write(vis_frame)
            
            # 统计信息
            frame_time = (time.time() - frame_start) * 1000
            processing_times.append(frame_time)
            
            tracking_stats.append({
                'frame': frame_count,
                'detections': len(detections),
                'tracks': len(tracks),
                'processing_time': frame_time
            })
            
            frame_count += 1
            
            # 进度显示
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"⏳ 处理进度: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # 清理资源
        cap.release()
        out.release()
        
        total_time = time.time() - start_time
        
        # 获取最终统计
        final_stats = tracker.get_comprehensive_stats()
        
        # 测试结果
        test_result = {
            'method': method,
            'test_name': test_name,
            'video_path': video_path,
            'output_video': output_video_path,
            'processing_stats': {
                'total_frames': frame_count,
                'total_detections': detection_count,
                'total_time': total_time,
                'avg_fps': frame_count / total_time,
                'avg_processing_time': np.mean(processing_times),
                'avg_detections_per_frame': detection_count / frame_count if frame_count > 0 else 0
            },
            'tracking_stats': final_stats,
            'frame_by_frame_stats': tracking_stats
        }
        
        print(f"✅ 测试完成: {test_name} - {method}")
        print(f"📊 处理帧数: {frame_count}")
        print(f"⚡ 平均FPS: {test_result['processing_stats']['avg_fps']:.1f}")
        print(f"🎯 总检测数: {detection_count}")
        print(f"🔄 全局重置: {final_stats['basic']['global_resets']}次")
        print(f"📈 个体重置: {final_stats['basic']['individual_resets']}次")
        
        return test_result
    
    def run_comprehensive_test(self, video_path, output_dir, test_name="comprehensive"):
        """运行综合测试（所有方法对比）"""
        print(f"\n{'='*80}")
        print(f"🚀 启动综合测试: {test_name}")
        print(f"{'='*80}")
        
        all_results = {}
        
        # 测试所有方法
        for method in self.test_methods:
            try:
                result = self.run_single_test(
                    video_path, output_dir, method, 
                    f"{test_name}_{method}"
                )
                if result:
                    all_results[method] = result
            except Exception as e:
                print(f"❌ 方法 {method} 测试失败: {e}")
                continue
        
        # 生成对比报告
        if all_results:
            self._generate_comparison_report(all_results, output_dir, test_name)
        
        self.test_results[test_name] = all_results
        
        print(f"✅ 综合测试完成: {test_name}")
        return all_results
    
    def _add_test_info_to_frame(self, frame, method, frame_count, tracker):
        """在帧上添加测试信息"""
        # 方法名称
        cv2.putText(frame, f"Method: {method.upper()}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 帧编号
        cv2.putText(frame, f"Frame: {frame_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 运动检测状态
        if hasattr(tracker, 'frame_motion_info') and tracker.frame_motion_info:
            motion_info = tracker.frame_motion_info
            if motion_info['is_motion']:
                status_text = f"Motion: {motion_info['magnitude']:.1f}px"
                color = (0, 165, 255) if motion_info['should_reset'] else (0, 255, 255)
                cv2.putText(frame, status_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 跟踪器统计
        active_trackers = len(tracker.trackers)
        cv2.putText(frame, f"Trackers: {active_trackers}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _generate_comparison_report(self, results, output_dir, test_name):
        """生成对比报告"""
        report_path = os.path.join(output_dir, f"{test_name}_comparison_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"相机运动补偿跟踪系统 - 对比测试报告\n")
            f.write(f"=" * 60 + "\n\n")
            f.write(f"测试名称: {test_name}\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试方法: {', '.join(results.keys())}\n\n")
            
            # 性能对比
            f.write("📊 性能对比\n")
            f.write("-" * 40 + "\n")
            for method, result in results.items():
                stats = result['processing_stats']
                f.write(f"{method.upper()}:\n")
                f.write(f"  平均FPS: {stats['avg_fps']:.1f}\n")
                f.write(f"  处理时间: {stats['avg_processing_time']:.2f}ms\n")
                f.write(f"  总帧数: {stats['total_frames']}\n")
                f.write(f"  总检测: {stats['total_detections']}\n\n")
            
            # 运动检测对比
            f.write("🎯 运动检测对比\n")
            f.write("-" * 40 + "\n")
            for method, result in results.items():
                tracking_stats = result['tracking_stats']
                f.write(f"{method.upper()}:\n")
                f.write(f"  全局重置: {tracking_stats['basic']['global_resets']}次\n")
                f.write(f"  个体重置: {tracking_stats['basic']['individual_resets']}次\n")
                f.write(f"  运动检测率: {tracking_stats['motion_detection']['motion_detection_rate']}\n")
                f.write(f"  重置触发率: {tracking_stats['motion_detection']['reset_trigger_rate']}\n\n")
            
            # 推荐方法
            f.write("💡 推荐方案\n")
            f.write("-" * 40 + "\n")
            
            # 基于性能和效果选择最佳方法
            best_method = self._select_best_method(results)
            f.write(f"推荐方法: {best_method.upper()}\n")
            f.write(f"推荐理由: 在性能和效果间达到最佳平衡\n\n")
        
        print(f"📋 对比报告已生成: {report_path}")
    
    def _select_best_method(self, results):
        """选择最佳方法"""
        scores = {}
        
        for method, result in results.items():
            # 综合评分：性能 + 效果
            fps_score = result['processing_stats']['avg_fps'] / 30.0  # 标准化FPS
            reset_effectiveness = (result['tracking_stats']['basic']['global_resets'] + 
                                 result['tracking_stats']['basic']['individual_resets'])
            
            # 权重评分
            scores[method] = fps_score * 0.3 + min(reset_effectiveness / 10.0, 1.0) * 0.7
        
        return max(scores, key=scores.get)


def main():
    """主测试函数"""
    print("🚀 相机运动补偿跟踪系统 - 测试程序")
    print("=" * 60)
    
    # 配置参数
    model_path = "/home/mingxing/worksapce/ultralytics/small_target_detection/yolov8_small_aircraft/weights/best.pt"
    video_path = "/home/mingxing/worksapce/ultralytics/vedio/vedio/short.mp4"
    output_dir = "/home/mingxing/worksapce/ultralytics/camera_motion_compensation/test_results"
    
    # 检查文件
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化测试系统
    test_system = CameraMotionCompensationTestSystem(model_path, conf_threshold=0.1)
    
    print(f"📁 输出目录: {output_dir}")
    print(f"🎯 使用模型: {model_path}")
    print(f"📺 测试视频: {video_path}")
    
    # 运行综合测试
    results = test_system.run_comprehensive_test(
        video_path, output_dir, "motion_compensation_v1"
    )
    
    if results:
        print(f"\n{'='*60}")
        print("✅ 所有测试完成！")
        print(f"📊 测试了 {len(results)} 种方法")
        print(f"📁 结果保存在: {output_dir}")
        print("=" * 60)
    else:
        print("❌ 测试失败！")


if __name__ == "__main__":
    main()

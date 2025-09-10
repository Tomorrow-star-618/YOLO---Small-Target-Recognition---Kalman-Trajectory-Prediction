"""
卡尔曼滤波跟踪模块

该模块提供了专门针对飞机目标跟踪的卡尔曼滤波器实现，
包括单目标跟踪、多目标管理和轨迹可视化功能。

主要组件：
- AircraftKalmanTracker: 飞机目标卡尔曼滤波跟踪器（标准版）
- EnhancedAircraftKalmanTracker: 增强版卡尔曼滤波跟踪器（支持长期预测）
- MultiTargetKalmanTracker: 多目标跟踪管理器
- EnhancedMultiTargetTracker: 增强版多目标跟踪管理器（云层遮挡优化）
- TrajectoryVisualizer: 轨迹可视化器

使用示例：
    # 标准版本
    from kalman.aircraft_kalman_tracker import AircraftKalmanTracker
    from kalman.multi_target_tracker import MultiTargetKalmanTracker
    
    # 增强版本（推荐）
    from kalman.enhanced_aircraft_kalman_tracker import AircraftKalmanTracker as EnhancedTracker
    from kalman.enhanced_multi_target_tracker import EnhancedMultiTargetTracker
    
    # 可视化
    from kalman.trajectory_visualizer import TrajectoryVisualizer
"""

# 增强版本（优化云层遮挡处理）
from .enhanced_aircraft_kalman_tracker import AircraftKalmanTracker as EnhancedAircraftKalmanTracker  
from .enhanced_multi_target_tracker import EnhancedMultiTargetTracker

# 为了兼容性，提供别名
AircraftKalmanTracker = EnhancedAircraftKalmanTracker
MultiTargetTracker = EnhancedMultiTargetTracker

__all__ = [
    # 标准版本
    'AircraftKalmanTracker',
    'MultiTargetKalmanTracker', 
    'TrajectoryVisualizer',
    # 增强版本
    'EnhancedAircraftKalmanTracker',
    'EnhancedMultiTargetTracker'
]

__version__ = '2.0.0'
__author__ = 'Aircraft Tracking Team'

"""
相机运动补偿系统 - 简化测试脚本
不依赖复杂导入的基础功能测试.
"""

import os
import sys


def check_system_requirements():
    """检查系统要求."""
    print("🔍 检查系统要求...")

    required_packages = ["cv2", "numpy", "ultralytics"]
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - 未安装")

    if missing_packages:
        print("\n📦 需要安装以下包:")
        for package in missing_packages:
            print(f"pip install {package}")
        return False

    print("✅ 所有依赖包已满足")
    return True


def check_file_structure():
    """检查文件结构."""
    print("\n📁 检查文件结构...")

    base_dir = "/home/mingxing/worksapce/ultralytics"
    required_files = [
        "camera_motion_compensation/global_motion_detector.py",
        "camera_motion_compensation/motion_reset_kalman_tracker.py",
        "camera_motion_compensation/motion_compensated_multi_tracker.py",
        "camera_motion_compensation/test_motion_compensation.py",
        "small_target_detection/yolov8_small_aircraft/weights/best.pt",
        "video/short.mp4",
    ]

    all_exist = True
    for file_path in required_files:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - 不存在")
            all_exist = False

    if all_exist:
        print("✅ 所有必需文件存在")
    else:
        print("❌ 部分文件缺失，请检查文件结构")

    return all_exist


def test_import_modules():
    """测试模块导入."""
    print("\n🔗 测试模块导入...")

    # 添加项目路径
    project_root = "/home/mingxing/worksapce/ultralytics"
    if project_root not in sys.path:
        sys.path.append(project_root)

    test_modules = [
        "camera_motion_compensation.global_motion_detector",
        "camera_motion_compensation.motion_reset_kalman_tracker",
        "camera_motion_compensation.motion_compensated_multi_tracker",
    ]

    success_count = 0
    for module_name in test_modules:
        try:
            __import__(module_name, fromlist=[""])
            print(f"✅ {module_name} - 导入成功")
            success_count += 1
        except Exception as e:
            print(f"❌ {module_name} - 导入失败: {e}")

    print(f"\n📊 导入结果: {success_count}/{len(test_modules)} 成功")
    return success_count == len(test_modules)


def run_basic_functionality_test():
    """运行基础功能测试."""
    print("\n⚡ 运行基础功能测试...")

    try:
        # 测试全局运动检测器
        print("🎯 测试全局运动检测器...")
        from camera_motion_compensation.global_motion_detector import GlobalMotionDetector

        detector = GlobalMotionDetector(method="optical_flow")
        print("✅ GlobalMotionDetector 初始化成功")

        # 测试统计功能
        stats = detector.get_stats()
        print(f"✅ 统计功能正常: {stats}")

    except Exception as e:
        print(f"❌ GlobalMotionDetector 测试失败: {e}")
        return False

    try:
        # 测试运动重置跟踪器
        print("🎯 测试运动重置跟踪器...")
        from camera_motion_compensation.motion_reset_kalman_tracker import MotionResetKalmanTracker

        # 创建测试用边界框
        test_bbox = [100, 100, 200, 200]
        tracker = MotionResetKalmanTracker(test_bbox, track_id="TEST001")
        print("✅ MotionResetKalmanTracker 初始化成功")

        # 测试基本功能
        track_info = tracker.get_track_info()
        print(f"✅ 跟踪信息获取正常: ID={track_info['track_id']}")

    except Exception as e:
        print(f"❌ MotionResetKalmanTracker 测试失败: {e}")
        return False

    try:
        # 测试多目标跟踪器
        print("🎯 测试多目标跟踪器...")
        from camera_motion_compensation.motion_compensated_multi_tracker import MotionCompensatedMultiTracker

        multi_tracker = MotionCompensatedMultiTracker(max_lost_frames=150, motion_detection_method="optical_flow")
        print("✅ MotionCompensatedMultiTracker 初始化成功")

        # 测试统计功能
        stats = multi_tracker.get_comprehensive_stats()
        print("✅ 综合统计功能正常")

    except Exception as e:
        print(f"❌ MotionCompensatedMultiTracker 测试失败: {e}")
        return False

    print("✅ 所有基础功能测试通过")
    return True


def create_test_output_structure():
    """创建测试输出目录结构."""
    print("\n📁 创建测试输出目录...")

    base_output_dir = "/home/mingxing/worksapce/ultralytics/camera_motion_compensation/test_results"

    directories = [
        base_output_dir,
        os.path.join(base_output_dir, "optical_flow"),
        os.path.join(base_output_dir, "feature_matching"),
        os.path.join(base_output_dir, "hybrid"),
        os.path.join(base_output_dir, "reports"),
    ]

    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ {directory}")
        except Exception as e:
            print(f"❌ {directory} - 创建失败: {e}")
            return False

    print("✅ 测试输出目录结构创建完成")
    return True


def generate_system_report():
    """生成系统状态报告."""
    print("\n📋 生成系统状态报告...")

    report_path = "/home/mingxing/worksapce/ultralytics/camera_motion_compensation/system_status_report.txt"

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("相机运动补偿系统 - 状态报告\n")
            f.write("=" * 50 + "\n\n")

            # 系统信息
            f.write("📋 系统信息\n")
            f.write(f"Python版本: {sys.version}\n")
            f.write(f"工作目录: {os.getcwd()}\n")
            f.write("项目根目录: /home/mingxing/worksapce/ultralytics\n\n")

            # 模块状态
            f.write("🔗 模块状态\n")
            f.write("全局运动检测器: 已实现\n")
            f.write("运动重置跟踪器: 已实现\n")
            f.write("多目标跟踪器: 已实现\n")
            f.write("测试框架: 已实现\n\n")

            # 功能特性
            f.write("🎯 功能特性\n")
            f.write("- 三种运动检测方法 (光流/特征匹配/混合)\n")
            f.write("- 智能卡尔曼重置机制\n")
            f.write("- 个体和全局运动补偿\n")
            f.write("- 详细性能统计\n")
            f.write("- 自适应参数调整\n\n")

            # 下一步
            f.write("📋 下一步操作\n")
            f.write("1. 运行完整测试: python test_motion_compensation.py\n")
            f.write("2. 检查测试结果视频\n")
            f.write("3. 分析性能统计报告\n")
            f.write("4. 根据结果调优参数\n\n")

        print(f"✅ 系统状态报告已生成: {report_path}")
        return True

    except Exception as e:
        print(f"❌ 报告生成失败: {e}")
        return False


def main():
    """主函数."""
    print("🚀 相机运动补偿系统 - 系统检查")
    print("=" * 60)

    # 执行所有检查
    checks = [
        ("系统要求检查", check_system_requirements),
        ("文件结构检查", check_file_structure),
        ("模块导入测试", test_import_modules),
        ("基础功能测试", run_basic_functionality_test),
        ("输出目录创建", create_test_output_structure),
        ("系统报告生成", generate_system_report),
    ]

    results = []
    for check_name, check_func in checks:
        print(f"\n{'=' * 20} {check_name} {'=' * 20}")
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} 执行失败: {e}")
            results.append((check_name, False))

    # 总结
    print(f"\n{'=' * 60}")
    print("📊 系统检查总结")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for check_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{check_name}: {status}")

    print(f"\n总体结果: {passed}/{total} 项检查通过")

    if passed == total:
        print("🎉 系统准备就绪！可以运行完整测试")
        print("运行命令: python camera_motion_compensation/test_motion_compensation.py")
    else:
        print("⚠️ 系统未完全准备就绪，请解决上述问题")

    print("=" * 60)


if __name__ == "__main__":
    main()

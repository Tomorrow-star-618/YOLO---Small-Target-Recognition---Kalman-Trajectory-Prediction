#!/usr/bin/env python3

"""灰度追踪系统快速启动脚本."""

import os
from pathlib import Path


def quick_start():
    """快速启动."""
    print("🚀 灰度追踪系统 - 快速启动")
    print("=" * 40)

    script_dir = Path(__file__).parent
    video_dir = script_dir.parent / "video"

    # 列出可用视频
    video_files = []
    for ext in ["*.mp4", "*.avi"]:
        video_files.extend(video_dir.glob(ext))

    if not video_files:
        print("❌ 未找到视频文件")
        return

    print("📹 选择视频文件:")
    for i, video in enumerate(video_files, 1):
        size_mb = video.stat().st_size / (1024 * 1024)
        print(f"   {i}. {video.name} ({size_mb:.1f} MB)")

    # 用户选择
    try:
        choice = int(input(f"\n请选择视频 (1-{len(video_files)}): "))
        if not (1 <= choice <= len(video_files)):
            print("❌ 选择无效")
            return

        selected_video = video_files[choice - 1]
        print(f"✅ 选中: {selected_video.name}")

        # 询问是否使用灰度模板
        use_template = input("\n是否使用灰度模板? (y/n): ").strip().lower() == "y"

        # 构建命令
        cmd_parts = ["python grayscale_tracking_system.py", f"-v ../video/{selected_video.name}"]

        if use_template:
            template_file = script_dir / "sample_template.npy"
            if template_file.exists():
                # 加载并格式化模板
                try:
                    import numpy as np

                    template = np.load(template_file)
                    template_str = str(template.tolist())
                    cmd_parts.append(f"-t '{template_str}'")
                    print("✅ 已加载示例灰度模板")
                except ImportError:
                    print("⚠️ NumPy未安装，跳过模板")
            else:
                print("⚠️ 示例模板文件不存在")

        # 输出文件名
        output_name = f"tracked_{selected_video.stem}.mp4"
        cmd_parts.append(f"-o output-vedio/{output_name}")

        # 完整命令
        full_cmd = " ".join(cmd_parts)

        print("\n🎬 执行命令:")
        print(f"   {full_cmd}")

        # 确认执行
        if input("\n开始处理? (y/n): ").strip().lower() == "y":
            print("\n⏳ 开始处理，请等待...")
            print("💡 处理过程中可按 Ctrl+C 中断")

            os.chdir(script_dir)
            exit_code = os.system(full_cmd)

            if exit_code == 0:
                print("\n🎉 处理完成!")
                output_file = script_dir / "output-vedio" / output_name
                if output_file.exists():
                    print(f"📁 输出文件: {output_file}")
                else:
                    print(f"📁 输出目录: {script_dir / 'output-vedio'}")
            else:
                print(f"\n❌ 处理失败 (退出代码: {exit_code})")
        else:
            print("\n👋 已取消")

    except KeyboardInterrupt:
        print("\n👋 用户取消")
    except ValueError:
        print("❌ 输入无效")
    except Exception as e:
        print(f"❌ 错误: {e}")


if __name__ == "__main__":
    quick_start()

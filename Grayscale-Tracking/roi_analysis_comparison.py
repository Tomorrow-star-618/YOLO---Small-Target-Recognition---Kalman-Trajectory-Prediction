#!/usr/bin/env python3

"""ROI多方法对比分析器 从30x30红外灰度矩阵中提取5x5感兴趣区域，使用多种方法找到最佳中心点."""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class ROIMultiMethodAnalyzer:
    """多方法ROI分析器."""

    def __init__(self):
        self.results = []
        self.methods = {
            "max_value": self.method_max_value,
            "mean_intensity": self.method_mean_intensity,
            "weighted_centroid": self.method_weighted_centroid,
            "gradient_magnitude": self.method_gradient_magnitude,
            "contrast_enhancement": self.method_contrast_enhancement,
            "local_variance": self.method_local_variance,
            "temperature_cluster": self.method_temperature_cluster,
            "edge_density": self.method_edge_density,
        }

    def analyze_image(self, image_path, roi_size=5):
        """分析单张图像的ROI."""
        print(f"\n{'=' * 80}")
        print(f"🔍 分析图像: {Path(image_path).name}")
        print(f"{'=' * 80}")

        # 读取并转换为灰度图像
        color_image = cv2.imread(str(image_path))
        if color_image is None:
            print("❌ 无法读取图像")
            return None

        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        print(f"📐 图像尺寸: {gray_image.shape}")

        image_result = {
            "image_path": str(image_path),
            "image_name": Path(image_path).name,
            "image_shape": gray_image.shape,
            "roi_size": roi_size,
            "methods": {},
        }

        # 对每种方法进行分析
        for method_name, method_func in self.methods.items():
            print(f"\n🔬 方法: {method_name}")
            try:
                result = method_func(gray_image, roi_size)
                image_result["methods"][method_name] = result
                print(f"   ✅ 中心点: ({result['center_x']}, {result['center_y']})")
                print(f"   📊 置信度: {result['confidence']:.4f}")
                print(f"   🎯 ROI均值: {result['roi_mean']:.2f}")
            except Exception as e:
                print(f"   ❌ 分析失败: {e}")
                image_result["methods"][method_name] = {"error": str(e)}

        self.results.append(image_result)
        return image_result

    def method_max_value(self, gray_image, roi_size):
        """方法1: 最大值法 - 寻找灰度值最大的点作为中心."""
        h, w = gray_image.shape
        half_roi = roi_size // 2
        best_score = -1
        best_center = (0, 0)

        for y in range(half_roi, h - half_roi):
            for x in range(half_roi, w - half_roi):
                roi = gray_image[y - half_roi : y + half_roi + 1, x - half_roi : x + half_roi + 1]
                max_val = np.max(roi)

                if max_val > best_score:
                    best_score = max_val
                    best_center = (x, y)

        # 提取最佳ROI数据
        x, y = best_center
        roi_data = gray_image[y - half_roi : y + half_roi + 1, x - half_roi : x + half_roi + 1]

        return {
            "center_x": x,
            "center_y": y,
            "confidence": float(best_score / 255.0),
            "roi_mean": float(np.mean(roi_data)),
            "roi_max": int(np.max(roi_data)),
            "roi_min": int(np.min(roi_data)),
            "method_description": "寻找ROI内最大灰度值最高的位置",
        }

    def method_mean_intensity(self, gray_image, roi_size):
        """方法2: 均值强度法 - 寻找ROI均值最大的位置."""
        h, w = gray_image.shape
        half_roi = roi_size // 2
        best_score = -1
        best_center = (0, 0)

        for y in range(half_roi, h - half_roi):
            for x in range(half_roi, w - half_roi):
                roi = gray_image[y - half_roi : y + half_roi + 1, x - half_roi : x + half_roi + 1]
                mean_val = np.mean(roi)

                if mean_val > best_score:
                    best_score = mean_val
                    best_center = (x, y)

        x, y = best_center
        roi_data = gray_image[y - half_roi : y + half_roi + 1, x - half_roi : x + half_roi + 1]

        return {
            "center_x": x,
            "center_y": y,
            "confidence": float(best_score / 255.0),
            "roi_mean": float(np.mean(roi_data)),
            "roi_max": int(np.max(roi_data)),
            "roi_min": int(np.min(roi_data)),
            "method_description": "寻找ROI内均值最高的位置",
        }

    def method_weighted_centroid(self, gray_image, roi_size):
        """方法3: 加权质心法 - 基于灰度值加权计算质心."""
        h, w = gray_image.shape
        half_roi = roi_size // 2

        # 计算全局加权质心
        total_weight = 0
        weighted_x = 0
        weighted_y = 0

        for y in range(h):
            for x in range(w):
                weight = gray_image[y, x] ** 2  # 平方加权突出高值
                total_weight += weight
                weighted_x += x * weight
                weighted_y += y * weight

        if total_weight > 0:
            center_x = int(weighted_x / total_weight)
            center_y = int(weighted_y / total_weight)
        else:
            center_x, center_y = w // 2, h // 2

        # 边界检查
        center_x = max(half_roi, min(w - half_roi - 1, center_x))
        center_y = max(half_roi, min(h - half_roi - 1, center_y))

        roi_data = gray_image[
            center_y - half_roi : center_y + half_roi + 1, center_x - half_roi : center_x + half_roi + 1
        ]

        return {
            "center_x": center_x,
            "center_y": center_y,
            "confidence": float(np.mean(roi_data) / 255.0),
            "roi_mean": float(np.mean(roi_data)),
            "roi_max": int(np.max(roi_data)),
            "roi_min": int(np.min(roi_data)),
            "method_description": "基于灰度值平方加权计算质心位置",
        }

    def method_gradient_magnitude(self, gray_image, roi_size):
        """方法4: 梯度幅值法 - 寻找梯度变化最大的区域."""
        # 计算梯度
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        h, w = gray_image.shape
        half_roi = roi_size // 2
        best_score = -1
        best_center = (0, 0)

        for y in range(half_roi, h - half_roi):
            for x in range(half_roi, w - half_roi):
                roi_grad = gradient_magnitude[y - half_roi : y + half_roi + 1, x - half_roi : x + half_roi + 1]
                roi_intensity = gray_image[y - half_roi : y + half_roi + 1, x - half_roi : x + half_roi + 1]

                # 结合梯度和强度
                score = np.mean(roi_grad) * np.mean(roi_intensity) / 255.0

                if score > best_score:
                    best_score = score
                    best_center = (x, y)

        x, y = best_center
        roi_data = gray_image[y - half_roi : y + half_roi + 1, x - half_roi : x + half_roi + 1]

        return {
            "center_x": x,
            "center_y": y,
            "confidence": float(best_score),
            "roi_mean": float(np.mean(roi_data)),
            "roi_max": int(np.max(roi_data)),
            "roi_min": int(np.min(roi_data)),
            "method_description": "结合梯度幅值和灰度强度寻找边缘清晰的高温区域",
        }

    def method_contrast_enhancement(self, gray_image, roi_size):
        """方法5: 对比度增强法 - 寻找与背景对比度最大的区域."""
        h, w = gray_image.shape
        half_roi = roi_size // 2
        best_score = -1
        best_center = (0, 0)

        # 计算全局背景均值
        global_mean = np.mean(gray_image)

        for y in range(half_roi, h - half_roi):
            for x in range(half_roi, w - half_roi):
                roi = gray_image[y - half_roi : y + half_roi + 1, x - half_roi : x + half_roi + 1]

                # 计算ROI与背景的对比度
                roi_mean = np.mean(roi)
                contrast_score = (roi_mean - global_mean) / (global_mean + 1e-6)

                if contrast_score > best_score:
                    best_score = contrast_score
                    best_center = (x, y)

        x, y = best_center
        roi_data = gray_image[y - half_roi : y + half_roi + 1, x - half_roi : x + half_roi + 1]

        return {
            "center_x": x,
            "center_y": y,
            "confidence": float(max(0, best_score)),
            "roi_mean": float(np.mean(roi_data)),
            "roi_max": int(np.max(roi_data)),
            "roi_min": int(np.min(roi_data)),
            "method_description": "寻找与全局背景对比度最大的区域",
        }

    def method_local_variance(self, gray_image, roi_size):
        """方法6: 局部方差法 - 寻找内部方差小但与周围差异大的区域."""
        h, w = gray_image.shape
        half_roi = roi_size // 2
        best_score = -1
        best_center = (0, 0)

        for y in range(half_roi, h - half_roi):
            for x in range(half_roi, w - half_roi):
                roi = gray_image[y - half_roi : y + half_roi + 1, x - half_roi : x + half_roi + 1]

                # ROI内部的均匀性（方差越小越好）
                roi_var = np.var(roi)
                roi_mean = np.mean(roi)

                # 与周围8个方向的差异
                surrounding_diff = 0
                count = 0
                directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

                for dy, dx in directions:
                    ny, nx = y + dy * roi_size, x + dx * roi_size
                    if 0 <= ny < h and 0 <= nx < w:
                        surrounding_diff += abs(gray_image[ny, nx] - roi_mean)
                        count += 1

                if count > 0:
                    avg_diff = surrounding_diff / count
                    # 分数 = 高强度 * 高差异 * 低方差
                    score = roi_mean * avg_diff / (1 + roi_var)

                    if score > best_score:
                        best_score = score
                        best_center = (x, y)

        x, y = best_center
        roi_data = gray_image[y - half_roi : y + half_roi + 1, x - half_roi : x + half_roi + 1]

        return {
            "center_x": x,
            "center_y": y,
            "confidence": float(best_score / (255.0 * 100)),  # 归一化
            "roi_mean": float(np.mean(roi_data)),
            "roi_max": int(np.max(roi_data)),
            "roi_min": int(np.min(roi_data)),
            "method_description": "寻找内部均匀但与周围差异明显的区域",
        }

    def method_temperature_cluster(self, gray_image, roi_size):
        """方法7: 温度聚类法 - 寻找高温像素密度最大的区域."""
        h, w = gray_image.shape
        half_roi = roi_size // 2
        best_score = -1
        best_center = (0, 0)

        # 定义高温阈值（75%分位数）
        high_temp_threshold = np.percentile(gray_image, 75)

        for y in range(half_roi, h - half_roi):
            for x in range(half_roi, w - half_roi):
                roi = gray_image[y - half_roi : y + half_roi + 1, x - half_roi : x + half_roi + 1]

                # 计算高温像素比例和强度
                high_temp_mask = roi >= high_temp_threshold
                high_temp_ratio = np.sum(high_temp_mask) / roi.size
                high_temp_intensity = np.mean(roi[high_temp_mask]) if np.any(high_temp_mask) else 0

                # 综合评分
                score = high_temp_ratio * high_temp_intensity

                if score > best_score:
                    best_score = score
                    best_center = (x, y)

        x, y = best_center
        roi_data = gray_image[y - half_roi : y + half_roi + 1, x - half_roi : x + half_roi + 1]

        return {
            "center_x": x,
            "center_y": y,
            "confidence": float(best_score / 255.0),
            "roi_mean": float(np.mean(roi_data)),
            "roi_max": int(np.max(roi_data)),
            "roi_min": int(np.min(roi_data)),
            "method_description": "寻找高温像素密度和强度都高的区域",
        }

    def method_edge_density(self, gray_image, roi_size):
        """方法8: 边缘密度法 - 寻找边缘密度适中且强度高的区域."""
        # 使用Canny边缘检测
        edges = cv2.Canny(gray_image, 50, 150)

        h, w = gray_image.shape
        half_roi = roi_size // 2
        best_score = -1
        best_center = (0, 0)

        for y in range(half_roi, h - half_roi):
            for x in range(half_roi, w - half_roi):
                roi_intensity = gray_image[y - half_roi : y + half_roi + 1, x - half_roi : x + half_roi + 1]
                roi_edges = edges[y - half_roi : y + half_roi + 1, x - half_roi : x + half_roi + 1]

                # 边缘密度
                edge_density = np.sum(roi_edges > 0) / roi_edges.size
                intensity_mean = np.mean(roi_intensity)

                # 平衡边缘密度和强度（避免过多噪声）
                score = intensity_mean * (1 - abs(edge_density - 0.3))  # 期望边缘密度约30%

                if score > best_score:
                    best_score = score
                    best_center = (x, y)

        x, y = best_center
        roi_data = gray_image[y - half_roi : y + half_roi + 1, x - half_roi : x + half_roi + 1]

        return {
            "center_x": x,
            "center_y": y,
            "confidence": float(best_score / 255.0),
            "roi_mean": float(np.mean(roi_data)),
            "roi_max": int(np.max(roi_data)),
            "roi_min": int(np.min(roi_data)),
            "method_description": "寻找边缘特征适中且强度高的目标区域",
        }

    def save_results_to_txt(self, output_file="roi_analysis_results.txt"):
        """保存结果到txt文件."""
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("ROI多方法分析结果报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            for img_result in self.results:
                f.write(f"图像: {img_result['image_name']}\n")
                f.write(f"尺寸: {img_result['image_shape']}\n")
                f.write(f"ROI大小: {img_result['roi_size']}x{img_result['roi_size']}\n")
                f.write("-" * 60 + "\n")

                # 按置信度排序方法
                methods_sorted = sorted(
                    [(name, data) for name, data in img_result["methods"].items() if "error" not in data],
                    key=lambda x: x[1].get("confidence", 0),
                    reverse=True,
                )

                f.write(
                    f"{'方法名称':<20} {'中心坐标':<12} {'置信度':<10} {'ROI均值':<10} {'最大值':<8} {'最小值':<8}\n"
                )
                f.write("-" * 80 + "\n")

                for method_name, data in methods_sorted:
                    f.write(
                        f"{method_name:<20} ({data['center_x']:>2},{data['center_y']:>2})      {data['confidence']:<10.4f} {data['roi_mean']:<10.2f} {data['roi_max']:<8} {data['roi_min']:<8}\n"
                    )

                f.write("\n方法详细描述:\n")
                for method_name, data in methods_sorted:
                    f.write(f"• {method_name}: {data['method_description']}\n")

                # 推荐最佳方法
                if methods_sorted:
                    best_method = methods_sorted[0]
                    f.write(f"\n🏆 推荐方法: {best_method[0]}\n")
                    f.write(f"   最佳中心点: ({best_method[1]['center_x']}, {best_method[1]['center_y']})\n")
                    f.write(f"   置信度: {best_method[1]['confidence']:.4f}\n")

                f.write("\n" + "=" * 80 + "\n\n")

        print(f"📄 结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="ROI多方法对比分析器")
    parser.add_argument(
        "--images",
        nargs="+",
        default=[
            "manual_patches/manual_195032_031s_01f_pos337x288_size30.png",
            "manual_patches/manual_194532_030s_01f_pos345x288_size30.png",
        ],
        help="要分析的图像文件路径列表",
    )
    parser.add_argument("--roi-size", type=int, default=5, help="ROI区域大小")
    parser.add_argument("--output", default="roi_analysis_results.txt", help="输出结果文件名")

    args = parser.parse_args()

    analyzer = ROIMultiMethodAnalyzer()

    # 分析每张图像
    for image_path in args.images:
        if Path(image_path).exists():
            analyzer.analyze_image(image_path, args.roi_size)
        else:
            print(f"⚠️  图像文件不存在: {image_path}")

    # 保存结果
    if analyzer.results:
        analyzer.save_results_to_txt(args.output)
        print(f"\n🎉 分析完成！共处理 {len(analyzer.results)} 张图像")
    else:
        print("❌ 没有成功分析任何图像")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO11 超小目标检测模型结构图生成工具
基于 yolo11-ultra-small.yaml 配置生成架构可视化图
"""

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免Qt依赖
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
import numpy as np

def create_yolo11_architecture():
    """创建 YOLO11 超小目标检测架构图"""
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # 颜色定义
    colors = {
        'input': '#FF6B6B',
        'backbone': '#4ECDC4', 
        'neck': '#45B7D1',
        'head': '#96CEB4',
        'detection': '#FFEAA7',
        'arrow': '#74B9FF',
        'text': '#2D3436'
    }
    
    # 标题
    ax.text(10, 13.5, 'YOLO11 超小目标检测架构 (Ultra Small Target)', 
            fontsize=20, fontweight='bold', ha='center', color=colors['text'])
    ax.text(10, 13, '针对红外场景下的飞机小目标检测优化', 
            fontsize=14, ha='center', color='gray', style='italic')
    
    # 输入层
    input_box = FancyBboxPatch((0.5, 11), 2, 1, boxstyle="round,pad=0.1", 
                              facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 11.5, 'Input\n640×640×3', fontsize=10, ha='center', va='center', 
            fontweight='bold', color='white')
    
    # Backbone 部分
    backbone_stages = [
        ('P1/2', (3.5, 11), '320×320×64'),
        ('P2/4', (5.5, 11), '160×160×128'),
        ('P3/8', (7.5, 11), '80×80×256'),
        ('P4/16', (9.5, 11), '40×40×512'),
        ('P5/32', (11.5, 11), '20×20×1024')
    ]
    
    for i, (name, pos, shape) in enumerate(backbone_stages):
        box = FancyBboxPatch((pos[0]-0.75, pos[1]-0.4), 1.5, 0.8, 
                            boxstyle="round,pad=0.05", 
                            facecolor=colors['backbone'], edgecolor='black')
        ax.add_patch(box)
        ax.text(pos[0], pos[1], f'{name}\n{shape}', fontsize=8, ha='center', va='center', 
                fontweight='bold')
        
        # 添加箭头
        if i < len(backbone_stages) - 1:
            arrow = FancyArrowPatch((pos[0]+0.75, pos[1]), (backbone_stages[i+1][1][0]-0.75, backbone_stages[i+1][1][1]),
                                  arrowstyle='->', mutation_scale=20, color=colors['arrow'], linewidth=2)
            ax.add_patch(arrow)
    
    # 标注 Backbone
    ax.text(7.5, 10.3, 'Backbone (YOLO11)', fontsize=12, ha='center', fontweight='bold', color=colors['text'])
    
    # SPPF 模块
    sppf_box = FancyBboxPatch((13, 10.6), 1.5, 0.8, boxstyle="round,pad=0.05", 
                             facecolor='#E17055', edgecolor='black')
    ax.add_patch(sppf_box)
    ax.text(13.75, 11, 'SPPF\n池化融合', fontsize=8, ha='center', va='center', fontweight='bold', color='white')
    
    # C2PSA 模块
    c2psa_box = FancyBboxPatch((15, 10.6), 1.5, 0.8, boxstyle="round,pad=0.05", 
                              facecolor='#A29BFE', edgecolor='black')
    ax.add_patch(c2psa_box)
    ax.text(15.75, 11, 'C2PSA\n注意力机制', fontsize=8, ha='center', va='center', fontweight='bold', color='white')
    
    # FPN/PAN Neck 部分
    neck_y = 8.5
    
    # 上采样路径
    upsample_positions = [(15, neck_y+1), (12, neck_y+1), (9, neck_y+1), (6, neck_y+1)]
    upsample_labels = ['特征融合', 'P4融合', 'P3融合', 'P2融合']
    
    for i, (pos, label) in enumerate(zip(upsample_positions, upsample_labels)):
        box = FancyBboxPatch((pos[0]-0.75, pos[1]-0.3), 1.5, 0.6, 
                            boxstyle="round,pad=0.05", 
                            facecolor=colors['neck'], edgecolor='black')
        ax.add_patch(box)
        ax.text(pos[0], pos[1], label, fontsize=8, ha='center', va='center', fontweight='bold')
        
        if i < len(upsample_positions) - 1:
            arrow = FancyArrowPatch((pos[0]-0.75, pos[1]), (upsample_positions[i+1][0]+0.75, upsample_positions[i+1][1]),
                                  arrowstyle='->', mutation_scale=15, color=colors['arrow'], linewidth=2)
            ax.add_patch(arrow)
    
    # 下采样路径
    downsample_positions = [(6, neck_y-1), (9, neck_y-1), (12, neck_y-1), (15, neck_y-1)]
    downsample_labels = ['P2→P3', 'P3→P4', 'P4→P5', 'P5输出']
    
    for i, (pos, label) in enumerate(zip(downsample_positions, downsample_labels)):
        box = FancyBboxPatch((pos[0]-0.75, pos[1]-0.3), 1.5, 0.6, 
                            boxstyle="round,pad=0.05", 
                            facecolor=colors['neck'], edgecolor='black')
        ax.add_patch(box)
        ax.text(pos[0], pos[1], label, fontsize=8, ha='center', va='center', fontweight='bold')
        
        if i < len(downsample_positions) - 1:
            arrow = FancyArrowPatch((pos[0]+0.75, pos[1]), (downsample_positions[i+1][0]-0.75, downsample_positions[i+1][1]),
                                  arrowstyle='->', mutation_scale=15, color=colors['arrow'], linewidth=2)
            ax.add_patch(arrow)
    
    # 标注 Neck
    ax.text(10.5, 7.8, 'Neck (FPN + PAN)', fontsize=12, ha='center', fontweight='bold', color=colors['text'])
    
    # 检测头部分
    head_y = 5.5
    head_positions = [(4, head_y), (7, head_y), (10, head_y), (13, head_y)]
    head_labels = ['P2检测头\n(超小目标)', 'P3检测头\n(小目标)', 'P4检测头\n(中目标)', 'P5检测头\n(大目标)']
    head_scales = ['1/4', '1/8', '1/16', '1/32']
    
    for pos, label, scale in zip(head_positions, head_labels, head_scales):
        # 检测头框
        box = FancyBboxPatch((pos[0]-1, pos[1]-0.5), 2, 1, 
                            boxstyle="round,pad=0.05", 
                            facecolor=colors['head'], edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(pos[0], pos[1], label, fontsize=9, ha='center', va='center', fontweight='bold')
        
        # 尺度标注
        ax.text(pos[0], pos[1]-0.8, f'步长{scale}', fontsize=8, ha='center', va='center', 
                color='red', fontweight='bold')
    
    # 特殊标注 P2 检测头
    p2_highlight = Rectangle((2.7, head_y-0.7), 2.6, 1.4, 
                           fill=False, edgecolor='red', linewidth=3, linestyle='--')
    ax.add_patch(p2_highlight)
    ax.text(4, head_y-1.2, '新增P2检测头\n专门处理超小目标', fontsize=9, ha='center', va='center', 
            color='red', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    # 输出部分
    output_y = 3
    output_positions = [(4, output_y), (7, output_y), (10, output_y), (13, output_y)]
    output_labels = ['超小目标\n检测结果', '小目标\n检测结果', '中目标\n检测结果', '大目标\n检测结果']
    
    for pos, label in zip(output_positions, output_labels):
        box = FancyBboxPatch((pos[0]-1, pos[1]-0.4), 2, 0.8, 
                            boxstyle="round,pad=0.05", 
                            facecolor=colors['detection'], edgecolor='black')
        ax.add_patch(box)
        ax.text(pos[0], pos[1], label, fontsize=8, ha='center', va='center', fontweight='bold')
        
        # 从检测头到输出的箭头
        arrow = FancyArrowPatch((pos[0], head_y-0.5), (pos[0], pos[1]+0.4),
                              arrowstyle='->', mutation_scale=15, color=colors['arrow'], linewidth=2)
        ax.add_patch(arrow)
    
    # NMS 融合
    nms_box = FancyBboxPatch((8, 1.5), 3, 0.8, boxstyle="round,pad=0.1", 
                            facecolor='#FD79A8', edgecolor='black', linewidth=2)
    ax.add_patch(nms_box)
    ax.text(9.5, 1.9, 'NMS 后处理 + 多尺度融合', fontsize=10, ha='center', va='center', 
            fontweight='bold', color='white')
    
    # 最终输出
    final_box = FancyBboxPatch((8, 0.2), 3, 0.8, boxstyle="round,pad=0.1", 
                              facecolor='#00B894', edgecolor='black', linewidth=2)
    ax.add_patch(final_box)
    ax.text(9.5, 0.6, '最终检测结果\n[x1,y1,x2,y2,conf,cls]', fontsize=10, ha='center', va='center', 
            fontweight='bold', color='white')
    
    # 创新点标注
    innovation_text = """
关键创新点：
• P2检测头：1/4尺度细节保留
• DFL损失：精确边界回归  
• 保守增强：避免小目标消失
• 权重优化：box=10, cls=0.3, dfl=2
• 低阈值策略：conf=0.001训练
    """
    
    ax.text(17, 6, innovation_text, fontsize=10, va='top', ha='left', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8),
            fontweight='bold')
    
    # 添加连接线显示特征流动
    # P2 特征连接
    arrow_p2 = FancyArrowPatch((5.5, 11), (6, neck_y+1),
                              arrowstyle='->', mutation_scale=12, color='red', linewidth=2, linestyle='--')
    ax.add_patch(arrow_p2)
    
    # P3 特征连接  
    arrow_p3 = FancyArrowPatch((7.5, 11), (9, neck_y+1),
                              arrowstyle='->', mutation_scale=12, color='blue', linewidth=2, linestyle='--')
    ax.add_patch(arrow_p3)
    
    plt.tight_layout()
    return fig

def main():
    """主函数"""
    print("🎨 生成 YOLO11 超小目标检测架构图...")
    
    # 创建架构图
    fig = create_yolo11_architecture()
    
    # 保存图片
    output_path = "/home/mingxing/worksapce/ultralytics/YOLO11_Ultra_Small_Architecture.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"✅ 架构图已保存到: {output_path}")
    print("📋 架构图包含以下内容:")
    print("   • 完整的 YOLO11 backbone 结构")
    print("   • 新增的 P2 检测头（超小目标专用）")
    print("   • FPN + PAN neck 特征融合")
    print("   • 四个尺度的检测输出")
    print("   • 关键创新点标注")
    
    # 显示图片（如果在图形环境中）
    print("💡 架构图已生成并保存为 PNG 文件")
    
    return output_path

if __name__ == "__main__":
    main()
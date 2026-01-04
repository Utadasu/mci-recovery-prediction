#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 阿尔茨海默病诊断系统 - 图像编码器架构可视化
===============================================

生成专业的图像编码器架构图，包含：
1. 主流程架构图 (参考U-Net风格)
2. 详细模块分解图 (参考DenseNet风格)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Polygon
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def create_image_encoder_architecture():
    """创建图像编码器完整架构图"""
    
    # 创建图形 - 调整尺寸和布局
    fig = plt.figure(figsize=(18, 10))
    
    # 定义柔和颜色方案 (参考示例图片)
    colors = {
        'input': '#6BB6FF',      # 浅蓝色 - 输入层
        'process': '#90C695',    # 浅绿色 - 处理层
        'attention': '#FFB366',  # 浅橙色 - 注意力机制
        'output': '#C88BFF',     # 浅紫色 - 输出层
        'control': '#FF8A8A',    # 浅红色 - 控制模块
        'text': '#2C2C2C',       # 深灰色 - 文本
        'arrow': '#4A4A4A',      # 灰色 - 箭头
        'frame': '#E8E8E8'       # 浅灰色 - 框架
    }
    
    # ==================== 子图1: 主架构流程 ====================
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_xlim(0, 20)
    ax1.set_ylim(0, 8)
    ax1.axis('off')
    ax1.set_title('图像编码器主架构流程图', fontsize=14, fontweight='bold', pad=15)
    
    # 绘制3D输入数据块 - 三个不重叠的立方体
    draw_separate_3d_cubes(ax1, 0.5, 4, colors)
    
    # 绘制预处理层
    preprocess_box = FancyBboxPatch((3, 3.5), 1.8, 1.5, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor=colors['process'], 
                                   edgecolor='black', alpha=0.8, linewidth=1)
    ax1.add_patch(preprocess_box)
    ax1.text(3.9, 4.25, 'Z-score标准化\n数据类型转换\n批次封装', 
             ha='center', va='center', fontsize=9, fontweight='bold')
    
    # 绘制骨干网络 (ResNet块) - 统一大小
    stages = ['Stage1\n12→24', 'Stage2\n24→48', 'Stage3\n48→96', 'Stage4\n96→192']
    stage_width = 1.5
    stage_height = 2
    for i, stage in enumerate(stages):
        x = 5.5 + i * 2
        stage_box = FancyBboxPatch((x, 3), stage_width, stage_height, 
                                  boxstyle="round,pad=0.05", 
                                  facecolor=colors['process'], 
                                  edgecolor='black', alpha=0.8, linewidth=1)
        ax1.add_patch(stage_box)
        ax1.text(x + stage_width/2, 4, stage, ha='center', va='center', 
                fontsize=9, fontweight='bold')
    
    # 绘制智能下采样层
    downsample_box = FancyBboxPatch((14, 4.5), 2, 1.5, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor=colors['attention'], 
                                   edgecolor='black', alpha=0.8, linewidth=1)
    ax1.add_patch(downsample_box)
    ax1.text(15, 5.25, '智能下采样\n1536→512\n(2,2,2)', 
             ha='center', va='center', fontsize=9, fontweight='bold')
    
    # 绘制CBAM3D注意力
    attention_box = FancyBboxPatch((14, 2), 2, 1.5, 
                                  boxstyle="round,pad=0.05", 
                                  facecolor=colors['attention'], 
                                  edgecolor='black', alpha=0.8, linewidth=1)
    ax1.add_patch(attention_box)
    ax1.text(15, 2.75, 'CBAM3D\n注意力机制\n通道+空间', 
             ha='center', va='center', fontsize=9, fontweight='bold')
    
    # 绘制输出层
    output_box = FancyBboxPatch((17, 3.5), 2, 1.5, 
                               boxstyle="round,pad=0.05", 
                               facecolor=colors['output'], 
                               edgecolor='black', alpha=0.8, linewidth=1)
    ax1.add_patch(output_box)
    ax1.text(18, 4.25, '512维特征\nL2标准化\n对比学习就绪', 
             ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # 绘制控制模块 (上方排列)
    control_modules = [
        ('早停策略\n(patience=15)', 3),
        ('学习率调度\n(ReduceLR)', 5.5),
        ('权重衰减\n(5e-4)', 8),
        ('混合精度训练', 10.5)
    ]
    
    for module, x in control_modules:
        control_box = FancyBboxPatch((x, 6.5), 1.8, 0.8, 
                                    boxstyle="round,pad=0.05", 
                                    facecolor=colors['control'], 
                                    edgecolor='black', alpha=0.8, linewidth=1)
        ax1.add_patch(control_box)
        ax1.text(x + 0.9, 6.9, module, ha='center', va='center', 
                fontsize=8, fontweight='bold')
    
    # 绘制数据流箭头 - 位置在框外
    draw_flow_arrows(ax1, colors['arrow'])
    
    # ==================== 子图2: 详细模块分解 ====================
    ax2 = plt.subplot(2, 1, 2)
    ax2.set_xlim(0, 20)
    ax2.set_ylim(0, 6)
    ax2.axis('off')
    ax2.set_title('CBAM3D注意力机制与智能下采样详细结构', fontsize=14, fontweight='bold', pad=15)
    
    # CBAM3D详细结构
    draw_cbam_detail_optimized(ax2, 1, 3, colors)
    
    # 智能下采样详细结构  
    draw_downsample_detail_optimized(ax2, 11, 3, colors)
    
    plt.tight_layout()
    return fig

def draw_separate_3d_cubes(ax, x, y, colors):
    """绘制三个分离的3D立方体表示多模态输入"""
    cube_size = 0.8
    cube_spacing = 1.2
    
    # 三个立方体的标签和位置
    cubes_info = [
        ('脑脊液', x, y, colors['input']),
        ('灰质', x, y - cube_spacing, colors['input']),
        ('白质', x, y - 2*cube_spacing, colors['input'])
    ]
    
    for label, cx, cy, color in cubes_info:
        # 绘制3D效果的立方体
        draw_single_3d_cube(ax, cx, cy, cube_size, color, label)
    
    # 添加整体尺寸标注
    ax.text(x + cube_size/2, y + 1, '[3,113,137,113]', 
           ha='center', va='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black'))

def draw_single_3d_cube(ax, x, y, size, color, label):
    """绘制单个3D立方体"""
    # 3D效果参数
    depth = 0.15
    
    # 前面
    front_face = FancyBboxPatch((x, y), size, size, 
                               boxstyle="round,pad=0.02", 
                               facecolor=color, edgecolor='black', 
                               alpha=0.9, linewidth=1)
    ax.add_patch(front_face)
    
    # 顶面
    top_points = np.array([
        [x, y + size], [x + size, y + size], 
        [x + size + depth, y + size + depth], [x + depth, y + size + depth]
    ])
    top_face = Polygon(top_points, facecolor=color, edgecolor='black', 
                      alpha=0.7, linewidth=1)
    ax.add_patch(top_face)
    
    # 右面
    right_points = np.array([
        [x + size, y], [x + size + depth, y + depth], 
        [x + size + depth, y + size + depth], [x + size, y + size]
    ])
    right_face = Polygon(right_points, facecolor=color, edgecolor='black', 
                        alpha=0.6, linewidth=1)
    ax.add_patch(right_face)
    
    # 标签
    ax.text(x + size/2, y + size/2, label, ha='center', va='center', 
           fontsize=8, fontweight='bold', color='white')

def draw_flow_arrows(ax, arrow_color):
    """绘制优化的数据流箭头"""
    # 箭头参数
    arrow_props = dict(arrowstyle='->', lw=2, color=arrow_color)
    
    # 主流程箭头
    arrows = [
        # 输入到预处理
        ((2.3, 4.25), (3, 4.25)),
        # 预处理到Stage1
        ((4.8, 4.25), (5.5, 4)),
        # Stage间连接
        ((7, 4), (7.5, 4)),
        ((9, 4), (9.5, 4)),
        ((11, 4), (11.5, 4)),
        # Stage4到下采样
        ((13, 4), (14, 5.25)),
        # 下采样到注意力的垂直连接
        ((15, 4.5), (15, 3.5)),
        # 注意力回流
        ((15, 3.5), (15, 4.5)),
        # 下采样到输出
        ((16, 5.25), (17, 4.25))
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start, arrowprops=arrow_props)

def draw_cbam_detail_optimized(ax, x, y, colors):
    """绘制优化的CBAM3D详细结构"""
    # 框架
    frame = FancyBboxPatch((x-0.5, y-2), 8.5, 4, 
                          boxstyle="round,pad=0.1", 
                          facecolor=colors['frame'], edgecolor='black', 
                          alpha=0.3, linewidth=1.5)
    ax.add_patch(frame)
    
    ax.text(x+4, y+1.7, 'CBAM3D注意力机制详细结构', 
           ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 输入特征
    input_box = FancyBboxPatch((x, y-0.3), 1.4, 0.8, 
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['input'], edgecolor='black', linewidth=1)
    ax.add_patch(input_box)
    ax.text(x+0.7, y+0.1, '输入特征\n[B,512,2,2,2]', 
           ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    
    # 通道注意力分支
    channel_boxes = [
        ('全局平均池化', x+2, y+0.3),
        ('全局最大池化', x+2, y-0.3),
        ('共享MLP\n512→32→512', x+4, y),
        ('Sigmoid激活', x+6, y)
    ]
    
    for text, bx, by in channel_boxes:
        box = FancyBboxPatch((bx, by-0.25), 1.4, 0.6, 
                            boxstyle="round,pad=0.05", 
                            facecolor=colors['attention'], edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(bx+0.7, by+0.05, text, ha='center', va='center', 
               fontsize=7, fontweight='bold')
    
    # 空间注意力分支
    spatial_boxes = [
        ('通道维度池化', x+2, y-1),
        ('3D卷积\n2→1通道', x+4, y-1),
        ('Sigmoid激活', x+6, y-1)
    ]
    
    for text, bx, by in spatial_boxes:
        box = FancyBboxPatch((bx, by-0.25), 1.4, 0.6, 
                            boxstyle="round,pad=0.05", 
                            facecolor=colors['attention'], edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(bx+0.7, by+0.05, text, ha='center', va='center', 
               fontsize=7, fontweight='bold')
    
    # 特征增强
    enhance_box = FancyBboxPatch((x+1, y-1.8), 2.4, 0.6, 
                                boxstyle="round,pad=0.05", 
                                facecolor=colors['output'], edgecolor='black', linewidth=1)
    ax.add_patch(enhance_box)
    ax.text(x+2.2, y-1.5, '特征增强 = 原特征 × 注意力权重', 
           ha='center', va='center', fontsize=8, color='white', fontweight='bold')

def draw_downsample_detail_optimized(ax, x, y, colors):
    """绘制优化的智能下采样详细结构"""
    # 框架
    frame = FancyBboxPatch((x-0.5, y-2), 7.5, 4, 
                          boxstyle="round,pad=0.1", 
                          facecolor=colors['frame'], edgecolor='black', 
                          alpha=0.3, linewidth=1.5)
    ax.add_patch(frame)
    
    ax.text(x+3, y+1.7, '智能下采样层结构', 
           ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 处理步骤 - 重新排列为2x3网格
    steps = [
        ('输入特征\n[B,1536,H,W,D]', x, y+0.5),
        ('3D卷积\n1536→512', x+2.5, y+0.5),
        ('批标准化\nBatchNorm3d', x+5, y+0.5),
        ('ReLU激活', x, y-0.7),
        ('自适应池化\n→(2,2,2)', x+2.5, y-0.7),
        ('输出特征\n[B,512,2,2,2]', x+5, y-0.7)
    ]
    
    for i, (text, bx, by) in enumerate(steps):
        color = colors['process'] if i < 5 else colors['output']
        text_color = 'white' if i == 5 else 'black'
        
        box = FancyBboxPatch((bx, by-0.3), 1.8, 0.7, 
                            boxstyle="round,pad=0.05", 
                            facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(bx+0.9, by+0.05, text, ha='center', va='center', 
               fontsize=8, color=text_color, fontweight='bold')

def save_architecture_diagram():
    """保存架构图"""
    fig = create_image_encoder_architecture()
    
    # 创建models目录
    import os
    os.makedirs('./models', exist_ok=True)
    
    # 保存为高质量图片
    plt.savefig('./models/image_encoder_architecture.png', 
                dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('./models/image_encoder_architecture.pdf', 
                bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print("✅ 图像编码器架构图已保存:")
    print("   📄 PNG格式: ./models/image_encoder_architecture.png")
    print("   📄 PDF格式: ./models/image_encoder_architecture.pdf")
    
    # 显示图形
    plt.show()
    
    return fig

if __name__ == "__main__":
    print("🎨 正在生成优化的图像编码器架构图...")
    save_architecture_diagram()
    print("🎉 架构图生成完成!") 
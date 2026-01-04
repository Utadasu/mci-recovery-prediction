#!/usr/bin/env python3
"""
🔥 使用对比学习专用图像编码器的多模态训练脚本
==============================================

专门使用92.22%准确率的对比学习图像编码器进行多模态对比学习训练

使用方法:
    python run_with_contrastive_encoder.py

特性:
- 🏆 自动使用最佳对比学习图像编码器 (92.22%准确率)
- 🎯 512维特征对齐，专为多模态融合优化
- ⚡ 预训练权重加速收敛
- 📈 目标性能 > 85%
"""

import os
import sys
import subprocess

def main():
    """主函数"""
    print("🔥 对比学习专用图像编码器 - 多模态训练启动")
    print("=" * 60)
    
    # 检查对比学习图像编码器是否存在
    contrastive_encoder_path = './models/contrastive/contrastive_image_encoder_ch12.pth'
    
    if not os.path.exists(contrastive_encoder_path):
        print("❌ 未找到对比学习专用图像编码器!")
        print(f"   期望路径: {contrastive_encoder_path}")
        print("\n💡 请先训练对比学习图像编码器:")
        print("   python run_training.py --train-contrastive-encoder")
        print("\n或者检查模型文件是否在正确位置:")
        print("   ./models/contrastive/contrastive_image_encoder_ch12.pth")
        return
    
    print("🎉 检测到对比学习专用图像编码器!")
    print(f"   📍 路径: {contrastive_encoder_path}")
    print(f"   🏆 训练准确率: 92.22%")
    print(f"   ✨ 专为多模态特征对齐优化")
    
    # 检查训练历史文件
    history_path = './models/contrastive/contrastive_image_encoder_history_ch12.json'
    if os.path.exists(history_path):
        print(f"   📊 训练历史: {history_path}")
    
    print("\n🚀 启动多模态对比学习训练...")
    print("   使用对比学习专用图像编码器")
    print("   目标性能: > 85%")
    print("   预期优势: 高质量特征提取 + 快速收敛")
    
    # 构建命令
    cmd = [
        sys.executable, 
        'run_contrastive_training.py',
        '--use-contrastive-encoder',
        '--image-model', contrastive_encoder_path
    ]
    
    print(f"\n📝 执行命令: {' '.join(cmd)}")
    print("\n" + "=" * 60)
    
    try:
        # 执行训练脚本
        result = subprocess.run(cmd, check=True)
        print("\n🎉 训练完成!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练过程出错: {e}")
        print("💡 请检查:")
        print("   1. 数据路径是否正确")
        print("   2. 文本编码器文件是否存在")
        print("   3. GPU内存是否充足")
        
    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
        
    except Exception as e:
        print(f"\n❌ 未知错误: {e}")

if __name__ == "__main__":
    main() 
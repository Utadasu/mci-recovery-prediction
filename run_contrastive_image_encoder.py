1.#!/usr/bin/env python3
"""
🔥 图像编码器预训练脚本 - 5折交叉验证版本
===========================================

功能特性:
- 🚀 专门训练图像编码器，为后续多模态对比学习提供预训练权重。
- 🎯 纯图像分类训练 (AD vs CN)，不涉及文本数据。
- 📊 5折分层交叉验证，确保模型鲁棒性。
- 🔧 智能配置选择、环境检测和命令行覆盖。
- 💡 支持消融实验，如禁用CBAM注意力模块。

模型架构 (3D CBAM):
--------------------
本脚本训练的模型基于带有3D卷积块注意力模块(CBAM)的ResNet架构。
CBAM包含两个子模块：通道注意力和空间注意力，用于自适应地提炼特征。

1. **3D CBAM 整体架构**
   输入特征首先通过通道注意力模块，然后其输出再经过空间注意力模块，实现顺序特征提炼。

   ```mermaid
   graph TD
       subgraph "模块：3D CBAM 整体架构"
           direction TB
           F_in["输入特征 F<br/>(C x D x H x W)"] --> ChannelAtt;
           
           ChannelAtt["3D 通道注意力模块<br/>(Channel Attention)"];
           
           F_in -- " " --> Mul1;
           ChannelAtt -- "生成 M_c (C x 1 x 1 x 1)" --> Mul1;
           
           Mul1["⊗<br/>逐元素乘法"];
           
           Mul1 --> F_prime;
           F_prime["F': 通道优化特征图<br/>(C x D x H x W)"];
           
           F_prime --> SpatialAtt;
           SpatialAtt["3D 空间注意力模块<br/>(Spatial Attention)"];
           
           F_prime -- " " --> Mul2;
           SpatialAtt -- "生成 M_s (1 x D x H x W)" --> Mul2;
           
           Mul2["⊗<br/>逐元素乘法"];
           
           Mul2 --> F_double_prime;
           F_double_prime["F'': 最终精炼特征图<br/>(C x D x H x W)"];
       end
   ```

2. **3D 通道注意力模块 (Channel Attention)**
   此模块关注输入数据的"什么"是有意义的。它通过全局池化和共享MLP为每个通道生成一个权重。

   ```mermaid
   graph TD
       subgraph "模块：3D通道注意力 (Channel Attention)"
           direction TB
           F_in["输入特征 F<br/>(C x D x H x W)"] -- "并行化" --> Pool;
           
           subgraph "Pool [并行池化]"
               direction LR
               AvgPool["全局平均池化<br/>(Global AvgPool)"];
               MaxPool["全局最大池化<br/>(Global MaxPool)"];
           end
           
           Pool -- "输出 (C x 1 x 1 x 1)" --> SharedMLP;
           
           subgraph "SharedMLP [共享多层感知机 (Shared MLP)]"
               direction TB
               mlp_in["(C x 1 x 1 x 1)"];
               mlp_fc1["FC1: C -> C/r"];
               mlp_relu["ReLU"];
               mlp_fc2["FC2: C/r -> C"];
               mlp_out["(C x 1 x 1 x 1)"];
               
               mlp_in --> mlp_fc1 --> mlp_relu --> mlp_fc2 --> mlp_out;
           end
           
           SharedMLP -- "逐元素相加" --> Add;
           Add["⊕<br/>Element-wise<br/>Sum"];
           
           Add -- "Sigmoid激活" --> Sigmoid;
           Sigmoid["Sigmoid"];
           
           Sigmoid -- "生成通道注意力图" --> M_c;
           M_c["M_c: 通道注意力图<br/>(C x 1 x 1 x 1)"];
       end
   ```

3. **3D 空间注意力模块 (Spatial Attention)**
   此模块关注特征的"哪里"是重要的。它通过跨通道池化和3D卷积来生成一个空间注意力图。

   ```mermaid
   graph TD
       subgraph "模块：3D空间注意力 (Spatial Attention)"
           direction TB
           F_prime_in["输入特征 F'<br/>(C x D x H x W)"] -- "跨通道池化" --> ChannelPool;
           
           subgraph "ChannelPool [跨通道池化]"
               direction LR
               AvgPool["通道平均池化<br/>(AvgPool over channels)"];
               MaxPool["通道最大池化<br/>(MaxPool over channels)"];
           end
           
           ChannelPool -- "生成特征图<br/>(1 x D x H x W)" --> Concat;
           
           Concat["Concat<br/>特征拼接<br/>(2 x D x H x W)"];
           
           Concat -- "3D卷积 + BN + ReLU" --> Conv3D;
           Conv3D["7x7x7 Conv3d<br/>(2 x D x H x W) -> (1 x D x H x W)"];
           
           Conv3D -- "Sigmoid激活" --> Sigmoid;
           Sigmoid["Sigmoid"];
           
           Sigmoid -- "生成空间注意力图" --> M_s;
           M_s["M_s: 空间注意力图<br/>(1 x D x H x W)"];
       end
   ```
"""

import os
import sys
import torch
import argparse
import numpy as np
import random
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
import json

# --- 动态路径设置 ---
# 将当前脚本所在目录的父目录添加到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

# --- 模块导入 ---
try:
    from train_image_encoder_for_contrastive import ContrastiveImageEncoderTrainer
    from data_utils import load_early_fusion_data, get_default_data_path
except ImportError as e:
    print(f"❌ 关键模块导入失败: {e}")
    print("请确保 train_image_encoder_for_contrastive.py 和 data_utils.py 在项目路径中。")
    sys.exit(1)

# --- 环境检查 ---
def check_environment():
    """检查运行环境，特别是CUDA"""
    print("🔍 检查运行环境...")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ GPU可用: {gpu_name} ({gpu_memory:.1f}GB)")
        return True, gpu_memory
    else:
        print("⚠️ CUDA不可用，将使用CPU。训练速度会非常慢。")
        return False, 0

def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"🌱 随机种子已设置为: {seed}")

# --- 核心训练逻辑 ---
def run_cross_validation(config):
    """运行完整的5折交叉验证训练流程"""
    print("\n" + "="*20 + " 🚀 开始5折交叉验证 " + "="*20)
    
    # 1. 加载数据
    data_path = config.get('data_path')
    try:
        images, labels = load_early_fusion_data(data_path, max_samples=config.get('max_samples'))
        print(f"📊 数据加载完成: {len(labels)}个样本 (AD: {np.sum(labels==1)}, CN: {np.sum(labels==0)})")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 2. 初始化交叉验证
    kfold = StratifiedKFold(
        n_splits=config['num_folds'], 
        shuffle=True, 
        random_state=config['random_state']
    )
    
    # 3. 初始化训练器和结果记录
    trainer = ContrastiveImageEncoderTrainer(device=config['device'])
    all_fold_results = []

    # 4. 循环训练每个折
    for fold, (train_idx, val_idx) in enumerate(kfold.split(images, labels)):
        print(f"\n--- Fold {fold + 1}/{config['num_folds']} ---")
        
        train_images, val_images = images[train_idx], images[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]

        # 训练并获取该折的结果
        fold_history = trainer.train_single_fold(
            fold_idx=fold,
            train_images=train_images,
            train_labels=train_labels,
            val_images=val_images,
            val_labels=val_labels,
            **config  # 传递所有配置，包括use_cbam
        )
        
        if fold_history:
            all_fold_results.append(fold_history)
            print(f"✅ Fold {fold + 1} 完成. 最佳验证准确率: {fold_history['best_val_accuracy']:.4f}")
        else:
            print(f"❌ Fold {fold + 1} 训练失败。")

    # 5. 汇总并保存结果
    if all_fold_results:
        save_cv_results(all_fold_results, trainer.save_dir)
    else:
        print("❌ 所有折的训练均失败，无法生成结果。")

def save_cv_results(results, save_dir):
    """汇总并保存交叉验证结果"""
    val_accuracies = [res['best_val_accuracy'] for res in results]
    mean_accuracy = np.mean(val_accuracies)
    std_accuracy = np.std(val_accuracies)

    summary = {
        'mean_validation_accuracy': mean_accuracy,
        'std_validation_accuracy': std_accuracy,
        'best_fold_accuracy': max(val_accuracies),
        'worst_fold_accuracy': min(val_accuracies),
        'individual_fold_accuracies': val_accuracies,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'full_history': results
    }

    print("\n" + "="*20 + " 📊 交叉验证最终结果 " + "="*20)
    print(f"   平均验证准确率: {summary['mean_validation_accuracy']:.4f} ± {summary['std_validation_accuracy']:.4f}")
    print(f"   各折准确率: {[f'{acc:.4f}' for acc in summary['individual_fold_accuracies']]}")
    
    # 保存到JSON文件
    save_path = os.path.join(save_dir, 'cv_results.json')
    try:
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=4, cls=np.encoder if 'encoder' in dir(np) else None)
        print(f"✅ 交叉验证结果已保存到: {save_path}")
    except Exception as e:
        print(f"❌ 保存结果失败: {e}")

# --- 主函数和参数解析 ---
def main():
    """主函数：解析参数并启动训练流程"""
    parser = argparse.ArgumentParser(description='🔥 图像编码器预训练脚本 (5折交叉验证)')
    
    # 主要参数
    parser.add_argument('--config', type=str, choices=['high', 'standard', 'memory', 'test'], default='auto', help='选择一个预设的训练配置。默认: auto')
    parser.add_argument('--data-path', type=str, default=None, help='覆盖默认的数据集路径。')
    
    # 覆盖配置参数
    parser.add_argument('--epochs', type=int, default=None, help='覆盖配置中的训练轮数。')
    parser.add_argument('--batch-size', type=int, default=None, help='覆盖配置中的批次大小。')
    
    # 消融实验开关
    parser.add_argument('--no-cbam', action='store_true', help='[消融实验] 禁用CBAM注意力模块。')
    parser.add_argument('--no-cv', action='store_true', help='禁用交叉验证，进行单次训练（用于快速测试）。')

    args = parser.parse_args()
    
    # ❗️为了确保每次运行结果一致，在此设置全局随机种子
    set_seed(42)

    print("=" * 60)
    print("🔥 图像编码器预训练启动")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 环境检查
    cuda_available, gpu_memory = check_environment()
    device = 'cuda' if cuda_available else 'cpu'

    # 2. 获取数据路径
    data_path = args.data_path or get_default_data_path()
    if not data_path:
        print("❌ 未找到数据路径，请使用 --data-path 指定。")
        sys.exit(1)

    # 3. 配置选择
    config_choice = args.config
    if config_choice == 'auto':
        if gpu_memory >= 32: config_choice = 'high'
        elif gpu_memory >= 16: config_choice = 'standard'
        else: config_choice = 'memory'
        print(f"🤖 已根据GPU显存({gpu_memory:.1f}GB)自动选择配置: '{config_choice}'")

    # 4. 定义预设配置
    configs = {
        'high':     {'base_channels': 12, 'num_epochs': 60, 'batch_size': 8,  'learning_rate': 1e-4, 'patience': 20},
        'standard': {'base_channels': 12, 'num_epochs': 40, 'batch_size': 8,  'learning_rate': 1e-4, 'patience': 15},
        'memory':   {'base_channels': 8,  'num_epochs': 40, 'batch_size': 4, 'learning_rate': 1.5e-4, 'patience': 15},
        'test':     {'base_channels': 4,  'num_epochs': 5,  'batch_size': 2,  'learning_rate': 1e-4, 'patience': 3, 'max_samples': 40}
    }
    train_config = configs[config_choice]

    # 5. 应用通用设置和命令行覆盖
    train_config.update({
        'device': device,
        'data_path': data_path,
        'num_folds': 1 if args.no_cv else 5,
        'random_state': 42,
        'use_cbam': not args.no_cbam  # 消融实验开关
    })

    if args.epochs: train_config['num_epochs'] = args.epochs
    if args.batch_size: train_config['batch_size'] = args.batch_size
    
    print("\n⚙️ 最终训练配置:")
    for key, value in train_config.items():
        print(f"   - {key}: {value}")

    # 6. 启动训练
    run_cross_validation(train_config)
    
    print("\n🎉 训练流程全部完成!")

if __name__ == '__main__':
    main() 
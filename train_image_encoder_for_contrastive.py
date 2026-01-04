#!/usr/bin/env python3
"""
🔥 对比学习图像编码器预训练脚本
===============================

功能特性:
- 🎯 专门为对比学习系统训练图像编码器
- 🧠 使用智能下采样层ImprovedResNetCBAM3D架构
- 📊 输出512维特征，与文本编码器对齐
- 💾 保存到对比学习预训练路径
- 🔧 优化的训练策略和参数
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

# 导入项目模块
from optimized_models import ImprovedResNetCBAM3D
from data_utils import load_early_fusion_data
from losses import ImprovedFocalLoss

# CUDA优化设置
torch.cuda.set_per_process_memory_fraction(0.95)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def get_default_data_path():
    """获取默认数据路径"""
    # 服务器默认路径
    server_path = "/root/autodl-tmp/DATA_MCI/test_data/"
    if os.path.exists(server_path):
        return server_path
    
    # 新项目路径下的本地调试路径
    new_project_paths = [
        "/autodl-fs/data/ZM_Files/备份5.27/test_data/",
        "/autodl-fs/data/ZM_Files/备份5.27/../test_data/",
        "/autodl-fs/data/test_data/"
    ]
    
    for path in new_project_paths:
        if os.path.exists(path):
            return path
    
    # 原有本地调试路径
    local_paths = [
        "./test_data/",
        "../test_data/",
        "../../test_data/"
    ]
    
    for path in local_paths:
        if os.path.exists(path):
            return path
    
    return server_path

class ContrastiveImageEncoderTrainer:
    """
    对比学习图像编码器训练器
    专门为多模态对比学习系统训练图像编码器
    """
    def __init__(self, device='cuda'):
        self.device = device
        
        # 🔥 对比学习专用保存路径
        self.save_dir = './models/contrastive'
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 训练历史记录
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epochs': []
        }
        
        print(f"🚀 ContrastiveImageEncoderTrainer 初始化完成")
        print(f"   设备: {device}")
        print(f"   保存目录: {self.save_dir}")
        print(f"   目标: 为对比学习系统训练图像编码器")
    
    def create_contrastive_image_model(self, base_channels=12, use_cbam=True):
        """
        创建对比学习专用的图像编码器模型
        
        Args:
            base_channels: 基础通道数
            use_cbam: 是否使用CBAM注意力模块
        """
        print(f"\n🔧 创建对比学习图像编码器...")
        print(f"   架构: ImprovedResNetCBAM3D + 智能下采样层")
        print(f"   基础通道数: {base_channels}")
        print(f"   使用CBAM: {'✅' if use_cbam else '❌'}")
        print(f"   输出特征: 512维 (与文本编码器对齐)")
        
        # 创建使用智能下采样层的模型
        model = ImprovedResNetCBAM3D(
            in_channels=3,
            num_classes=2,  # AD vs CN
            base_channels=base_channels,
            dropout_rate=0.3,
            use_global_pool=False,  # 🔥 使用智能下采样层
            use_cbam=use_cbam  # 传递CBAM开关
        ).to(self.device)
        
        # 修改fusion层，输出512维特征以匹配文本编码器
        fusion_input_dim = base_channels * 16 * 2 * 2 * 2  # 智能下采样层输出维度
        
        model.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),  # 🎯 输出512维特征
            nn.LayerNorm(512),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3)
        ).to(self.device)
        
        # 保留分类头用于预训练
        model.classifier = nn.Linear(512, 2).to(self.device)
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   总参数量: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        print(f"   特征维度: {fusion_input_dim} → 1024 → 512")
        
        return model
    
    def prepare_data(self, data_path, batch_size=4, max_samples=None):
        """准备训练数据 - 改进版本，确保患者级别分割"""
        print(f"\n📊 准备训练数据...")
        print(f"   数据路径: {data_path}")
        print(f"   批次大小: {batch_size}")
        
        # 加载早期融合数据
        image_data, labels = load_early_fusion_data(data_path, max_samples=max_samples)
        
        print(f"   图像数据形状: {image_data.shape}")
        print(f"   标签形状: {labels.shape}")
        print(f"   AD样本数: {np.sum(labels==1)}")
        print(f"   CN样本数: {np.sum(labels==0)}")
        
        # 🔥 改进：患者级别数据分割，避免数据泄露
        # 假设每个类别的样本是按患者顺序排列的
        ad_indices = np.where(labels == 1)[0]
        cn_indices = np.where(labels == 0)[0]
        
        # 患者级别分割 - 假设每个患者有多个扫描
        # 为了安全起见，我们按索引分组来模拟患者分割
        def patient_level_split(indices, train_ratio=0.8):
            """患者级别分割"""
            n_samples = len(indices)
            # 假设每10个样本来自同一患者（这是一个简化假设）
            patient_size = 10
            n_patients = n_samples // patient_size
            
            # 随机打乱患者顺序
            patient_indices = np.arange(n_patients)
            np.random.seed(42)  # 固定随机种子
            np.random.shuffle(patient_indices)
            
            # 分割患者
            n_train_patients = int(n_patients * train_ratio)
            train_patients = patient_indices[:n_train_patients]
            val_patients = patient_indices[n_train_patients:]
            
            # 获取对应的样本索引
            train_indices = []
            val_indices = []
            
            for p in train_patients:
                start_idx = p * patient_size
                end_idx = min((p + 1) * patient_size, n_samples)
                train_indices.extend(indices[start_idx:end_idx])
            
            for p in val_patients:
                start_idx = p * patient_size
                end_idx = min((p + 1) * patient_size, n_samples)
                val_indices.extend(indices[start_idx:end_idx])
            
            return train_indices, val_indices
        
        # 分别对AD和CN进行患者级别分割
        ad_train_indices, ad_val_indices = patient_level_split(ad_indices, train_ratio=0.8)
        cn_train_indices, cn_val_indices = patient_level_split(cn_indices, train_ratio=0.8)
        
        # 合并训练集和验证集索引
        train_indices = ad_train_indices + cn_train_indices
        val_indices = ad_val_indices + cn_val_indices
        
        # 打乱训练集索引
        np.random.seed(42)
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        
        print(f"   🔄 患者级别分割结果:")
        print(f"     训练集: {len(train_indices)} 样本")
        print(f"     验证集: {len(val_indices)} 样本")
        print(f"     训练集AD: {np.sum(labels[train_indices] == 1)}")
        print(f"     训练集CN: {np.sum(labels[train_indices] == 0)}")
        print(f"     验证集AD: {np.sum(labels[val_indices] == 1)}")
        print(f"     验证集CN: {np.sum(labels[val_indices] == 0)}")
        
        # 创建数据集
        from torch.utils.data import TensorDataset, Subset
        
        # 创建PyTorch张量
        image_tensor = torch.FloatTensor(image_data)
        label_tensor = torch.LongTensor(labels)
        
        # 创建完整数据集
        full_dataset = TensorDataset(image_tensor, label_tensor)
        
        # 创建训练集和验证集
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        
        # 🔧 改进：为训练集和验证集使用不同的数据加载策略
        # 训练集：使用数据增强，较小batch size
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=False,
            drop_last=True  # 丢弃最后不完整的batch
        )
        
        # 验证集：不使用数据增强，可以使用稍大的batch size
        val_batch_size = min(batch_size * 2, 8)  # 验证时可以用更大的batch
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,  # 验证集不需要打乱
            num_workers=2,
            pin_memory=False,
            drop_last=False
        )
        
        print(f"   训练集大小: {len(train_dataset)}")
        print(f"   验证集大小: {len(val_dataset)}")
        print(f"   训练批次数: {len(train_loader)}")
        print(f"   验证批次数: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, model, val_loader, criterion, epoch):
        """验证一个epoch"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch:02d} [Val]")
            
            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device)
                
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, data_path, base_channels=12, num_epochs=50, batch_size=4, 
              learning_rate=1e-4, max_samples=None, patience=15):
        """
        主训练函数 - 改进版本，添加过拟合检测和正则化
        
        Args:
            data_path: 数据路径
            base_channels: 基础通道数
            num_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            max_samples: 最大样本数
            patience: 早停耐心值
        """
        print(f"\n🚀 开始训练对比学习图像编码器")
        print(f"   目标: 为多模态对比学习系统提供预训练图像编码器")
        print(f"   训练轮数: {num_epochs}")
        print(f"   学习率: {learning_rate}")
        print(f"   批次大小: {batch_size}")
        print(f"   早停耐心: {patience}")
        
        # 创建模型
        model = self.create_contrastive_image_model(base_channels)
        
        # 准备数据
        train_loader, val_loader = self.prepare_data(data_path, batch_size, max_samples)
        
        # 🔧 改进：更强的正则化策略
        criterion = ImprovedFocalLoss(alpha=1.0, gamma=2.0, reduction='mean')
        
        # 使用更强的权重衰减
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
        
        # 改进的学习率调度策略
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6
        )
        
        # 训练记录
        best_val_acc = 0.0
        patience_counter = 0
        best_model_path = None
        
        # 🚨 过拟合检测参数
        overfitting_threshold = 10.0  # 验证准确率比训练准确率高出10%就认为可能过拟合
        consecutive_overfitting = 0
        max_consecutive_overfitting = 3
        
        print(f"\n📈 开始训练循环...")
        print(f"🚨 过拟合检测: 当验证准确率持续比训练准确率高{overfitting_threshold}%时将触发警告")
        
        for epoch in range(1, num_epochs + 1):
            # 训练阶段
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion, epoch)
            
            # 验证阶段
            val_loss, val_acc = self.validate_epoch(model, val_loader, criterion, epoch)
            
            # 学习率调度 - 基于验证准确率
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['epochs'].append(epoch)
            
            # 🚨 过拟合检测
            acc_diff = val_acc - train_acc
            if acc_diff > overfitting_threshold:
                consecutive_overfitting += 1
                print(f"  🚨 过拟合警告: 验证准确率比训练准确率高 {acc_diff:.2f}% (连续{consecutive_overfitting}次)")
                
                if consecutive_overfitting >= max_consecutive_overfitting:
                    print(f"  ⚠️  检测到严重过拟合，建议检查数据分割或增加正则化")
            else:
                consecutive_overfitting = 0
            
            # 打印结果
            print(f"\nEpoch {epoch:02d}/{num_epochs}:")
            print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"  准确率差异: {acc_diff:+.2f}% (验证-训练)")
            print(f"  学习率: {current_lr:.6f}")
            
            # 🔧 改进的模型保存策略：考虑过拟合情况
            save_model = False
            
            if val_acc > best_val_acc:
                # 如果验证准确率提升，但要检查是否过拟合严重
                if acc_diff <= overfitting_threshold * 1.5:  # 允许适度的验证准确率优势
                    save_model = True
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    print(f"  ⚠️  验证准确率提升但疑似过拟合严重，不保存模型")
                    patience_counter += 1
            else:
                patience_counter += 1
            
            if save_model:
                # 🔥 保存到对比学习专用路径
                model_filename = f"contrastive_image_encoder_ch{base_channels}.pth"
                best_model_path = os.path.join(self.save_dir, model_filename)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'train_loss': train_loss,
                    'acc_diff': acc_diff,
                    'base_channels': base_channels,
                    'feature_dim': 512,
                    'architecture': 'ImprovedResNetCBAM3D_SmartDownsample',
                    'config': {  # 🔥 添加config字段供ImprovedImageEncoder读取
                        'base_channels': base_channels,
                        'feature_dim': 512,
                        'use_global_pool': False,
                        'dropout_rate': 0.3,
                        'in_channels': 3,
                        'num_classes': 2
                    },
                    'training_config': {
                        'num_epochs': num_epochs,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'use_global_pool': False,
                        'weight_decay': 5e-4,
                        'patient_level_split': True
                    }
                }, best_model_path)
                
                print(f"  ✅ 新的最佳模型! 验证准确率: {val_acc:.2f}% (Epoch {epoch})")
                print(f"     模型已保存: {best_model_path}")
            else:
                print(f"  ⏳ 耐心计数: {patience_counter}/{patience}")
            
            # 早停检查
            if patience_counter >= patience:
                print(f"\n⏹️  早停触发! 最佳验证准确率: {best_val_acc:.2f}%")
                break
            
            # 🚨 如果连续过拟合太严重，提前停止
            if consecutive_overfitting >= max_consecutive_overfitting * 2:
                print(f"\n🚨 检测到严重过拟合，提前停止训练")
                print(f"   建议: 1) 检查数据分割是否正确 2) 增加正则化 3) 减少模型复杂度")
                break
        
        # 保存训练历史
        history_path = os.path.join(self.save_dir, f"contrastive_image_encoder_history_ch{base_channels}.json")
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        print(f"\n🎉 训练完成!")
        print(f"   最佳验证准确率: {best_val_acc:.2f}%")
        print(f"   最佳模型路径: {best_model_path}")
        print(f"   训练历史: {history_path}")
        
        # 🔍 训练总结和建议
        if best_model_path:
            # 加载最佳模型检查点来获取详细信息
            checkpoint = torch.load(best_model_path, map_location='cpu')
            final_acc_diff = checkpoint.get('acc_diff', 0)
            
            print(f"\n📊 训练总结:")
            print(f"   最佳模型的准确率差异: {final_acc_diff:+.2f}%")
            
            if abs(final_acc_diff) <= 5.0:
                print(f"   ✅ 模型训练良好，无明显过拟合")
            elif final_acc_diff > 5.0:
                print(f"   ⚠️  存在轻微过拟合，但在可接受范围内")
            else:
                print(f"   🔧 训练准确率高于验证准确率，可能需要更多训练")
            
            print(f"   🎯 模型已准备用于对比学习系统!")
        else:
            print(f"   ❌ 未能训练出满意的模型，建议调整超参数")
        
        return model, best_val_acc, best_model_path
    
    def save_training_plots(self, base_channels=12):
        """保存训练曲线图"""
        if not self.train_history['epochs']:
            return
        
        plt.figure(figsize=(15, 5))
        
        # 损失曲线
        plt.subplot(1, 3, 1)
        plt.plot(self.train_history['epochs'], self.train_history['train_loss'], 'b-', label='训练损失')
        plt.plot(self.train_history['epochs'], self.train_history['val_loss'], 'r-', label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('训练损失曲线')
        plt.legend()
        plt.grid(True)
        
        # 准确率曲线
        plt.subplot(1, 3, 2)
        plt.plot(self.train_history['epochs'], self.train_history['train_acc'], 'b-', label='训练准确率')
        plt.plot(self.train_history['epochs'], self.train_history['val_acc'], 'r-', label='验证准确率')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('训练准确率曲线')
        plt.legend()
        plt.grid(True)
        
        # 学习曲线对比
        plt.subplot(1, 3, 3)
        plt.plot(self.train_history['epochs'], self.train_history['val_acc'], 'g-', linewidth=2, label='验证准确率')
        plt.axhline(y=max(self.train_history['val_acc']), color='r', linestyle='--', 
                   label=f'最佳: {max(self.train_history["val_acc"]):.2f}%')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy (%)')
        plt.title('验证准确率趋势')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = os.path.join(self.save_dir, f"contrastive_image_encoder_training_ch{base_channels}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 训练曲线已保存: {plot_path}")

    def train_single_fold(self, fold_idx, train_images, train_labels, val_images, val_labels,
                         base_channels=12, num_epochs=50, batch_size=4, learning_rate=1e-4, 
                         patience=15, use_cbam=True, **kwargs):
        """
        🎯 训练单个交叉验证折
        
        Args:
            fold_idx: 当前折索引
            train_images: 训练图像数据
            train_labels: 训练标签
            val_images: 验证图像数据  
            val_labels: 验证标签
            base_channels: 基础通道数
            num_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            patience: 早停耐心轮数
            use_cbam: 是否使用CBAM注意力模块
            **kwargs: 其他配置参数
            
        Returns:
            dict: 训练结果
        """
        print(f"\n🎯 开始第{fold_idx+1}折训练...")
        print(f"   训练样本: {len(train_labels)}")
        print(f"   验证样本: {len(val_labels)}")
        print(f"   基础通道数: {base_channels}")
        print(f"   训练轮数: {num_epochs}")
        print(f"   学习率: {learning_rate}")
        print(f"   早停耐心: {patience}")
        
        # 重置训练历史
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epochs': []
        }
        
        # 创建模型
        model = self.create_contrastive_image_model(
            base_channels=base_channels,
            use_cbam=use_cbam
        )
        
        # 准备数据加载器
        from torch.utils.data import TensorDataset, DataLoader
        
        # 转换为张量
        train_images_tensor = torch.FloatTensor(train_images)
        train_labels_tensor = torch.LongTensor(train_labels)
        val_images_tensor = torch.FloatTensor(val_images)
        val_labels_tensor = torch.LongTensor(val_labels)
        
        # 创建数据集和加载器
        train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
        val_dataset = TensorDataset(val_images_tensor, val_labels_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
        
        # 损失函数和优化器
        criterion = ImprovedFocalLoss(alpha=1.0, gamma=2.0, reduction='mean')
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=False, min_lr=1e-6
        )
        
        # 训练记录
        best_val_acc = 0.0
        patience_counter = 0
        best_epoch = 0
        
        print(f"   开始训练循环...")
        
        for epoch in range(1, num_epochs + 1):
            # 训练阶段
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion, epoch)
            
            # 验证阶段
            val_loss, val_acc = self.validate_epoch(model, val_loader, criterion, epoch)
            
            # 学习率调度
            scheduler.step(val_acc)
            
            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['epochs'].append(epoch)
            
            # 更新最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                
                # 保存当前折的最佳模型
                fold_model_path = os.path.join(self.save_dir, f"fold_{fold_idx}_best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'fold_idx': fold_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'train_loss': train_loss,
                    'base_channels': base_channels,
                    'feature_dim': 512,
                    'config': {
                        'base_channels': base_channels,
                        'feature_dim': 512,
                        'use_global_pool': False,
                        'dropout_rate': 0.3,
                        'in_channels': 3,
                        'num_classes': 2
                    }
                }, fold_model_path)
                
            else:
                patience_counter += 1
            
            # 每10轮或最后一轮打印结果
            if epoch % 10 == 0 or epoch == num_epochs or patience_counter >= patience:
                print(f"   Epoch {epoch:02d}: Train={train_acc:.2f}%, Val={val_acc:.2f}%, Best={best_val_acc:.2f}%")
            
            # 早停检查
            if patience_counter >= patience:
                print(f"   ⏹️ 早停触发 (耐心={patience}), 最佳验证准确率: {best_val_acc:.2f}%")
                break
        
        # 保存此折的训练历史
        fold_history_path = os.path.join(self.save_dir, f"fold_{fold_idx}_history.json")
        with open(fold_history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        print(f"   ✅ 第{fold_idx+1}折完成: 最佳验证准确率 {best_val_acc:.4f}% (Epoch {best_epoch})")
        
        # 返回结果
        return {
            'fold_idx': fold_idx,
            'best_val_accuracy': best_val_acc,
            'best_epoch': best_epoch,
            'final_train_acc': self.train_history['train_acc'][-1] if self.train_history['train_acc'] else 0,
            'final_val_acc': self.train_history['val_acc'][-1] if self.train_history['val_acc'] else 0,
            'total_epochs': len(self.train_history['epochs']),
            'fold_model_path': os.path.join(self.save_dir, f"fold_{fold_idx}_best_model.pth"),
            'fold_history_path': fold_history_path,
            'converged': patience_counter < patience
        }

def main():
    """主函数"""
    print("🔥 对比学习图像编码器预训练脚本")
    print("=" * 60)
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查CUDA
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
        print(f"🚀 GPU: {gpu_name} ({gpu_memory}GB)")
    else:
        device = 'cpu'
        print("⚠️  使用CPU训练")
    
    # 获取数据路径
    data_path = get_default_data_path()
    print(f"📁 数据路径: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ 数据路径不存在: {data_path}")
        print("💡 请检查数据路径或运行数据准备脚本")
        return
    
    # 创建训练器
    trainer = ContrastiveImageEncoderTrainer(device=device)
    
    # 训练配置选择
    print(f"\n⚙️ 选择训练配置:")
    print("1. 🔥 高性能配置 (32GB+ GPU)")
    print("2. 💾 标准配置 (16GB+ GPU)")
    print("3. 🔧 内存优化配置 (<16GB GPU)")
    print("4. 🧪 快速测试配置 (调试用)")
    
    choice = input("请选择配置 (1-4): ").strip()
    
    if choice == "1":
        print("\n🔥 高性能配置")
        model, best_acc, model_path = trainer.train(
            data_path=data_path,
            base_channels=12,
            num_epochs=60,
            batch_size=8,
            learning_rate=1e-4,
            max_samples=None,
            patience=20
        )
        
    elif choice == "2":
        print("\n💾 标准配置")
        model, best_acc, model_path = trainer.train(
            data_path=data_path,
            base_channels=12,
            num_epochs=50,
            batch_size=4,
            learning_rate=1e-4,
            max_samples=None,
            patience=15
        )
        
    elif choice == "3":
        print("\n🔧 内存优化配置")
        model, best_acc, model_path = trainer.train(
            data_path=data_path,
            base_channels=8,
            num_epochs=40,
            batch_size=2,
            learning_rate=1e-4,
            max_samples=None,
            patience=15
        )
        
    elif choice == "4":
        print("\n🧪 快速测试配置")
        model, best_acc, model_path = trainer.train(
            data_path=data_path,
            base_channels=8,
            num_epochs=10,
            batch_size=4,
            learning_rate=1e-4,
            max_samples=50,  # 限制样本数
            patience=5
        )
        
    else:
        print("❌ 无效选择，使用默认标准配置")
        model, best_acc, model_path = trainer.train(
            data_path=data_path,
            base_channels=12,
            num_epochs=50,
            batch_size=4,
            learning_rate=1e-4,
            max_samples=None,
            patience=15
        )
    
    # 保存训练曲线
    base_channels = 12 if choice in ["1", "2"] else 8
    trainer.save_training_plots(base_channels)
    
    # 总结
    print(f"\n" + "=" * 60)
    print("🎉 对比学习图像编码器预训练完成")
    print("=" * 60)
    print(f"🎯 最佳验证准确率: {best_acc:.2f}%")
    print(f"💾 模型保存路径: {model_path}")
    print(f"📁 保存目录: {trainer.save_dir}")
    print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n📝 后续步骤:")
    print("1. 🔗 使用此模型进行多模态对比学习训练")
    print("2. 📊 在对比学习中加载此预训练权重")
    print("3. 🎯 期待多模态融合性能提升")
    
    print(f"\n🚀 启动对比学习训练:")
    print(f"python run_contrastive_training.py")

if __name__ == "__main__":
    main() 
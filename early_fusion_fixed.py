#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复版早期融合训练脚本
解决了内存不足、类别不平衡和BatchNorm问题
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda import amp
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime
import json
from sklearn.metrics import balanced_accuracy_score, f1_score

def create_memory_optimized_early_fusion_loaders(data_loaders, gpu_memory_gb=32, debug=True):
    """
    创建内存优化的早期融合数据加载器
    
    参数:
    - data_loaders: 包含各组织类型训练和验证数据加载器的字典
    - gpu_memory_gb: GPU显存大小(GB)
    - debug: 是否启用调试模式
    
    返回:
    - 包含早期融合训练和验证数据加载器的字典
    """
    print(f"\n===== 创建内存优化的早期融合数据集 =====")
    print(f"检测到GPU显存: {gpu_memory_gb:.1f}GB")
    
    # 根据GPU显存自动选择最佳配置
    if gpu_memory_gb >= 30:  # 32GB显存
        batch_size = 4
        num_workers = 4
        print("🔥 使用高端GPU配置")
    elif gpu_memory_gb >= 20:  # 24GB显存
        batch_size = 2
        num_workers = 2
        print("⚡ 使用中高端GPU配置")
    elif gpu_memory_gb >= 10:  # 16GB显存
        batch_size = 2
        num_workers = 2
        print("🎯 使用中端GPU配置")
    else:  # 8GB及以下
        batch_size = 1
        num_workers = 1
        print("💻 使用入门GPU配置")
    
    print(f"最终配置: batch_size={batch_size}, num_workers={num_workers}")
    
    # 导入数据集类
    from early_fusion import HierarchicalEarlyFusionDataset
    
    # 检查数据加载器结构并提取正确的加载器
    if 'train' in data_loaders and 'val' in data_loaders:
        # 新的数据结构: {'train': {...}, 'val': {...}, 'test': {...}}
        train_loaders = data_loaders['train']
        val_loaders = data_loaders['val']
        
        train_csf_loader = train_loaders['CSF']
        train_grey_loader = train_loaders['GRAY']
        train_white_loader = train_loaders['WHITE']
        
        val_csf_loader = val_loaders['CSF']
        val_grey_loader = val_loaders['GRAY']
        val_white_loader = val_loaders['WHITE']
        
        print("✅ 使用新的数据结构格式")
        
    elif 'train_CSF' in data_loaders:
        # 旧的数据结构: {'train_CSF': ..., 'val_CSF': ..., ...}
        train_csf_loader = data_loaders['train_CSF']
        train_grey_loader = data_loaders['train_GRAY']
        train_white_loader = data_loaders['train_WHITE']
        
        val_csf_loader = data_loaders['val_CSF']
        val_grey_loader = data_loaders['val_GRAY']
        val_white_loader = data_loaders['val_WHITE']
        
        print("✅ 使用旧的数据结构格式")
        
    else:
        raise ValueError(f"无法识别的数据加载器格式。可用键: {list(data_loaders.keys())}")
    
    # 创建融合数据集
    train_fusion_dataset = HierarchicalEarlyFusionDataset(
        train_csf_loader,
        train_grey_loader,
        train_white_loader,
        debug=debug
    )
    
    val_fusion_dataset = HierarchicalEarlyFusionDataset(
        val_csf_loader,
        val_grey_loader,
        val_white_loader,
        debug=debug
    )
    
    # 创建内存优化的数据加载器
    train_fusion_loader = torch.utils.data.DataLoader(
        train_fusion_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # 禁用以节省内存
        persistent_workers=False,  # 禁用持久化worker
        drop_last=True  # 丢弃最后一个不完整的批次
    )
    
    val_fusion_loader = torch.utils.data.DataLoader(
        val_fusion_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        drop_last=False
    )
    
    # 验证融合后的数据加载器
    print(f"✅ 训练融合加载器: 批次大小={train_fusion_loader.batch_size}, 样本数={len(train_fusion_dataset)}")
    print(f"✅ 验证融合加载器: 批次大小={val_fusion_loader.batch_size}, 样本数={len(val_fusion_dataset)}")
    
    # 测试数据加载
    try:
        sample_batch, sample_labels = next(iter(train_fusion_loader))
        print(f"✅ 融合后批次形状: {sample_batch.shape}, 标签形状: {sample_labels.shape}")
        print(f"✅ 标签分布: {torch.bincount(sample_labels)}")
        
        # 估算内存使用
        batch_memory = sample_batch.numel() * 4 / 1024**2  # MB
        print(f"✅ 单批次内存估算: {batch_memory:.1f}MB")
        
    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")
        raise
    
    return {
        'train': train_fusion_loader,
        'val': val_fusion_loader,
        'batch_size': batch_size,
        'num_workers': num_workers
    }

def train_memory_optimized_early_fusion(data_loaders, device, save_dir='./models'):
    """
    内存优化版的早期融合模型训练
    
    解决的问题:
    1. GPU内存不足 -> 自适应批次大小 + 梯度累积
    2. 类别不平衡 -> 改进的损失函数 + 类别权重
    3. BatchNorm问题 -> LayerNorm替代
    """
    print(f"\n===== 内存优化版早期融合模型训练 =====")
    
    # 内存优化设置
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # 检查GPU显存
    if torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(device)
        total_memory = gpu_properties.total_memory / 1024**3
        print(f"GPU: {gpu_properties.name}")
        print(f"总显存: {total_memory:.1f}GB")
        torch.cuda.empty_cache()
    else:
        total_memory = 4  # CPU默认值
    
    # 创建内存优化的数据加载器
    fusion_loaders = create_memory_optimized_early_fusion_loaders(
        data_loaders, 
        gpu_memory_gb=total_memory,
        debug=True
    )
    
    train_loader = fusion_loaders['train']
    val_loader = fusion_loaders['val']
    batch_size = fusion_loaders['batch_size']
    
    # 根据显存选择模型配置
    if total_memory >= 30:  # 32GB
        base_channels = 12  # 从8增加到12，提升模型容量
        accumulation_steps = 2
        print("🔥 使用增强标准模型配置")
    elif total_memory >= 20:  # 24GB
        base_channels = 8  # 从6增加到8
        accumulation_steps = 3  # 减少累积步数以利用更大模型
        print("⚡ 使用增强紧凑模型配置")
    else:  # 16GB及以下
        base_channels = 6
        accumulation_steps = 8
        print("💻 使用超紧凑模型配置")
    
    effective_batch_size = batch_size * accumulation_steps
    print(f"配置: base_channels={base_channels}, 梯度累积={accumulation_steps}步")
    print(f"等效批次大小: {effective_batch_size}")
    
    # 创建内存优化模型
    from optimized_models import create_improved_resnet3d
    
    model = create_improved_resnet3d(
        in_channels=3,
        device=device,
        base_channels=base_channels,
        dropout_rate=0.3
    )
    
    # 设置为评估模式测试，然后切换回训练模式
    model.eval()
    test_input = torch.randn(1, 3, 113, 137, 113).to(device)
    with torch.no_grad():
        test_output = model(test_input)
    print(f"✅ 模型测试成功: {test_input.shape} -> {test_output.shape}")
    del test_input, test_output
    torch.cuda.empty_cache()
    
    model.train()  # 切换回训练模式
    
    # 计算类别权重
    try:
        train_stats = train_loader.dataset.get_stats()
        ad_count = train_stats['ad_count']
        cn_count = train_stats['cn_count']
        total = ad_count + cn_count
        
        # 改进的类别权重计算，使用更平滑的权重范围
        class_weights = torch.FloatTensor([
            1.5 * total / (2 * ad_count),  # 基于样本比例的权重
            1.5 * total / (2 * cn_count)
        ]).clamp(0.5, 2.0).to(device)  # 限制权重在0.5-2.0之间，减少极端波动
        
        print(f"✅ 数据集统计: AD={ad_count}, CN={cn_count}")
        print(f"✅ 类别权重: {class_weights}")
        
    except Exception as e:
        print(f"⚠️ 无法获取类别统计，使用默认权重: {e}")
        class_weights = torch.FloatTensor([1.0, 1.0]).to(device)
    
    # 改进的Focal Loss - 添加类别平衡正则化
    class ImprovedFocalLoss(nn.Module):
        def __init__(self, alpha=None, gamma=2.0, weight=None, label_smoothing=0.05, balance_reg=0.1):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.weight = weight
            self.label_smoothing = label_smoothing
            self.balance_reg = balance_reg  # 类别平衡正则化系数
            
        def forward(self, inputs, targets):
            # 标签平滑
            num_classes = inputs.size(1)
            if self.label_smoothing > 0:
                targets_one_hot = F.one_hot(targets, num_classes).float()
                targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + \
                                self.label_smoothing / num_classes
            else:
                targets_one_hot = F.one_hot(targets, num_classes).float()
            
            # 计算交叉熵
            log_probs = F.log_softmax(inputs, dim=1)
            ce_loss = -(targets_one_hot * log_probs).sum(dim=1)
            
            # 计算概率
            probs = torch.exp(log_probs)
            target_probs = (targets_one_hot * probs).sum(dim=1)
            
            # Focal loss
            focal_weight = (1 - target_probs) ** self.gamma
            focal_loss = focal_weight * ce_loss
            
            # 应用类别权重
            if self.weight is not None:
                weight_t = self.weight[targets]
                focal_loss = focal_loss * weight_t
            
            # 类别平衡正则化：惩罚预测概率的极端偏差
            # 计算每个类别的平均预测概率
            class_probs = probs.mean(dim=0)
            # 计算类别平衡正则化项：鼓励类别概率接近均匀分布
            balance_loss = torch.sum((class_probs - 1/num_classes) ** 2)
            
            # 总损失 = Focal损失 + 平衡正则化
            total_loss = focal_loss.mean() + self.balance_reg * balance_loss
            
            return total_loss
    
    # 使用改进的损失函数
    criterion = ImprovedFocalLoss(
        gamma=2.5,  # 增加gamma值，更关注难分类样本
        weight=class_weights,
        label_smoothing=0.1  # 增加标签平滑，提高泛化能力
    )
    
    # 优化器配置
    lr = 0.0001
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=0.005,  # 降低权重衰减，减少正则化强度
        eps=1e-8
    )
    
    # 学习率调度器优化
    num_epochs = 60  # 增加到60轮，给模型更多时间优化
    warmup_epochs = 8  # 延长预热期
    
    # 使用更稳定的余弦退火调度器，增加重启机制
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs - warmup_epochs,
        eta_min=lr * 0.01,  # 适当提高最小学习率，避免学习率过低导致模型停滞
        verbose=False
    )
    
    # 梯度裁剪配置
    grad_clip_max_norm = 1.0  # 保持梯度裁剪强度
    print(f"✅ 优化器配置: AdamW, 初始LR={lr}, 权重衰减={optimizer.param_groups[0]['weight_decay']}")
    print(f"✅ 学习率调度: CosineAnnealingLR, 预热={warmup_epochs}轮, T_max={num_epochs-warmup_epochs}")
    print(f"✅ 梯度裁剪: max_norm={grad_clip_max_norm}")
    
    # 混合精度训练
    scaler = amp.GradScaler()
    
    # 训练统计
    stats = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_balanced_acc': [],
        'val_f1_score': [],
        'val_acc_per_class': [],
        'lr': []
    }
    
    # 训练状态
    best_val_acc = 0.0
    best_balanced_acc = 0.0  # 最佳平衡准确率
    best_model_state = None
    patience = 25  # 增加耐心值，从15增到25
    no_improve_epochs = 0
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n🚀 开始训练，总轮次: {num_epochs}")
    print(f"批次大小: {batch_size}, 梯度累积: {accumulation_steps}步")
    print(f"等效批次大小: {effective_batch_size}")
    
    for epoch in range(num_epochs):
        # 学习率预热
        if epoch < warmup_epochs:
            # 使用更平滑的预热曲线
            warmup_factor = min(1.0, (epoch + 1) / warmup_epochs)
            current_lr = lr * (0.01 + 0.99 * warmup_factor**2)  # 二次曲线预热
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            scheduler.step()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        optimizer.zero_grad()
        
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # 简单的数据增强：随机添加少量噪声
            if torch.rand(1).item() < 0.3:  # 30%概率添加噪声
                noise = torch.randn_like(inputs) * 0.01
                inputs = inputs + noise
            
            # 混合精度前向传播
            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps  # 归一化梯度累积
            
            # 反向传播
            scaler.scale(loss).backward()
            
            # 梯度累积
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # 统计
            train_loss += loss.item() * accumulation_steps
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_pbar.set_postfix({
                'loss': f'{loss.item() * accumulation_steps:.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
            
            # 定期清理内存
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        conf_matrix = torch.zeros(2, 2, dtype=torch.long)
        
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                with amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # 保存所有标签和预测，用于计算平衡准确率和F1分数
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                
                # 更新混淆矩阵
                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    conf_matrix[t.long(), p.long()] += 1
        
        # 计算指标
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # 计算平衡准确率和F1分数
        val_balanced_acc = 100. * balanced_accuracy_score(all_labels, all_predictions)
        val_f1 = 100. * f1_score(all_labels, all_predictions, average='weighted')
        
        # 计算每个类别的准确率
        val_acc_per_class = []
        for i in range(2):
            correct = conf_matrix[i, i].item()
            total = conf_matrix[i, :].sum().item()
            val_acc_per_class.append(100.0 * correct / max(1, total))
        
        # 记录统计
        stats['train_loss'].append(avg_train_loss)
        stats['train_acc'].append(train_acc)
        stats['val_loss'].append(avg_val_loss)
        stats['val_acc'].append(val_acc)
        stats['val_balanced_acc'].append(val_balanced_acc)
        stats['val_f1_score'].append(val_f1)
        stats['val_acc_per_class'].append(val_acc_per_class)
        stats['lr'].append(optimizer.param_groups[0]['lr'])
        
        # 打印信息
        print(f'\nEpoch [{epoch+1}/{num_epochs}] - 内存优化版:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Val Balanced Acc: {val_balanced_acc:.2f}%, Val F1: {val_f1:.2f}%')
        print(f'Val Acc per class: AD={val_acc_per_class[0]:.2f}%, CN={val_acc_per_class[1]:.2f}%')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'混淆矩阵:\n{conf_matrix}')
        
        # 检查是否两个类别都有预测
        both_classes_predicted = conf_matrix[0, 0] > 0 and conf_matrix[1, 1] > 0
        
        # 改进的最佳模型保存条件：同时考虑验证准确率和平衡准确率
        current_score = 0.7 * val_acc + 0.3 * val_balanced_acc  # 加权综合分数
        best_score = 0.7 * best_val_acc + 0.3 * best_balanced_acc
        
        if current_score > best_score and both_classes_predicted:
            # 更新最佳指标
            best_val_acc = val_acc
            best_balanced_acc = val_balanced_acc
            
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_balanced_acc': val_balanced_acc,
                'val_f1': val_f1,
                'val_loss': avg_val_loss,
                'stats': stats,
                'conf_matrix': conf_matrix.tolist()
            }
            
            model_path = os.path.join(save_dir, 'best_memory_optimized_early_fusion.pth')
            torch.save(best_model_state, model_path)
            print(f'✅ 保存最佳模型: {model_path}，综合分数: {current_score:.2f}')
            print(f'   - 验证准确率: {val_acc:.2f}%')
            print(f'   - 平衡准确率: {val_balanced_acc:.2f}%')
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if not both_classes_predicted:
                print(f'⚠️ 模型只预测单一类别，跳过保存')
            else:
                print(f'⚠️ 模型性能无改善 ({no_improve_epochs}/{patience})')
        
        # 早停检查
        if no_improve_epochs >= patience:
            print(f'早停在epoch {epoch+1}')
            break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state_dict'])
        print(f'✅ 已加载最佳模型，验证准确率: {best_val_acc:.2f}%')
    
    return {
        'model': model,
        'best_val_acc': best_val_acc,
        'best_epoch': best_model_state['epoch'] if best_model_state else -1,
        'model_path': model_path if best_model_state else None,
        'stats': stats,
        'final_conf_matrix': conf_matrix.tolist(),
        'architecture': 'ImprovedResNetCBAM3D-MemoryOptimized'
    }

if __name__ == "__main__":
    print("内存优化的早期融合训练脚本")
    print("使用方法:")
    print("1. 从main.py调用train_memory_optimized_early_fusion函数")
    print("2. 或者导入此模块使用相关函数") 
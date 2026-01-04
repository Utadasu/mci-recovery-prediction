#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速微调脚本 - 基于已训练模型进行快速性能提升
适用于时间紧张的情况
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os

def quick_finetune_model(model_path, data_loaders, device, epochs=5):
    """
    快速微调已训练模型
    
    参数:
    - model_path: 已训练模型路径
    - data_loaders: 数据加载器
    - device: 计算设备
    - epochs: 微调轮次(默认5轮)
    
    返回:
    - 微调后的性能
    """
    print(f"\n===== 快速微调模式 (仅{epochs}轮) =====")
    
    # 加载预训练模型
    from optimized_models import ImprovedResNetCBAM3D
    model = ImprovedResNetCBAM3D(
        in_channels=3,
        num_classes=2,
        base_channels=12,
        dropout_rate=0.3
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"✅ 加载预训练模型: {model_path}")
    
    # 数据加载器
    from early_fusion_fixed import create_memory_optimized_early_fusion_loaders
    fusion_loaders = create_memory_optimized_early_fusion_loaders(
        data_loaders, gpu_memory_gb=32, debug=False
    )
    
    train_loader = fusion_loaders['train']
    val_loader = fusion_loaders['val']
    
    # 优化器 - 使用更小的学习率进行微调
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.00001,  # 比初始训练小10倍
        weight_decay=0.001
    )
    
    # 学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    model.train()
    
    for epoch in range(epochs):
        # 训练阶段
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"微调 {epoch+1}/{epochs}")
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_acc = 100. * train_correct / train_total
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_acc:.2f}%'
            })
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        # 更新最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), './models/quick_finetuned_model.pth')
            print(f"✅ 保存微调模型，验证准确率: {val_acc:.2f}%")
        
        scheduler.step()
        model.train()
        
        print(f"轮次 [{epoch+1}/{epochs}] - 训练准确率: {100.*train_correct/train_total:.2f}%, "
              f"验证准确率: {val_acc:.2f}%")
    
    print(f"\n🎉 快速微调完成！最佳验证准确率: {best_val_acc:.2f}%")
    return best_val_acc

if __name__ == "__main__":
    # 示例用法
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "./models/best_memory_optimized_early_fusion.pth"
    
    # 这里需要传入实际的数据加载器
    # best_acc = quick_finetune_model(model_path, data_loaders, device)
    print("快速微调脚本就绪！") 
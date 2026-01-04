#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
深度架构微调 - 更深层次的信息保留优化
通过架构改进和渐进式训练来保留更多信息并提升性能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import torch.nn.functional as F

class DeepProgressiveResNetCBAM3D(nn.Module):
    """
    深层渐进式ResNetCBAM3D - 深度特征提取优化
    特点：
    1. 深度可分离卷积
    2. 渐进式特征聚合
    3. 跨层特征融合
    4. 自适应感受野调整
    """
    
    def __init__(self, in_channels=3, num_classes=2, base_channels=12, dropout_rate=0.3):
        super(DeepProgressiveResNetCBAM3D, self).__init__()
        
        self.base_channels = base_channels
        
        # 深层输入处理 - 渐进式特征提取
        self.progressive_conv_blocks = nn.ModuleList([
            # 第一层：细节特征提取
            DepthwiseSeparableConv3D(in_channels, base_channels, kernel_size=3),
            # 第二层：中层特征聚合  
            DepthwiseSeparableConv3D(base_channels, base_channels, kernel_size=5),
            # 第三层：深层语义特征
            DepthwiseSeparableConv3D(base_channels, base_channels, kernel_size=7),
        ])
        
        # 跨层特征融合模块
        self.cross_layer_fusion = CrossLayerFusion(base_channels, base_channels)
        
        # 自适应感受野调整
        self.adaptive_receptive_field = AdaptiveReceptiveField(base_channels)
        
        # 改进的CBAM注意力模块
        self.spatial_attention = SpatialAttention3D()
        self.channel_attention = ChannelAttention3D(base_channels)
        
        # 深层残差块序列 - 渐进式深度提取
        self.deep_residual_sequence = nn.ModuleList([
            DeepResidualBlock3D(base_channels, base_channels, dropout_rate),
            DeepResidualBlock3D(base_channels, base_channels * 2, dropout_rate, use_se=True),
            DeepResidualBlock3D(base_channels * 2, base_channels * 4, dropout_rate, use_se=True),
            DeepResidualBlock3D(base_channels * 4, base_channels * 4, dropout_rate, use_se=True),  # 新增深层
        ])
        
        # 特征重组和聚合
        self.feature_aggregation = FeatureAggregation(base_channels * 4)
        
        # 自适应全局池化
        self.adaptive_pool = nn.AdaptiveAvgPool3d((2, 2, 2))
        
        # 深层分类器 - 层次化特征分析
        feature_dim = base_channels * 4 * 2 * 2 * 2
        self.classifier = HierarchicalClassifier(feature_dim, num_classes, dropout_rate)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """改进的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # 使用He初始化，针对ReLU激活函数优化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 使用Xavier初始化
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 渐进式深度特征提取
        progressive_features = []
        current_features = x
        
        for i, conv_block in enumerate(self.progressive_conv_blocks):
            current_features = conv_block(current_features)
            progressive_features.append(current_features)
        
        # 跨层特征融合
        fused_features = self.cross_layer_fusion(progressive_features)
        
        # 自适应感受野调整
        adaptive_features = self.adaptive_receptive_field(fused_features)
        
        # 应用注意力机制
        attended_features = self.channel_attention(adaptive_features)
        attended_features = self.spatial_attention(attended_features)
        
        # 深层残差序列处理
        deep_features = attended_features
        residual_outputs = []
        
        for deep_block in self.deep_residual_sequence:
            deep_features = deep_block(deep_features)
            residual_outputs.append(deep_features)
        
        # 特征聚合
        aggregated_features = self.feature_aggregation(residual_outputs[-1])
        
        # 全局池化
        pooled_features = self.adaptive_pool(aggregated_features)
        
        # 扁平化并分类
        flattened = pooled_features.view(pooled_features.size(0), -1)
        output = self.classifier(flattened)
        
        return output


class DepthwiseSeparableConv3D(nn.Module):
    """深度可分离3D卷积 - 减少参数同时保持表达能力"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super(DepthwiseSeparableConv3D, self).__init__()
        
        if padding is None:
            padding = kernel_size // 2
        
        # 深度卷积：每个输入通道单独卷积
        self.depthwise = nn.Conv3d(
            in_channels, in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=in_channels
        )
        
        # 点卷积：1x1x1卷积组合特征
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x


class CrossLayerFusion(nn.Module):
    """跨层特征融合 - 融合不同深度的特征"""
    
    def __init__(self, channels, out_channels):
        super(CrossLayerFusion, self).__init__()
        
        # 特征权重学习
        self.weight_conv = nn.Conv3d(channels * 3, 3, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        
        # 特征融合
        self.fusion_conv = nn.Conv3d(channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, feature_list):
        # 确保所有特征具有相同的空间尺寸
        base_size = feature_list[0].shape[2:]
        aligned_features = []
        
        for feat in feature_list:
            if feat.shape[2:] != base_size:
                feat = F.interpolate(feat, size=base_size, mode='trilinear', align_corners=False)
            aligned_features.append(feat)
        
        # 计算特征权重
        concat_features = torch.cat(aligned_features, dim=1)
        weights = self.weight_conv(concat_features)
        weights = self.softmax(weights)
        
        # 加权融合
        fused = sum(w * f for w, f in zip(weights.split(1, dim=1), aligned_features))
        
        # 输出处理
        output = self.fusion_conv(fused)
        output = self.bn(output)
        output = self.relu(output)
        
        return output


class AdaptiveReceptiveField(nn.Module):
    """自适应感受野调整 - 动态调整感受野大小"""
    
    def __init__(self, channels):
        super(AdaptiveReceptiveField, self).__init__()
        
        # 多种感受野的卷积
        self.conv_3x3 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.conv_5x5 = nn.Conv3d(channels, channels, kernel_size=5, padding=2)
        self.conv_7x7 = nn.Conv3d(channels, channels, kernel_size=7, padding=3)
        
        # 感受野选择网络
        self.selection_net = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 4, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        self.bn = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # 计算不同感受野的特征
        feat_3x3 = self.conv_3x3(x)
        feat_5x5 = self.conv_5x5(x)
        feat_7x7 = self.conv_7x7(x)
        
        # 自适应选择权重
        selection_weights = self.selection_net(x)
        w1, w2, w3 = selection_weights[:, 0:1], selection_weights[:, 1:2], selection_weights[:, 2:3]
        
        # 加权融合
        adaptive_feat = w1 * feat_3x3 + w2 * feat_5x5 + w3 * feat_7x7
        
        # 残差连接
        output = adaptive_feat + x
        output = self.bn(output)
        output = self.relu(output)
        
        return output


class DeepResidualBlock3D(nn.Module):
    """深层残差块 - 增强版残差学习"""
    
    def __init__(self, in_channels, out_channels, dropout_rate=0.3, use_se=False):
        super(DeepResidualBlock3D, self).__init__()
        
        # 主分支
        self.conv1 = nn.Conv3d(in_channels, out_channels // 2, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(out_channels // 2)
        
        self.conv2 = nn.Conv3d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels // 2)
        
        self.conv3 = nn.Conv3d(out_channels // 2, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm3d(out_channels)
        
        # SE模块（可选）
        self.use_se = use_se
        if use_se:
            self.se_module = SEModule3D(out_channels)
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm3d(out_channels)
            )
        
        self.dropout = nn.Dropout3d(dropout_rate)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        
        out = self.bn3(self.conv3(out))
        
        if self.use_se:
            out = self.se_module(out)
        
        out += residual
        out = self.relu(out)
        
        return out


class SEModule3D(nn.Module):
    """3D Squeeze-and-Excitation模块"""
    
    def __init__(self, channels, reduction=16):
        super(SEModule3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class FeatureAggregation(nn.Module):
    """特征聚合模块 - 整合深层特征"""
    
    def __init__(self, channels):
        super(FeatureAggregation, self).__init__()
        
        # 全局上下文建模
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 2, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 局部细节增强
        self.local_enhance = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # 全局上下文权重
        global_weight = self.global_context(x)
        
        # 局部特征增强
        local_feat = self.local_enhance(x)
        
        # 融合全局和局部信息
        enhanced = x * global_weight + local_feat
        enhanced = self.bn(enhanced)
        enhanced = self.relu(enhanced)
        
        return enhanced


class HierarchicalClassifier(nn.Module):
    """层次化分类器 - 渐进式决策"""
    
    def __init__(self, input_dim, num_classes, dropout_rate=0.3):
        super(HierarchicalClassifier, self).__init__()
        
        # 多层渐进式分类
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(input_dim // 4, input_dim // 8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate // 2)  # 最后层降低dropout
        )
        
        self.final_classifier = nn.Linear(input_dim // 8, num_classes)
        
        # 辅助分类器（可选，用于深度监督）
        self.aux_classifier = nn.Linear(input_dim // 4, num_classes)
    
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        
        # 主要输出
        main_output = self.final_classifier(x3)
        
        # 训练时可以添加辅助损失
        if self.training:
            aux_output = self.aux_classifier(x2)
            return main_output, aux_output
        else:
            return main_output


class ChannelAttention3D(nn.Module):
    """3D通道注意力模块"""
    
    def __init__(self, channels, reduction=16):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention3D(nn.Module):
    """3D空间注意力模块"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention_input))
        return x * attention


def deep_architecture_finetune(model_path, data_loaders, device, epochs=10):
    """
    深度架构微调
    
    参数:
    - model_path: 原始模型路径
    - data_loaders: 数据加载器
    - device: 计算设备
    - epochs: 训练轮次
    
    返回:
    - 微调后的性能
    """
    print(f"\n===== 深度架构微调 (增强版模型，{epochs}轮) =====")
    
    # 创建增强版模型
    enhanced_model = DeepProgressiveResNetCBAM3D(
        in_channels=3,
        num_classes=2,
        base_channels=12,
        dropout_rate=0.2  # 降低dropout以保留更多信息
    ).to(device)
    
    print("✅ 创建深层渐进式架构模型")
    print(f"   - 深度可分离卷积")
    print(f"   - 渐进式特征聚合")
    print(f"   - 跨层特征融合")
    print(f"   - 自适应感受野调整")
    print(f"   - SE模块增强")
    print(f"   - 层次化分类器")
    
    # 数据加载器
    from early_fusion_fixed import create_memory_optimized_early_fusion_loaders
    fusion_loaders = create_memory_optimized_early_fusion_loaders(
        data_loaders, gpu_memory_gb=32, debug=False
    )
    
    train_loader = fusion_loaders['train']
    val_loader = fusion_loaders['val']
    
    # 优化器 - 使用更细致的学习率
    optimizer = optim.AdamW(
        enhanced_model.parameters(),
        lr=0.0001,  # 适中的学习率
        weight_decay=0.01,
        eps=1e-8
    )
    
    # 学习率调度器 - 温和的余弦退火
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # 改进的损失函数 - 标签平滑
    class LabelSmoothingLoss(nn.Module):
        def __init__(self, num_classes=2, smoothing=0.1):
            super(LabelSmoothingLoss, self).__init__()
            self.num_classes = num_classes
            self.smoothing = smoothing
            
        def forward(self, inputs, targets):
            log_probs = F.log_softmax(inputs, dim=1)
            targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
            targets_smooth = (1 - self.smoothing) * targets_one_hot + self.smoothing / self.num_classes
            loss = -torch.sum(targets_smooth * log_probs, dim=1).mean()
            return loss
    
    criterion = LabelSmoothingLoss(smoothing=0.1)
    
    best_val_acc = 0.0
    enhanced_model.train()
    
    print(f"\n🚀 开始深度架构微调训练...")
    
    for epoch in range(epochs):
        # 训练阶段
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"架构微调 {epoch+1}/{epochs}")
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = enhanced_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(enhanced_model.parameters(), max_norm=1.0)
            
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
        enhanced_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        class_correct = [0, 0]
        class_total = [0, 0]
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = enhanced_model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # 类别准确率统计
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1
        
        val_acc = 100. * val_correct / val_total
        ad_acc = 100. * class_correct[0] / class_total[0] if class_total[0] > 0 else 0
        cn_acc = 100. * class_correct[1] / class_total[1] if class_total[1] > 0 else 0
        
        # 更新最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(enhanced_model.state_dict(), './models/enhanced_architecture_model.pth')
            print(f"\n✅ 保存增强架构模型，验证准确率: {val_acc:.2f}%")
        
        scheduler.step()
        enhanced_model.train()
        
        print(f"\n轮次 [{epoch+1}/{epochs}] - 深度架构微调:")
        print(f"   训练准确率: {100.*train_correct/train_total:.2f}%")
        print(f"   验证准确率: {val_acc:.2f}%")
        print(f"   类别准确率: AD={ad_acc:.2f}%, CN={cn_acc:.2f}%")
        print(f"   学习率: {optimizer.param_groups[0]['lr']:.6f}")
    
    print(f"\n🎉 深度架构微调完成！")
    print(f"🏆 最佳验证准确率: {best_val_acc:.2f}%")
    print(f"📁 增强模型保存: ./models/enhanced_architecture_model.pth")
    
    return best_val_acc


if __name__ == "__main__":
    print("深度架构微调脚本就绪！") 
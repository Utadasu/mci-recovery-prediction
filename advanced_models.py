import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

# 深度可分离卷积实现
class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size, 
                                  stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# 渐进式特征丢弃
class ProgressiveDropout(nn.Module):
    def __init__(self, start_p=0.1, end_p=0.5, max_epochs=50, warmup_epochs=5):
        super(ProgressiveDropout, self).__init__()
        self.start_p = start_p
        self.end_p = end_p
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        
    def forward(self, x):
        if self.training:
            # 计算当前应用的dropout率
            if self.current_epoch < self.warmup_epochs:
                # 预热阶段，线性从0增加到start_p
                p = (self.start_p * self.current_epoch) / self.warmup_epochs
            else:
                # 主要训练阶段，从start_p逐渐提高到end_p
                progress = min(1.0, (self.current_epoch - self.warmup_epochs) / 
                              (self.max_epochs - self.warmup_epochs))
                p = self.start_p + (self.end_p - self.start_p) * progress
            
            # 应用平滑过渡的dropout
            mask = torch.bernoulli(torch.ones_like(x) * (1 - p)) / (1 - p)
            return x * mask
        return x
        
    def update_epoch(self, epoch):
        """更新当前轮次"""
        self.current_epoch = epoch
        
    def get_current_dropout_rate(self):
        """返回当前dropout率，用于监控"""
        if self.current_epoch < self.warmup_epochs:
            return (self.start_p * self.current_epoch) / self.warmup_epochs
        else:
            progress = min(1.0, (self.current_epoch - self.warmup_epochs) / 
                          (self.max_epochs - self.warmup_epochs))
            return self.start_p + (self.end_p - self.start_p) * progress

# 高级正则化工具
class MixUp(nn.Module):
    """MixUp数据增强"""
    def __init__(self, alpha=0.2):
        super(MixUp, self).__init__()
        self.alpha = alpha
        
    def forward(self, x, target=None, training=True):
        if not training or self.alpha <= 0:
            return x, target
            
        batch_size = x.size(0)
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 随机置换索引
        index = torch.randperm(batch_size).to(x.device)
        
        # 混合数据
        mixed_x = lam * x + (1 - lam) * x[index, :]
        
        # 混合标签
        if target is not None:
            mixed_target = (lam * F.one_hot(target, num_classes=2) + 
                           (1 - lam) * F.one_hot(target[index], num_classes=2))
            return mixed_x, mixed_target
            
        return mixed_x, None

# CBAM (Convolutional Block Attention Module)
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # 共享MLP
        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction_ratio, channels, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 5, 7), "内核大小必须是 3, 5 或 7"
        padding = kernel_size // 2
        
        # 轻量级空间注意力
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(1)  # 添加BN提高稳定性
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 沿通道维度计算平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 连接特征
        out = torch.cat([avg_out, max_out], dim=1)
        
        # 空间注意力映射
        out = self.conv(out)
        out = self.bn(out)
        
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):  # 增加reduction ratio降低参数量
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=3)  # 降低内核大小
        
        # 简化融合机制
        self.channel_weight = 0.7  # 固定权重替代学习参数
        self.spatial_weight = 0.3
        
        # 移除额外的特征增强网络来减少内存使用
        
    def forward(self, x):
        # 保存输入作为残差连接
        identity = x
        
        # 获取并应用通道注意力
        channel_activation = self.channel_att(x)
        channel_refined = x * channel_activation
        
        # 获取并应用空间注意力
        spatial_activation = self.spatial_att(x)
        spatial_refined = x * spatial_activation
        
        # 使用固定权重融合
        fused = self.channel_weight * channel_refined + self.spatial_weight * spatial_refined
        
        # 残差连接
        out = fused + identity
        
        return out

# 密集连接块
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=16, num_layers=4):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.layers.append(self._make_dense_layer(in_channels + i * growth_rate, growth_rate))
    
    def _make_dense_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout3d(0.2)
        )
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)

# 过渡层 - 用于降低特征图尺寸和通道数
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.Dropout3d(0.2),
            nn.AvgPool3d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.transition(x)

# 主要的高级模型 - DenseNet3D + CBAM
class DenseNet3D(nn.Module):
    def __init__(self, in_channels=1, growth_rate=12, block_config=(6, 12, 24, 16), 
                 num_init_features=32, num_classes=2, drop_rate=0.3):
        super(DenseNet3D, self).__init__()
        
        # 初始卷积层
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        # 密集块和过渡层
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # 添加密集块
            block = DenseBlock(
                num_features,
                growth_rate=growth_rate,
                num_layers=num_layers
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features += num_layers * growth_rate
            
            # 添加CBAM注意力
            cbam = CBAM(num_features)
            self.features.add_module(f'cbam{i+1}', cbam)
            
            # 添加过渡层（除了最后一个块）
            if i != len(block_config) - 1:
                transition = TransitionLayer(
                    num_features,
                    num_features // 2
                )
                self.features.add_module(f'transition{i+1}', transition)
                num_features = num_features // 2
        
        # 批归一化和池化
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.features.add_module('avg_pool', nn.AdaptiveAvgPool3d((1, 1, 1)))  # 自适应池化，适用于任何输入尺寸
        
        # 分类器 - 输入特征数仅取决于通道数，与空间尺寸无关
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(256, num_classes)
        )
        
        # 打印网络结构信息
        print(f"DenseNet3D initialized with input size: 113×137×113")
        print(f"Initial features: {num_init_features}")
        print(f"Growth rate: {growth_rate}")
        print(f"Block config: {block_config}")
        print(f"Final feature channels: {num_features}")
        
        # MixUp正则化
        self.mixup = MixUp(alpha=0.2)
        
        # 权重初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x, target=None, apply_mixup=False):
        # 可选应用MixUp
        if apply_mixup and target is not None:
            x, target = self.mixup(x, target, training=self.training)
        
        # 特征提取
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        
        if apply_mixup and target is not None:
            return out, target
        return out

# 添加通道注意力的残差块
class ResidualCBAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualCBAMBlock, self).__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                    stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1)  # 添加轻量dropout以提高鲁棒性
        )
        
        # 第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3,
                    stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels)
        )
        
        # 添加改进的CBAM注意力机制 - 减小内部通道以提高效率
        self.cbam = CBAM(out_channels, reduction_ratio=8)
        
        # Shortcut连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        # 主路径
        out = self.conv1(x)
        out = self.conv2(out)
        
        # 应用CBAM注意力
        out = self.cbam(out)
        
        # 残差连接
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out

# 结合CBAM的ResNet3D
class ResNetCBAM3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, base_channels=8):  # 降低基础通道数
        super(ResNetCBAM3D, self).__init__()
        
        # 初始卷积层 - 减小通道数
        self.conv1 = nn.Conv3d(in_channels, base_channels, kernel_size=5, 
                              stride=2, padding=2, bias=False)  # 增加stride降低分辨率
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # 减少残差块通道数和数量
        self.layer1 = self._make_layer(base_channels, base_channels*2, 1, stride=1)  # 只用1个block
        self.layer2 = self._make_layer(base_channels*2, base_channels*4, 1, stride=2)
        self.layer3 = self._make_layer(base_channels*4, base_channels*8, 1, stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # 减小分类器规模
        self.fc = nn.Sequential(
            nn.Linear(base_channels*8, 128),  # 减小全连接层尺寸
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        
        # 打印网络结构信息
        print(f"ResNetCBAM3D initialized with input size: 113×137×113")
        print(f"Base channels: {base_channels}")
        print(f"Final feature channels: {base_channels*8}")
        
        # 初始化权重
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        # 第一个块可能需要下采样
        layers.append(ResidualCBAMBlock(in_channels, out_channels, stride))
        
        # 添加剩余块
        for _ in range(1, blocks):
            layers.append(ResidualCBAMBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """增强的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # 使用MSRA/He初始化，为ReLU激活函数优化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                # 标准初始化BatchNorm
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 对于线性层使用截断正态分布初始化
                nn.init.trunc_normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 特征提取主干
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # 全局池化
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # 分类器
        x = self.fc(x)
        
        return x

# 注意力引导的动态融合模型
class DynamicFusionModel(nn.Module):
    def __init__(self, num_models=3, num_classes=2, hidden_dim=64):
        super(DynamicFusionModel, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # 移除对固定num_models的依赖
        
        # Cross-attention layers for each model output
        self.query_proj = nn.Linear(num_classes, hidden_dim)
        self.key_proj = nn.Linear(num_classes, hidden_dim)
        self.value_proj = nn.Linear(num_classes, hidden_dim)
        
        # Multi-head attention for cross-model interactions
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Dynamic feature network - 使用卷积网络处理可变长度的输入
        self.feature_network = nn.Sequential(
            nn.Linear(num_classes, hidden_dim * 2),  # 每个模型输出单独处理
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Dynamic weighting network - 使用AdaptiveAvgPool1d适应任意数量的模型
        self.weight_network = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),  # 为每个模型输出一个权重
            nn.Softmax(dim=1)  # 在模型维度上进行softmax
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Uncertainty estimation branch
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),  # 为每个模型输出一个不确定性值
            nn.Softplus()  # 确保正值
        )
    
    def forward(self, model_outputs):
        # model_outputs should be a list of tensors, each of shape [batch_size, num_classes]
        batch_size = model_outputs[0].size(0)
        actual_num_models = len(model_outputs)
        
        # Stack model outputs for easier processing
        stacked_outputs = torch.stack(model_outputs, dim=1)  # [batch_size, num_models, num_classes]
        
        # Apply cross-attention mechanism
        queries = self.query_proj(stacked_outputs)  # [batch_size, num_models, hidden_dim]
        keys = self.key_proj(stacked_outputs)       # [batch_size, num_models, hidden_dim]
        values = self.value_proj(stacked_outputs)   # [batch_size, num_models, hidden_dim]
        
        # Cross-attention between different model outputs
        attended_features, _ = self.cross_attention(queries, keys, values)
        # attended_features: [batch_size, num_models, hidden_dim]
        
        # Average along model dimension to get a single feature vector per sample
        global_features = torch.mean(attended_features, dim=1)  # [batch_size, hidden_dim]
        
        # Process each model output separately and combine
        processed_features = []
        for i in range(actual_num_models):
            model_output = model_outputs[i]  # [batch_size, num_classes]
            processed = self.feature_network(model_output)  # [batch_size, hidden_dim]
            processed_features.append(processed)
        
        # Stack processed features
        stacked_processed = torch.stack(processed_features, dim=1)  # [batch_size, num_models, hidden_dim]
        
        # Combine global features with processed features - 取平均
        combined_features = global_features + torch.mean(stacked_processed, dim=1)  # [batch_size, hidden_dim]
        
        # Estimate uncertainty for each model
        uncertainty_weights = self.uncertainty_estimator(combined_features).repeat(1, actual_num_models)  # [batch_size, num_models]
        confidence_weights = 1.0 / (uncertainty_weights + 1e-8)  # 转换为置信度
        normalized_weights = F.softmax(confidence_weights, dim=1)  # [batch_size, num_models]
        
        # Generate dynamic weights based on features
        dynamic_weights_input = combined_features.unsqueeze(1).repeat(1, actual_num_models, 1)  # [batch_size, num_models, hidden_dim]
        dynamic_weights = self.weight_network(dynamic_weights_input.view(-1, self.hidden_dim)).view(batch_size, actual_num_models)  # [batch_size, num_models]
        
        # Combine both weights
        final_weights = (dynamic_weights + normalized_weights) / 2
        final_weights = final_weights.unsqueeze(2)  # [batch_size, num_models, 1]
        
        # Apply the weights to model outputs
        weighted_outputs = stacked_outputs * final_weights  # [batch_size, num_models, num_classes]
        fused_output = weighted_outputs.sum(dim=1)  # [batch_size, num_classes]
        
        # Additional refinement through classifier
        final_output = self.classifier(combined_features)
        
        # Residual connection with the fused output
        result = final_output + fused_output
        
        return result

# 不确定性加权融合
class UncertaintyWeightedFusion(nn.Module):
    def __init__(self, num_models=3, num_classes=2, hidden_dim=64):
        super(UncertaintyWeightedFusion, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Temperature scaling parameters (learnable)
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Model-specific calibration networks - 使用ModuleList动态处理
        self.calibration_network = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Uncertainty estimation network
        self.uncertainty_network = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensures positive uncertainties
        )
        
        # Meta-learning network to determine fusion strategy
        self.meta_network = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2),  # 2 values: alpha for uncertainty weight and beta for voting weight
            nn.Sigmoid()  # Constrains to [0, 1]
        )
        
        # Additional processing for final output
        self.output_refiner = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, model_outputs):
        # model_outputs: list of tensor predictions from each model [batch_size, num_classes]
        batch_size = model_outputs[0].size(0)
        actual_num_models = len(model_outputs)
        
        # Stack outputs for easier processing
        stacked_outputs = torch.stack(model_outputs, dim=1)  # [batch_size, num_models, num_classes]
        
        # Apply temperature scaling to each model output
        scaled_outputs = []
        for i in range(actual_num_models):
            # Apply temperature scaling
            temp = torch.clamp(self.temperature, min=0.1, max=10.0)  # 使用单个温度参数
            scaled = model_outputs[i] / temp
            
            # Apply calibration network
            calibrated = self.calibration_network(scaled)
            
            # Apply softmax to get proper probabilities
            probs = F.softmax(calibrated, dim=1)
            scaled_outputs.append(probs)
        
        # Stack calibrated outputs
        calibrated_stack = torch.stack(scaled_outputs, dim=1)  # [batch_size, num_models, num_classes]
        
        # Calculate prediction entropy as a measure of uncertainty
        entropy = -torch.sum(calibrated_stack * torch.log(calibrated_stack + 1e-8), dim=2)  # [batch_size, num_models]
        
        # Estimate uncertainties for each model output
        model_uncertainties = []
        for i in range(actual_num_models):
            uncertainty = self.uncertainty_network(model_outputs[i])  # [batch_size, 1]
            model_uncertainties.append(uncertainty)
        estimated_uncertainties = torch.cat(model_uncertainties, dim=1)  # [batch_size, num_models]
        
        # Combine network-estimated and entropy-based uncertainties
        combined_uncertainty = 0.5 * estimated_uncertainties + 0.5 * entropy
        
        # Convert uncertainties to weights (higher uncertainty -> lower weight)
        uncertainty_weights = 1.0 / (combined_uncertainty + 1e-8)
        
        # Normalize weights
        uncertainty_weights = F.softmax(uncertainty_weights, dim=1)  # [batch_size, num_models]
        
        # Compute voting weights based on maximum probability
        max_probs, _ = torch.max(calibrated_stack, dim=2)  # [batch_size, num_models]
        voting_weights = F.softmax(max_probs, dim=1)  # [batch_size, num_models]
        
        # Meta-learning to determine optimal fusion strategy
        # 使用平均logits处理动态数量的模型
        avg_output = torch.mean(torch.stack(model_outputs), dim=0)  # [batch_size, num_classes]
        fusion_params = self.meta_network(avg_output)  # [batch_size, 2]
        alpha = fusion_params[:, 0].unsqueeze(1)  # Uncertainty weight importance
        beta = fusion_params[:, 1].unsqueeze(1)   # Voting weight importance
        
        # Adaptive fusion weights
        fusion_weights = (alpha * uncertainty_weights + beta * voting_weights) / (alpha + beta + 1e-8)
        fusion_weights = fusion_weights.unsqueeze(2)  # [batch_size, num_models, 1]
        
        # Apply fusion weights to calibrated outputs
        weighted_outputs = calibrated_stack * fusion_weights  # [batch_size, num_models, num_classes]
        fused_output = weighted_outputs.sum(dim=1)  # [batch_size, num_classes]
        
        # Final refinement
        refined_output = self.output_refiner(fused_output) + fused_output  # Residual connection
        
        return refined_output, uncertainty_weights.detach()  # Return weights for analysis

# 自适应融合模型 - 用于融合多个模型的输出 (旧版，保留为兼容)
class AdaptiveFusionModel(nn.Module):
    def __init__(self, num_models=3, num_classes=2):
        super(AdaptiveFusionModel, self).__init__()
        self.num_models = num_models
        self.num_classes = num_classes
        
        # 灵活的融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(num_models * num_classes, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, model_outputs):
        # 连接所有模型的输出
        combined = torch.cat(model_outputs, dim=1)
        
        # 应用融合网络
        return self.fusion_net(combined)

class SimpleCNN3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(SimpleCNN3D, self).__init__()
        
        # 计算输入大小：原始大小 113×137×113
        # 经过3次最大池化(stride=2): 113/8 × 137/8 × 113/8 ≈ 14×17×14
        
        # 减少卷积层数量和通道数
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            ChannelAttentionBlock(16),  # 添加通道注意力
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout3d(0.3),
            
            # 第二个卷积块
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            ChannelAttentionBlock(32),  # 添加通道注意力
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout3d(0.3),
            
            # 第三个卷积块
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            ChannelAttentionBlock(64),  # 添加通道注意力
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout3d(0.3),
        )
        
        # 计算全连接层的输入特征数 (根据原始大小 113×137×113)
        # 113/8=14.125 向下取整为14, 137/8=17.125 向下取整为17, 113/8=14.125 向下取整为14
        # 因此特征图大小为 14x17x14 
        self._to_linear = 64 * 14 * 17 * 14
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        # 特征提取
        x = self.features(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 分类
        x = self.classifier(x)
        
        return x 

# Advanced 3D Attention Mechanism with Multi-Scale Features
class Advanced3DAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(Advanced3DAttention, self).__init__()
        self.channels = channels
        
        # Multi-scale spatial pooling
        self.spatial_pools = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool3d(kernel_size=scale, stride=1, padding=scale//2),
                nn.Conv3d(channels, channels//reduction_ratio, kernel_size=1),
                nn.ReLU(inplace=True)
            ) for scale in [3, 5, 7]
        ])
        
        # Spatial attention projection
        self.spatial_projection = nn.Conv3d(3*channels//reduction_ratio, 1, kernel_size=1)
        
        # Channel attention with split processing for improved efficiency
        self.channel_pool_avg = nn.AdaptiveAvgPool3d(1)
        self.channel_pool_max = nn.AdaptiveMaxPool3d(1)
        
        # Shared MLP for channel attention
        self.channel_mlp = nn.Sequential(
            nn.Conv3d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction_ratio, channels, kernel_size=1, bias=False)
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1),
            nn.LayerNorm([channels, 1, 1, 1]),
            nn.Sigmoid()
        )
        
        # 1x1 conv for final refinement
        self.refine = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        batch_size, channels, d, h, w = x.size()
        
        # Save input for residual connection
        identity = x
        
        # Multi-scale spatial attention
        spatial_features = []
        for pool in self.spatial_pools:
            spatial_features.append(pool(x))
        
        # Concatenate multi-scale features
        spatial_concat = torch.cat(spatial_features, dim=1)
        spatial_attention = self.spatial_projection(spatial_concat)
        spatial_attention = torch.sigmoid(spatial_attention)
        
        # Apply spatial attention
        spatial_output = x * spatial_attention
        
        # Channel attention - split processing
        avg_pool = self.channel_pool_avg(x)
        max_pool = self.channel_pool_max(x)
        
        avg_channel_attention = self.channel_mlp(avg_pool)
        max_channel_attention = self.channel_mlp(max_pool)
        
        # Combine channel attention with dynamic fusion
        channel_attention = torch.sigmoid(avg_channel_attention + max_channel_attention)
        channel_output = x * channel_attention
        
        # Gating mechanism to balance spatial and channel attention
        concat_features = torch.cat([spatial_output, channel_output], dim=1)
        gate_weights = self.gate(concat_features)
        
        # Apply gating to blend spatial and channel outputs
        blended = gate_weights * spatial_output + (1 - gate_weights) * channel_output
        
        # Final refinement
        refined = self.refine(blended)
        
        # Add residual connection
        output = refined + identity
        
        return output

# Enhanced Residual CBAM Block with Advanced 3D Attention
class EnhancedResidualCBAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_depthwise=True):
        super(EnhancedResidualCBAMBlock, self).__init__()
        
        # Determine if we'll use depthwise separable convolution
        if use_depthwise:
            self.conv1 = DepthwiseSeparableConv3d(in_channels, out_channels, 
                                                kernel_size=3, stride=stride, padding=1)
            self.conv2 = DepthwiseSeparableConv3d(out_channels, out_channels, 
                                                kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        # Advanced 3D attention mechanism
        self.attention = Advanced3DAttention(out_channels)
        
        # Residual connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        # Final activation
        self.relu = nn.ReLU(inplace=True)
        
        # Progressive dropout
        self.dropout = ProgressiveDropout(start_p=0.1, end_p=0.3, max_epochs=50)
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        # Apply advanced attention
        out = self.attention(out)
        
        # Apply dropout
        out = self.dropout(out)
        
        # Add residual connection
        out += residual
        out = self.relu(out)
        
        return out 

# Enhanced Ensemble Model with Cross-Tissue Knowledge Transfer
class EnhancedEnsembleModel(nn.Module):
    def __init__(self, base_model_class, in_channels=1, num_classes=2, tissue_types=None):
        super(EnhancedEnsembleModel, self).__init__()
        self.tissue_types = tissue_types or ['CSF', 'GREY', 'WHITE']
        
        # 为每个组织类型创建专用模型
        self.tissue_models = nn.ModuleDict({
            tissue: base_model_class(in_channels=in_channels, num_classes=num_classes)
            for tissue in self.tissue_types
        })
        
        # 可学习的温度缩放参数
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # 降低初始温度值，减少软化程度
        
        # 改进的融合权重机制 - 使用组织特异性权重
        self.tissue_importance = nn.Parameter(torch.ones(len(self.tissue_types)))
        
        # 改进跨组织注意力机制的隐藏维度
        hidden_dim = 128  # 增加隐藏维度
        
        # 改进的跨组织注意力机制
        self.cross_tissue_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=4,  # 增加头数以捕获更复杂的跨组织关系
            batch_first=True,
            dropout=0.2  # 添加dropout减少过拟合
        )
        
        # 特征变换层
        self.feature_transform = nn.ModuleDict({
            tissue: nn.Sequential(
                nn.Linear(num_classes, hidden_dim),
                nn.LayerNorm(hidden_dim),  # 添加层归一化
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)  # 减轻过拟合
            ) for tissue in self.tissue_types
        })
        
        # 改进的特征细化层
        self.feature_refinement = nn.ModuleDict({
            tissue: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # 添加层归一化
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),  # 减轻过拟合
                nn.Linear(hidden_dim, num_classes)
            ) for tissue in self.tissue_types
        })
        
        # 改进的融合模型 - 使用更复杂的架构
        fusion_in_dim = len(self.tissue_types) * num_classes
        self.fusion_model = nn.Sequential(
            nn.Linear(fusion_in_dim, 64),
            nn.LayerNorm(64),  # 添加层归一化
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # 添加更强的dropout
            nn.Linear(64, 32),
            nn.LayerNorm(32),  # 添加层归一化
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )
        
        # 校准层
        self.calibration = nn.Softmax(dim=1)
    
    def forward(self, x_dict, apply_fusion=True):
        """
        前向传播函数
        """
        individual_outputs = {}
        transformed_features = {}
        original_features = {}  # 存储原始特征
        
        # 处理每个组织特定模型
        for tissue, model in self.tissue_models.items():
            if tissue in x_dict:
                # 获取原始模型输出
                with torch.cuda.amp.autocast(enabled=True):  # 使用混合精度降低内存
                    output = model(x_dict[tissue])
                
                # 保存原始输出
                individual_outputs[tissue] = output
                original_features[tissue] = output  # 保存原始特征用于残差连接
                
                # 应用温度缩放获得更软的概率
                temp = torch.clamp(self.temperature, min=1.0, max=5.0)  # 限制温度范围
                soft_output = output / temp
                
                # 变换特征用于跨注意力
                transformed_features[tissue] = self.feature_transform[tissue](soft_output)
        
        # 如果有多个组织，应用改进的跨组织注意力
        if len(transformed_features) > 1:
            # 堆叠特征用于注意力计算
            available_tissues = [t for t in self.tissue_types if t in transformed_features]
            feature_stack = torch.stack([transformed_features[t] for t in available_tissues], dim=1)
            
            # 应用自注意力机制
            attended_features, attention_weights = self.cross_tissue_attention(
                feature_stack, feature_stack, feature_stack
            )
            
            # 更新每个组织的特征，使用残差连接
            for i, tissue in enumerate(available_tissues):
                refined = self.feature_refinement[tissue](attended_features[:, i])
                # 残差连接 - 添加到原始输出
                individual_outputs[tissue] = original_features[tissue] + refined
        
        # 如果不需要融合或只有一个组织，返回单独的输出
        if not apply_fusion or len(individual_outputs) < 2:
            return individual_outputs
        
        # 改进的融合策略 - 动态加权并结合注意力
        tissue_weights = F.softmax(self.tissue_importance, dim=0)
        
        # 准备融合输入 - 连接所有组织输出
        fusion_inputs = []
        for i, tissue in enumerate(self.tissue_types):
            if tissue in individual_outputs:
                # 应用组织特定权重
                fusion_inputs.append(individual_outputs[tissue])
        
        if fusion_inputs:
            # 连接所有输出用于融合
            concat_outputs = torch.cat(fusion_inputs, dim=1)
            
            # 使用改进的融合模型
            fused_output = self.fusion_model(concat_outputs)
            
            # 添加残差连接 - 使用权重平均作为残差
            weighted_avg = sum(individual_outputs[t] * tissue_weights[i] 
                              for i, t in enumerate(self.tissue_types) if t in individual_outputs)
            
            # 最终输出结合残差和融合结果
            final_output = fused_output + 0.2 * weighted_avg  # 残差权重为0.2
            individual_outputs['fused'] = self.calibration(final_output)
        
        return individual_outputs

# Enhanced Ensemble Trainer Function
def create_ensemble_model(device, base_model_class=ResNetCBAM3D, in_channels=1, num_classes=2,
                       tissue_types=None):
    """创建增强型集成模型，包含所有优化项
    
    参数:
        device: 设备对象（cuda或cpu）
        base_model_class: 基础模型类，默认为ResNetCBAM3D
        in_channels: 输入通道数
        num_classes: 类别数
        tissue_types: 组织类型列表
    
    返回:
        model: 创建的模型
    """
    # 创建模型
    model = EnhancedEnsembleModel(
        base_model_class=base_model_class,
        in_channels=in_channels,
        num_classes=num_classes,
        tissue_types=tissue_types
    )
    return model.to(device)

# 修复line 702附近的类，确保ChannelAttentionBlock存在
class ChannelAttentionBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # Shared MLP for channel attention
        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction_ratio, channels, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out) 
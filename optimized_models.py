import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# 稳定的BatchNorm实现
class StableBatchNorm3d(nn.BatchNorm3d):
    def __init__(self, num_features, eps=1e-5, momentum=0.05):
        super(StableBatchNorm3d, self).__init__(
            num_features, eps=eps, momentum=momentum
        )
    
    def forward(self, input):
        self._check_input_dim(input)
        
        # 应用更稳定的计算
        if self.training:
            # 计算均值和方差时增加eps，避免数值不稳定
            mean = input.mean(dim=[0, 2, 3, 4], keepdim=True)
            var = input.var(dim=[0, 2, 3, 4], unbiased=False, keepdim=True) + self.eps
            
            # 使用累积移动平均
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
            
            # 归一化
            normalized = (input - mean) / torch.sqrt(var)
            return self.weight.view(1, -1, 1, 1, 1) * normalized + self.bias.view(1, -1, 1, 1, 1)
        else:
            # 测试阶段，使用累积统计量
            mean = self.running_mean.view(1, -1, 1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1, 1) + self.eps
            normalized = (input - mean) / torch.sqrt(var)
            return self.weight.view(1, -1, 1, 1, 1) * normalized + self.bias.view(1, -1, 1, 1, 1)



# 优化的ResNetCBAM3D模型 - 真正的CBAM实现
class ImprovedResNetCBAM3D(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, base_channels=12, dropout_rate=0.3, use_global_pool=True, use_cbam=True):
        super(ImprovedResNetCBAM3D, self).__init__()
        
        # 保存输入通道数作为类属性
        self.in_channels = in_channels
        self.use_global_pool = use_global_pool
        self.use_cbam = use_cbam
        print(f"   模型配置: 使用 CBAM -> {'✅' if self.use_cbam else '❌'}")
        
        # 改进的初始层设计 - 更适合3D MRI数据
        self.stem = nn.Sequential(
            # 第一步：大核卷积捕获更多上下文信息
            nn.Conv3d(in_channels, base_channels//2, kernel_size=7, stride=1, padding=3, bias=False),
            StableBatchNorm3d(base_channels//2),
            nn.ReLU(inplace=False),
            
            # 第二步：细化特征并开始下采样
            nn.Conv3d(base_channels//2, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            StableBatchNorm3d(base_channels),
            nn.ReLU(inplace=False),
            
            # 第三步：抗混叠下采样
            AntiAliasDownsample3D(base_channels, stride=2, kernel_size=3)
        )
        
        # 使用真正CBAM注意力的残差层
        self.layer1 = self._make_layer(base_channels, base_channels*2, 3, stride=1, 
                                     dropout_rate=dropout_rate, stochastic_depth_prob=0.0)
        self.layer2 = self._make_layer(base_channels*2, base_channels*4, 4, stride=2, 
                                     dropout_rate=dropout_rate, stochastic_depth_prob=0.1)
        self.layer3 = self._make_layer(base_channels*4, base_channels*8, 3, stride=2, 
                                     dropout_rate=dropout_rate, stochastic_depth_prob=0.2)
        
        # 新增：额外的下采样层 (layer4) - 逐渐增加随机深度概率
        self.layer4 = self._make_layer(base_channels*8, base_channels*16, 2, stride=2, 
                                     dropout_rate=dropout_rate, stochastic_depth_prob=0.3)
        
        if use_global_pool:
            # 原始方案：全局平均池化
            self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            final_feature_dim = base_channels*16
        else:
            # 新方案：Layer5智能下采样层
            self.layer5 = self._make_advanced_downsample_layer(
                base_channels*16, base_channels*16, 
                target_size=(2, 2, 2),  # 目标空间尺寸 [2,2,2]
                dropout_rate=dropout_rate
            )
            # 最终特征维度 = 通道数 × 空间尺寸
            final_feature_dim = base_channels*16 * 2 * 2 * 2  # 192 * 8 = 1536维
        
        # 特征融合和分类 - 更鲁棒的设计，避免batch_size=1时BatchNorm问题
        self.fusion = nn.Sequential(
            nn.Linear(final_feature_dim, 512),
            nn.LayerNorm(512),  # 使用LayerNorm替代BatchNorm1d，避免batch_size=1问题
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),  # 使用LayerNorm替代BatchNorm1d
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_rate)
        )
        
        # 分类头
        self.classifier = nn.Linear(256, num_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride, dropout_rate, stochastic_depth_prob):
        layers = []
        
        # 首个块处理通道和尺寸变化
        layers.append(EnhancedResidualBlock(
            in_channels, out_channels, stride, dropout_rate, 
            stochastic_depth_prob=stochastic_depth_prob,
            use_cbam=self.use_cbam
        ))
        
        # 后续块保持通道和尺寸，逐渐增加随机深度概率
        for i in range(1, blocks):
            # 在同一层内逐渐增加随机深度概率
            block_stochastic_prob = stochastic_depth_prob * (1 + i * 0.1)
            layers.append(EnhancedResidualBlock(
                out_channels, out_channels, 1, dropout_rate,
                stochastic_depth_prob=block_stochastic_prob,
                use_cbam=self.use_cbam
            ))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # 使用更好的初始化方法
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, StableBatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 使用截断正态分布初始化线性层
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features=False):
        """
        前向传播
        Args:
            x: 输入特征 [B, C, D, H, W]
        """
        # 输入形状检查和调整
        if len(x.shape) == 6:  # 处理形状为 [B, C, 1, D, H, W] 的情况
            print(f"检测到输入形状有额外维度: {x.shape}，自动调整为5D张量")
            x = x.squeeze(2)  # 去除多余的维度，变为 [B, C, D, H, W]
            
        if len(x.shape) != 5:
            raise ValueError(f"期望输入形状为[B, C, D, H, W]，但得到: {x.shape}")
            
        # 检查通道数是否匹配
        if x.shape[1] != self.in_channels:
            print(f"⚠️ 警告: 输入通道数({x.shape[1]})与模型期望通道数({self.in_channels})不匹配")
            # 如果通道数多余，截取前in_channels个通道
            if x.shape[1] > self.in_channels:
                print(f"截取前{self.in_channels}个通道")
                x = x[:, :self.in_channels, ...]
            # 如果通道数不足，使用复制扩展通道数
            else:
                repeat_times = math.ceil(self.in_channels / x.shape[1])
                print(f"通道不足，复制{repeat_times}次扩展通道")
                x = x.repeat(1, repeat_times, 1, 1, 1)[:, :self.in_channels, ...]
                
        # 输入归一化 - 使用更稳定的归一化方式
        mean = x.mean(dim=[2, 3, 4], keepdim=True)
        std = x.std(dim=[2, 3, 4], keepdim=True) + 1e-6
        x = (x - mean) / std
        
        # 特征提取 - 改进的stem
        x = self.stem(x)
        
        # 残差层 - 使用真正的CBAM注意力
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # 新增：额外的下采样层
        features = self.layer4(x)
        
        if self.use_global_pool:
            # 全局池化 - 替代特征金字塔池化
            x = self.global_pool(features)
            x = x.view(x.size(0), -1)  # 将特征压平
        else:
            # 新方案：Layer5智能下采样层
            x = self.layer5(features)
            # 智能下采样后需要flatten：[B, C, 2, 2, 2] → [B, C*2*2*2]
            x = x.view(x.size(0), -1)  # 将特征压平
        
        # 特征融合
        x = self.fusion(x)
        
        if return_features:
            return x
        
        # 分类
        logits = self.classifier(x)
        
        return logits

    def _make_advanced_downsample_layer(self, in_channels, out_channels, target_size=(2, 2, 2), dropout_rate=0.3):
        """
        🔥 智能下采样层 - 替代全局池化的高级方案
        
        特点:
        - 🎯 保留空间信息的同时进行尺寸压缩
        - 🔧 深度可分离卷积减少参数量
        - 💡 集成注意力机制增强重要特征
        - 📊 自适应池化，灵活控制输出尺寸
        - ⚖️ 平衡计算效率与特征表达能力
        """
        return AdvancedDownsampleLayer(in_channels, out_channels, target_size, dropout_rate)

# 增强的残差块 - 使用真正的CBAM和改进的随机深度
class EnhancedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.2, 
                 groups=4, stochastic_depth_prob=0.0, use_cbam=True):
        super(EnhancedResidualBlock, self).__init__()
        
        # 确保通道数可被分组数整除
        groups = min(groups, in_channels//2, out_channels//2)
        if in_channels % groups != 0 or out_channels % groups != 0:
            groups = 1
        
        # 主卷积路径
        if stride > 1:
            # 如果有下采样，使用抗混叠下采样
            self.conv1 = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, 
                         padding=1, groups=groups, bias=False),
                StableBatchNorm3d(out_channels),
                nn.ReLU(inplace=False),
                AntiAliasDownsample3D(out_channels, stride=stride, kernel_size=3)
            )
        else:
            # 普通卷积
            self.conv1 = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, 
                         padding=1, groups=groups, bias=False),
                StableBatchNorm3d(out_channels),
                nn.ReLU(inplace=False)
            )
        
        # 添加dropout
        self.dropout1 = nn.Dropout3d(dropout_rate)
        
        # 第二个卷积
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, 
                     groups=groups, bias=False),
            StableBatchNorm3d(out_channels)
        )
        
        # 真正的CBAM注意力机制 (可消融)
        if use_cbam:
            self.cbam = TrueCBAM3D(out_channels, reduction_ratio=8, spatial_kernel_size=7)
        else:
            self.cbam = nn.Identity()
        
        # Shortcut连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            if stride > 1:
                # shortcut也使用抗混叠下采样
                self.shortcut = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                    AntiAliasDownsample3D(out_channels, stride=stride, kernel_size=3),
                    StableBatchNorm3d(out_channels)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    StableBatchNorm3d(out_channels)
                )
        
        # 增强的随机深度
        self.stochastic_depth = StochasticDepth(drop_prob=stochastic_depth_prob, mode='batch')
        
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        # 主路径
        out = self.conv1(x)
        out = self.dropout1(out)
        out = self.conv2(out)
        
        # CBAM注意力
        out = self.cbam(out)
        
        # 使用增强的随机深度
        out = self.stochastic_depth(out, identity)
        out = self.relu(out)
        
        return out

# 创建改进的ResNetCBAM3D模型
def create_improved_resnet3d(in_channels=3, num_classes=2, device='cuda', base_channels=12, dropout_rate=0.3, use_cbam=True):
    """
    创建改进版的ResNetCBAM3D模型。
    
    ✨ 新版本改进特性:
    - 🎯 真正的CBAM注意力机制：通道注意力 + 空间注意力，更好地定位3D MRI病灶
    - 🔄 抗混叠下采样：使用高斯模糊 + 下采样，减少特征损失和混叠效应  
    - 📊 增强的随机深度：逐层递增的随机深度概率，提高正则化效果
    - 🏗️ 优化的网络初始层：7x7大核卷积 + 抗混叠下采样，更适合3D MRI数据
    - 🎚️ 更稳定的归一化：按空间维度归一化，避免批次间干扰
    - 🎲 分类器简化：移除多头设计，减少过拟合风险
    
    架构变化:
    - Layer4新增额外下采样层，特征维度: [B,96,15,18,15] → [B,192,8,9,8]
    - 移除特征金字塔池化，改为直接全局池化和特征压平
    - 特征维度流: 输入[B,3,113,137,113] → 输出[B,2]
    - 支持内存效率模式，通过减少base_channels降低内存占用
    
    Args:
        in_channels: 输入通道数
        num_classes: 分类类别数
        device: 运行设备
        base_channels: 基础通道数 (默认12, 内存效率模式推荐使用8)
        dropout_rate: Dropout比率
        use_cbam: 是否使用CBAM注意力模块
        
    Returns:
        torch.nn.Module: 改进的ResNetCBAM3D模型，具备真正的CBAM注意力机制
    """
    # 打印模型配置信息
    print(f"🚀 创建增强版ImprovedResNetCBAM3D模型")
    print(f"   基础通道数: {base_channels}, Dropout率: {dropout_rate}")
    print(f"   ✅ 真正CBAM注意力 ✅ 抗混叠下采样 ✅ 增强随机深度")
    print(f"   使用CBAM: {'✅' if use_cbam else '❌'}")
    
    # 检测是否启用内存高效模式
    if base_channels <= 8:
        print("   💾 已启用内存高效模式，模型参数将显著减少")
    
    model = ImprovedResNetCBAM3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,  # 可配置基础通道数
        dropout_rate=dropout_rate,    # 可配置Dropout率
        use_cbam=use_cbam             # 传递CBAM开关
    )
    return model.to(device)

# EMA模型 - 用于模型权重的指数移动平均，减少测试时的波动
class EMAModel(nn.Module):
    def __init__(self, model, decay=0.999):
        super(EMAModel, self).__init__()
        self.module = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # 初始化EMA参数
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

# 真正的CBAM注意力模块 - 包含通道注意力和空间注意力
class TrueCBAM3D(nn.Module):
    def __init__(self, channels, reduction_ratio=8, spatial_kernel_size=7):
        super(TrueCBAM3D, self).__init__()
        
        # 通道注意力模块
        self.channel_attention = ChannelAttention3D(channels, reduction_ratio)
        
        # 空间注意力模块  
        self.spatial_attention = SpatialAttention3D(spatial_kernel_size)
    
    def forward(self, x):
        # 先应用通道注意力
        x = self.channel_attention(x) * x
        
        # 再应用空间注意力
        x = self.spatial_attention(x) * x
        
        return x

class ChannelAttention3D(nn.Module):
    def __init__(self, channels, reduction_ratio=8):
        super(ChannelAttention3D, self).__init__()
        reduced_channels = max(channels // reduction_ratio, 4)
        
        # 全局平均池化和最大池化
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # 共享的MLP
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv3d(reduced_channels, channels, kernel_size=1, bias=False)
        )
        
    def forward(self, x):
        # 通过平均池化和最大池化获取全局信息
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        
        # 相加后通过sigmoid得到通道注意力权重
        channel_att = torch.sigmoid(avg_out + max_out)
        
        return channel_att

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 在通道维度上进行平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, D, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, D, H, W]
        
        # 拼接两个特征图
        spatial_input = torch.cat([avg_out, max_out], dim=1)  # [B, 2, D, H, W]
        
        # 通过卷积得到空间注意力权重
        spatial_att = self.spatial_conv(spatial_input)  # [B, 1, D, H, W]
        
        return spatial_att

# 抗混叠下采样模块
class AntiAliasDownsample3D(nn.Module):
    def __init__(self, channels, stride=2, kernel_size=3):
        super(AntiAliasDownsample3D, self).__init__()
        
        # 先进行轻微模糊，再下采样
        self.blur = nn.Sequential(
            nn.ReplicationPad3d(kernel_size//2),  # 使用复制填充替代反射填充
            nn.Conv3d(channels, channels, kernel_size=kernel_size, stride=1, 
                     padding=0, groups=channels, bias=False)
        )
        
        # 下采样
        self.downsample = nn.AvgPool3d(kernel_size=stride, stride=stride)
        
        # 初始化模糊核为高斯核
        self._init_blur_kernel(kernel_size)
        
    def _init_blur_kernel(self, kernel_size):
        # 创建3D高斯核
        sigma = 0.5
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        
        # 1D高斯
        gauss_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        
        # 3D高斯核 = 1D高斯的外积
        gauss_3d = gauss_1d.view(-1, 1, 1) * gauss_1d.view(1, -1, 1) * gauss_1d.view(1, 1, -1)
        gauss_3d = gauss_3d / gauss_3d.sum()
        
        # 为每个通道设置相同的高斯核
        for name, param in self.blur.named_parameters():
            if 'weight' in name:
                with torch.no_grad():
                    for i in range(param.shape[0]):  # 每个输出通道
                        param[i, 0] = gauss_3d
                        
    def forward(self, x):
        # 💡 核心修复: 检查输入尺寸，如果小于下采样核，则跳过
        kernel_size = self.downsample.kernel_size
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3  # 将int转换为元组

        if any(x.shape[i+2] < k for i, k in enumerate(kernel_size)):
            return x
            
        x = self.blur(x)
        x = self.downsample(x)
        return x

# 增强的随机深度实现
class StochasticDepth(nn.Module):
    def __init__(self, drop_prob=0.0, mode='batch'):
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob
        self.mode = mode  # 'batch' 或 'sample'
        
    def forward(self, x, residual):
        if not self.training or self.drop_prob == 0.0:
            return x + residual
            
        # 生成随机mask
        if self.mode == 'batch':
            # 整个batch使用相同的随机决策
            keep_prob = 1.0 - self.drop_prob
            if torch.rand(1).item() >= self.drop_prob:
                return x + residual
            else:
                return residual
        else:
            # 每个样本独立决策
            batch_size = x.shape[0]
            keep_prob = 1.0 - self.drop_prob
            random_tensor = keep_prob + torch.rand((batch_size, 1, 1, 1, 1), 
                                                 dtype=x.dtype, device=x.device)
            binary_mask = torch.floor(random_tensor)
            return (x / keep_prob) * binary_mask + residual

# 🔥 智能下采样层 - 替代全局池化的核心组件
class AdvancedDownsampleLayer(nn.Module):
    """
    智能下采样层 - 替代全局池化，保留更多空间信息
    
    核心优势:
    - 🎯 保留空间结构信息，避免全局池化的信息损失
    - 🔧 深度可分离卷积，参数效率高
    - 💡 集成注意力机制，突出重要特征
    - 📊 自适应池化，灵活控制输出尺寸
    - ⚖️ 平衡计算效率与特征表达能力
    """
    def __init__(self, in_channels, out_channels, target_size=(2, 2, 2), dropout_rate=0.3):
        super(AdvancedDownsampleLayer, self).__init__()
        
        self.target_size = target_size
        
        # 方案1: 深度可分离卷积下采样 (参数高效)
        self.depthwise_downsample = nn.Sequential(
            # 深度卷积：每个通道独立处理
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, 
                     padding=1, groups=in_channels, bias=False),
            StableBatchNorm3d(in_channels),
            nn.ReLU(inplace=False),
            
            # 点卷积：通道间信息交互
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            StableBatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
            nn.Dropout3d(dropout_rate)
        )
        
        # 方案2: 注意力引导的特征选择
        self.spatial_attention = SpatialSelectionAttention3D(out_channels)
        
        # 方案3: 自适应池化到目标尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool3d(target_size)
        
        # 可选：额外的特征增强
        self.feature_enhance = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, 
                     groups=out_channels//4 if out_channels >= 4 else 1, bias=False),
            StableBatchNorm3d(out_channels),
            nn.ReLU(inplace=False)
        )
        
        print(f"🔥 AdvancedDownsampleLayer: {in_channels}→{out_channels}通道，目标尺寸{target_size}")
    
    def forward(self, x):
        """
        前向传播：[B, C, 8, 9, 8] → [B, C, 2, 2, 2]
        
        Args:
            x: 输入特征 [B, in_channels, D, H, W]
            
        Returns:
            features: 下采样特征 [B, out_channels, target_D, target_H, target_W]
        """
        # 步骤1: 深度可分离卷积下采样
        x = self.depthwise_downsample(x)  # [B, C, 4, 5, 4] (大约减半)
        
        # 步骤2: 空间注意力增强重要区域
        x = self.spatial_attention(x)
        
        # 步骤3: 自适应池化到精确目标尺寸
        x = self.adaptive_pool(x)  # [B, C, 2, 2, 2]
        
        # 步骤4: 特征增强（可选）
        x = self.feature_enhance(x)
        
        # 返回时保持空间维度，由上级决定是否flatten
        return x

# 🎯 空间选择注意力模块
class SpatialSelectionAttention3D(nn.Module):
    """
    空间选择注意力 - 专门用于下采样过程中的重要区域选择
    
    功能:
    - 识别空间中最重要的区域
    - 在下采样过程中保留关键信息
    - 适用于医学图像的病灶定位
    """
    def __init__(self, channels, reduction_ratio=8):
        super(SpatialSelectionAttention3D, self).__init__()
        
        # 通道压缩
        reduced_channels = max(channels // reduction_ratio, 4)
        
        # 空间注意力生成网络
        self.attention_conv = nn.Sequential(
            # 降维
            nn.Conv3d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=False),
            
            # 空间感知卷积
            nn.Conv3d(reduced_channels, reduced_channels, kernel_size=7, 
                     padding=3, groups=reduced_channels, bias=False),
            nn.ReLU(inplace=False),
            
            # 升维 + 注意力权重生成
            nn.Conv3d(reduced_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # 全局上下文感知
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv3d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [B, C, D, H, W]
        
        Returns:
            enhanced_x: 注意力增强的特征 [B, C, D, H, W]
        """
        # 计算空间注意力权重
        spatial_att = self.attention_conv(x)  # [B, 1, D, H, W]
        
        # 计算全局上下文权重
        global_att = self.global_context(x)  # [B, C, 1, 1, 1]
        
        # 结合空间和通道注意力
        enhanced_x = x * spatial_att * global_att
        
        return enhanced_x 
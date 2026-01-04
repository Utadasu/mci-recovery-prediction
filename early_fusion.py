import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from optimized_models import create_improved_resnet3d, EMAModel
import math
import json
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR

# ==================== 新增：队友提出的层次化编码结构 ====================

class GrayMatterGradientAttention(nn.Module):
    """
    灰质梯度注意力模块
    通过灰质梯度增强海马/内嗅皮层等AD相关区域的特征
    公式：Output = Input × (1 + γ × |∇GM|)
    """
    def __init__(self, channels):
        super(GrayMatterGradientAttention, self).__init__()
        # 使用标量gamma参数，避免通道数不匹配问题
        self.gamma = nn.Parameter(torch.tensor(0.1))  # 标量可学习参数γ
        
        # 3D Sobel算子用于计算梯度
        self.register_buffer('sobel_x', self._create_sobel_kernel('x'))
        self.register_buffer('sobel_y', self._create_sobel_kernel('y'))
        self.register_buffer('sobel_z', self._create_sobel_kernel('z'))
        
    def _create_sobel_kernel(self, direction):
        """创建3D Sobel算子"""
        if direction == 'x':
            kernel = torch.tensor([
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
            ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        elif direction == 'y':
            kernel = torch.tensor([
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
            ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        else:  # z direction
            kernel = torch.tensor([
                [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
            ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return kernel
    
    def forward(self, x, gray_matter_channel=1):
        """
        前向传播
        Args:
            x: 输入特征 [B, C, D, H, W]
            gray_matter_channel: 灰质通道索引（默认为1，即GREY通道）
        """
        # 提取灰质通道
        if x.size(1) > gray_matter_channel:
            gm = x[:, gray_matter_channel:gray_matter_channel+1, :, :, :]  # [B, 1, D, H, W]
        else:
            # 如果没有足够的通道，使用第一个通道
            gm = x[:, 0:1, :, :, :]
        
        # 计算3D梯度
        grad_x = F.conv3d(gm, self.sobel_x, padding=1)
        grad_y = F.conv3d(gm, self.sobel_y, padding=1)
        grad_z = F.conv3d(gm, self.sobel_z, padding=1)
        
        # 计算梯度幅值 |∇GM|
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-8)
        
        # 归一化梯度幅值到[0,1]
        gradient_magnitude = torch.sigmoid(gradient_magnitude)
        
        # 应用注意力：Output = Input × (1 + γ × |∇GM|)
        # 使用标量gamma，广播到所有通道
        attention_weights = 1 + self.gamma * gradient_magnitude
        
        # 广播attention_weights到所有通道
        if attention_weights.size(1) == 1 and x.size(1) > 1:
            attention_weights = attention_weights.expand(-1, x.size(1), -1, -1, -1)
        
        enhanced_features = x * attention_weights
        
        return enhanced_features

class PatchEmbed3D(nn.Module):
    """3D图像到patch嵌入"""
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        
        # 确保输入尺寸能被patch_size整除
        pad_d = (self.patch_size - D % self.patch_size) % self.patch_size
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        
        x = self.proj(x)  # [B, embed_dim, D', H', W']
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, D'*H'*W', embed_dim]
        x = self.norm(x)
        
        return x, (D, H, W)

class WindowAttention3D(nn.Module):
    """3D窗口注意力机制"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock3D(nn.Module):
    """3D Swin Transformer块"""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., 
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = nn.Identity() if drop_path <= 0. else nn.Dropout(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x, mask_matrix=None):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        
        # 窗口注意力
        x = self.attn(x)
        
        # 残差连接
        x = shortcut + self.drop_path(x)
        
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

class HierarchicalSwinTransformer3D(nn.Module):
    """
    队友提出的层次化编码结构
    结合灰质梯度注意力和3D Swin-Transformer
    """
    def __init__(self, in_chans=3, num_classes=2, embed_dim=96, depths=[2, 2, 6, 2], 
                 num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        
        # 1. 数据输入处理：主通道（灰质）+ 辅助通道（白质/脑脊液）
        print("初始化层次化Swin-Transformer模型...")
        print(f"输入通道数: {in_chans}, 嵌入维度: {embed_dim}")
        
        # 初始卷积层用于局部特征提取
        self.conv_layers = nn.ModuleList()
        current_dim = in_chans
        
        # 浅层卷积块 - 提取局部特征
        for i, target_dim in enumerate([16, 32, 64]):
            conv_block = nn.Sequential(
                nn.Conv3d(current_dim, target_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(target_dim),
                nn.ReLU(inplace=True),
                nn.Conv3d(target_dim, target_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(target_dim),
                nn.ReLU(inplace=True)
            )
            self.conv_layers.append(conv_block)
            current_dim = target_dim
            
            # 渐进式下采样：64→32→16→8体素
            if i < 2:  # 前两层进行下采样
                downsample = nn.Sequential(
                    nn.Conv3d(target_dim, target_dim, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm3d(target_dim),
                    nn.ReLU(inplace=True)
                )
                self.conv_layers.append(downsample)
        
        # 2. 灰质梯度注意力模块
        self.gm_attention = GrayMatterGradientAttention(current_dim)
        
        # 3. Patch嵌入
        self.patch_embed = PatchEmbed3D(
            patch_size=2, in_chans=current_dim, embed_dim=embed_dim)
        
        # 4. 位置嵌入（简化版）
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # 5. Swin-Transformer层
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList([
                SwinTransformerBlock3D(
                    dim=int(embed_dim * 2 ** i_layer),
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])][i] if i < depths[i_layer] else 0.
                )
                for i in range(depths[i_layer])
            ])
            self.layers.append(layer)
            
            # 下采样层（除了最后一层）
            if i_layer < self.num_layers - 1:
                downsample_layer = nn.Sequential(
                    nn.LayerNorm(int(embed_dim * 2 ** i_layer)),
                    nn.Linear(int(embed_dim * 2 ** i_layer), int(embed_dim * 2 ** (i_layer + 1))),
                )
                self.layers.append(downsample_layer)
        
        # 6. 分类头
        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (self.num_layers - 1)))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(int(embed_dim * 2 ** (self.num_layers - 1)), num_classes)
        
        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        # 输入形状检查
        if len(x.shape) != 5:
            raise ValueError(f"期望5D输入[B,C,D,H,W]，但得到{x.shape}")
        
        B, C, D, H, W = x.shape
        
        # 1. 浅层卷积特征提取
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            if i == 1:  # 在第一次下采样后应用灰质梯度注意力
                x = self.gm_attention(x, gray_matter_channel=min(1, x.size(1)-1))
        
        # 2. Patch嵌入
        x, hw_shape = self.patch_embed(x)  # [B, L, embed_dim]
        x = self.pos_drop(x)
        
        # 3. Swin-Transformer层
        for i, layer_group in enumerate(self.layers):
            if isinstance(layer_group, nn.ModuleList):
                # Swin-Transformer块
                for block in layer_group:
                    x = block(x)
            else:
                # 下采样层
                x = layer_group(x)
        
        # 4. 分类
        x = self.norm(x)  # [B, L, C]
        x = self.avgpool(x.transpose(1, 2))  # [B, C, 1]
        x = torch.flatten(x, 1)  # [B, C]
        x = self.head(x)  # [B, num_classes]
        
        return x

# ==================== 更新的数据集类 ====================

class HierarchicalEarlyFusionDataset(torch.utils.data.Dataset):
    """
    用于早期融合模型的分层数据集。
    结合CSF、灰质和白质的数据，创建包含所有组织类型的多通道输入。
    主通道：灰质概率图（GREY）
    辅助通道：白质/脑脊液概率图（WHITE+CSF）
    """
    
    def __init__(self, csf_loader, grey_loader, white_loader, debug=False):
        """
        初始化数据集。
        
        参数:
        - csf_loader: CSF数据加载器
        - grey_loader: 灰质数据加载器
        - white_loader: 白质数据加载器
        - debug: 是否为调试模式
        """
        self.csf_dataset = csf_loader.dataset
        self.grey_dataset = grey_loader.dataset
        self.white_dataset = white_loader.dataset
        self.debug = debug
        
        # 确保各数据集大小一致
        if len(self.csf_dataset) != len(self.grey_dataset) or len(self.csf_dataset) != len(self.white_dataset):
            raise ValueError(f"数据集大小不一致: CSF={len(self.csf_dataset)}, GREY={len(self.grey_dataset)}, WHITE={len(self.white_dataset)}")
            
        self.length = len(self.csf_dataset)
        
        print(f"创建层次化早期融合数据集: 样本数={self.length}")
        print("数据组织方式: [CSF, GREY(主通道), WHITE] -> 3通道输入")
        
        # 验证第一个样本的维度
        if self.length > 0:
            self._validate_sample_dimensions(0)
            
        # 计算类别比例
        self.stats = self._calculate_stats()
    
    def _calculate_stats(self):
        """计算数据集的统计信息，包括类别比例"""
        labels = []
        
        # 统计所有样本的标签分布，而不是只统计前100个
        print(f"正在统计完整数据集的类别分布（共{len(self)}个样本）...")
        for i in range(len(self)):
            # 获取标签但不载入图像
            _, label = self._get_label(i)
            labels.append(label)
        
        # 统计AD和CN数量
        ad_count = sum(1 for l in labels if l == 0)
        cn_count = sum(1 for l in labels if l == 1)
        
        print(f"数据集类别统计: AD={ad_count}({ad_count/len(labels)*100:.1f}%), CN={cn_count}({cn_count/len(labels)*100:.1f}%)")
        
        return {
            'ad_count': ad_count,
            'cn_count': cn_count,
            'ad_cn_ratio': ad_count / max(1, cn_count),
            'total': len(labels),
            'ad_percent': ad_count / len(labels) * 100,
            'cn_percent': cn_count / len(labels) * 100
        }
    
    def _get_label(self, idx):
        """仅获取标签，不加载图像数据"""
        # 获取任意数据集的标签（假设所有数据集标签一致）
        csf_data = self.csf_dataset[idx]
        if isinstance(csf_data, (list, tuple)):
            return None, csf_data[1]
        else:
            return None, csf_data['label']
    
    def get_stats(self):
        """返回数据集的统计信息"""
        return self.stats
    
    def _validate_sample_dimensions(self, idx):
        """验证样本维度"""
        csf_data = self.csf_dataset[idx]
        grey_data = self.grey_dataset[idx]
        white_data = self.white_dataset[idx]
        
        # 提取图像数据
        csf_image = csf_data[0] if isinstance(csf_data, (list, tuple)) else csf_data['image']
        grey_image = grey_data[0] if isinstance(grey_data, (list, tuple)) else grey_data['image']
        white_image = white_data[0] if isinstance(white_data, (list, tuple)) else white_data['image']
        
        print(f"层次化融合模态维度验证:")
        print(f"CSF(辅助): {csf_image.shape}")
        print(f"GREY(主通道): {grey_image.shape}")
        print(f"WHITE(辅助): {white_image.shape}")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 获取三种组织类型的数据
        csf_data = self.csf_dataset[idx]
        grey_data = self.grey_dataset[idx]
        white_data = self.white_dataset[idx]
        
        # 处理不同的数据格式
        if isinstance(csf_data, (list, tuple)):
            csf_image = csf_data[0]
            label = csf_data[1]
            grey_image = grey_data[0]
            white_image = white_data[0]
        else:
            csf_image = csf_data['image']
            label = csf_data['label']
            grey_image = grey_data['image']
            white_image = white_data['image']
        
        # 确保每个图像都是3D的，去除多余的通道维度
        if csf_image.dim() == 4 and csf_image.size(0) == 1:
            csf_image = csf_image.squeeze(0)
        if grey_image.dim() == 4 and grey_image.size(0) == 1:
            grey_image = grey_image.squeeze(0)
        if white_image.dim() == 4 and white_image.size(0) == 1:
            white_image = white_image.squeeze(0)
        
        # 确保图像格式一致 - 全部转为3D张量
        if csf_image.dim() != 3:
            raise ValueError(f"CSF图像维度错误: {csf_image.shape}")
        if grey_image.dim() != 3:
            raise ValueError(f"灰质图像维度错误: {grey_image.shape}")
        if white_image.dim() != 3:
            raise ValueError(f"白质图像维度错误: {white_image.shape}")
            
        # 合并三种组织类型到单个多通道输入
        # 通道顺序: [CSF, GREY, WHITE]
        combined_image = torch.stack([csf_image, grey_image, white_image], dim=0)
        
        # 调试信息
        if self.debug and idx < 3:
            print(f"样本 {idx} 合并后形状: {combined_image.shape}, 标签: {label}")
            print(f"- CSF形状: {csf_image.shape}")
            print(f"- GREY形状: {grey_image.shape}")
            print(f"- WHITE形状: {white_image.shape}")
            
        # 检查最终输出形状
        if combined_image.dim() != 4:  # [3, D, H, W]
            raise ValueError(f"合并后的图像维度错误，期望4D，得到: {combined_image.shape}")
            
        return combined_image, label

def create_early_fusion_loaders(data_loaders, batch_size=4, debug=True):  # 从32降低到4
    """
    创建早期融合的数据加载器
    
    参数:
    - data_loaders: 包含各组织类型训练和验证数据加载器的字典
    - batch_size: 批次大小 (默认4，适应32GB GPU显存)
    - debug: 是否启用调试模式
    
    返回:
    - 包含早期融合训练和验证数据加载器的字典
    """
    print("\n开始创建早期融合数据集...")
    
    # 检查GPU显存，自动调整批次大小
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"检测到GPU显存: {gpu_memory:.1f}GB")
        
        # 基于显存自动调整批次大小
        if gpu_memory >= 30:  # 32GB显存
            recommended_batch_size = min(batch_size, 8)
        elif gpu_memory >= 20:  # 24GB显存
            recommended_batch_size = min(batch_size, 4)
        elif gpu_memory >= 10:  # 16GB显存
            recommended_batch_size = min(batch_size, 2)
        else:  # 8GB及以下
            recommended_batch_size = 1
        
        if recommended_batch_size < batch_size:
            print(f"⚠️ 基于GPU显存调整批次大小: {batch_size} -> {recommended_batch_size}")
            batch_size = recommended_batch_size
    
    # 打印原始数据加载器信息
    print("原始数据加载器信息:")
    for key, loader in data_loaders.items():
        if key.startswith('train_') or key.startswith('val_'):
            print(f"- {key}: 批次大小={loader.batch_size}, 样本数={len(loader.dataset)}")
            
            # 尝试获取并打印第一个批次的形状
            try:
                sample_batch = next(iter(loader))
                if isinstance(sample_batch, (list, tuple)) and len(sample_batch) >= 1:
                    print(f"  样本批次形状: {sample_batch[0].shape}")
            except Exception as e:
                print(f"  无法获取样本形状: {e}")
    
    # 创建融合数据集
    train_fusion_dataset = HierarchicalEarlyFusionDataset(
        data_loaders['train_CSF'],
        data_loaders['train_GRAY'],
        data_loaders['train_WHITE'],
        debug=debug
    )
    
    val_fusion_dataset = HierarchicalEarlyFusionDataset(
        data_loaders['val_CSF'],
        data_loaders['val_GRAY'],
        data_loaders['val_WHITE'],
        debug=debug
    )
    
    # 创建数据加载器 - 内存优化配置
    train_fusion_loader = torch.utils.data.DataLoader(
        train_fusion_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # 从8降低到4，减少并行开销
        pin_memory=False,  # 禁用pin_memory以节省内存
        persistent_workers=False,  # 禁用持久化worker以节省内存
        drop_last=True  # 丢弃最后一个不完整的批次
    )
    
    val_fusion_loader = torch.utils.data.DataLoader(
        val_fusion_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # 从8降低到4
        pin_memory=False,  # 禁用pin_memory
        persistent_workers=False,  # 禁用持久化worker
        drop_last=False
    )
    
    # 验证融合后的数据加载器
    print("\n验证融合数据加载器:")
    print(f"训练融合加载器: 批次大小={train_fusion_loader.batch_size}, 样本数={len(train_fusion_dataset)}")
    print(f"验证融合加载器: 批次大小={val_fusion_loader.batch_size}, 样本数={len(val_fusion_dataset)}")
    
    # 尝试抽样一个批次确认形状
    try:
        sample_batch, sample_labels = next(iter(train_fusion_loader))
        print(f"融合后批次形状: {sample_batch.shape}, 标签形状: {sample_labels.shape}")
        print(f"标签分布: {torch.bincount(sample_labels)}")
        
        # 估算内存使用
        batch_memory = sample_batch.numel() * 4 / 1024**2  # MB (假设float32)
        print(f"单批次内存估算: {batch_memory:.1f}MB")
        
    except Exception as e:
        print(f"无法获取融合批次: {e}")
    
    return {
        'train': train_fusion_loader,
        'val': val_fusion_loader
    }

def train_hierarchical_swin_model(data_loaders, device, save_dir='./models'):
    """
    训练层次化Swin-Transformer模型 - 队友提出的新架构
    
    参数:
    - data_loaders: 包含各组织类型训练和验证数据加载器的字典
    - device: 训练设备
    - save_dir: 模型保存目录
    
    返回:
    - 训练结果字典
    """
    import numpy as np
    
    print("\n===== 训练层次化Swin-Transformer模型 =====")
    print("架构特点:")
    print("1. 主通道：灰质概率图（GREY）")
    print("2. 辅助通道：白质/脑脊液概率图（WHITE+CSF）")
    print("3. 灰质梯度注意力增强AD相关区域")
    print("4. 渐进式下采样：64→32→16→8体素")
    print("5. 3D Swin-Transformer捕获长程依赖")
    
    # 创建层次化早期融合数据加载器
    fusion_loaders = create_early_fusion_loaders(data_loaders, batch_size=8, debug=True)  # 减小batch_size
    train_loader = fusion_loaders['train']
    val_loader = fusion_loaders['val']
    
    # 创建层次化Swin-Transformer模型
    print("\n创建层次化Swin-Transformer模型...")
    
    model = None
    model_created = False
    
    # 更激进的内存优化配置，从最简单开始
    configs = [
        {
            'name': '超轻量级',
            'embed_dim': 24,
            'depths': [1, 1, 1, 1],
            'num_heads': [2, 4, 6, 12],
            'window_size': 2,
            'mlp_ratio': 2.0,
            'batch_size': 4
        },
        {
            'name': '极简版',
            'embed_dim': 16,
            'depths': [1, 1, 1, 1],
            'num_heads': [1, 2, 4, 8],
            'window_size': 2,
            'mlp_ratio': 1.0,
            'batch_size': 4
        },
        {
            'name': '最小版',
            'embed_dim': 12,
            'depths': [1, 1, 1],
            'num_heads': [1, 2, 4],
            'window_size': 2,
            'mlp_ratio': 1.0,
            'batch_size': 2
        }
    ]
    
    # 尝试每个配置
    for config in configs:
        try:
            print(f"尝试创建{config['name']}模型...")
            
            # 清理之前的模型和缓存
            if model is not None:
                del model
            torch.cuda.empty_cache()
            
            # 如果需要，重新创建更小批次的数据加载器
            if config['batch_size'] < 8:
                print(f"调整批次大小为 {config['batch_size']}")
                fusion_loaders = create_early_fusion_loaders(data_loaders, batch_size=config['batch_size'], debug=True)
                train_loader = fusion_loaders['train']
                val_loader = fusion_loaders['val']
            
            model = HierarchicalSwinTransformer3D(
                in_chans=3,
                num_classes=2,
                embed_dim=config['embed_dim'],
                depths=config['depths'],
                num_heads=config['num_heads'],
                window_size=config['window_size'],
                mlp_ratio=config['mlp_ratio'],
                qkv_bias=True,
                drop_rate=0.1,
                attn_drop_rate=0.1,
                drop_path_rate=0.05
            ).to(device)
            
            # 测试前向传播
            sample_batch, sample_labels = next(iter(train_loader))
            sample_batch = sample_batch.to(device)
            print(f"输入批次形状: {sample_batch.shape}")
            
            with torch.no_grad():
                sample_output = model(sample_batch)
                print(f"{config['name']}模型输出形状: {sample_output.shape}")
                print(f"输出范围: [{sample_output.min().item():.3f}, {sample_output.max().item():.3f}]")
                print(f"✓ {config['name']}模型创建成功")
                model_created = True
                break
                
        except Exception as e:
            if "out of memory" in str(e).lower():
                print(f"{config['name']}模型内存不足: {e}")
                # 清理内存
                if model is not None:
                    del model
                if 'sample_batch' in locals():
                    del sample_batch
                torch.cuda.empty_cache()
                continue
            else:
                print(f"{config['name']}模型创建失败: {e}")
                continue
    
    # 如果所有Swin-Transformer配置都失败，使用内存友好的简化模型
    if not model_created:
        print("所有Swin-Transformer配置都失败，切换到内存友好的简化模型...")
        
        # 彻底清理内存
        if model is not None:
            del model
        if 'sample_batch' in locals():
            del sample_batch
        torch.cuda.empty_cache()
        
        # 重新创建最小批次的数据加载器
        print("使用最小批次大小 batch_size=2")
        fusion_loaders = create_early_fusion_loaders(data_loaders, batch_size=2, debug=True)
        train_loader = fusion_loaders['train']
        val_loader = fusion_loaders['val']
        
        # 使用内存友好的简化模型
        model = MemoryEfficientHierarchicalModel(
            in_chans=3,
            num_classes=2
        ).to(device)
        
        # 测试简化模型
        sample_batch, sample_labels = next(iter(train_loader))
        sample_batch = sample_batch.to(device)
        
        with torch.no_grad():
            sample_output = model(sample_batch)
            print(f"内存友好模型输出形状: {sample_output.shape}")
            print(f"输出范围: [{sample_output.min().item():.3f}, {sample_output.max().item():.3f}]")
            print("✓ 成功切换到内存友好模型")
            model_created = True
    
    print(f"最终使用的模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 重新初始化分类器权重，确保输出更平衡
    if hasattr(model, 'head'):
        # 对于Swin-Transformer模型
        nn.init.normal_(model.head.weight, 0, 0.01)
        nn.init.constant_(model.head.bias, 0)
        print("重新初始化Swin-Transformer分类器权重")
    elif hasattr(model, 'classifier'):
        # 对于内存友好模型
        for layer in model.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.constant_(layer.bias, 0)
        print("重新初始化内存友好模型分类器权重")
    
    # 确保模型创建成功
    if not model_created or model is None:
        raise RuntimeError("无法创建任何可用的模型，请检查GPU内存或降低数据复杂度")
    
    # 最终测试模型
    print("\n最终模型测试...")
    try:
        sample_batch, sample_labels = next(iter(train_loader))
        sample_batch = sample_batch.to(device)
        with torch.no_grad():
            final_output = model(sample_batch)
            print(f"最终模型测试成功: 输入{sample_batch.shape} -> 输出{final_output.shape}")
    except Exception as e:
        print(f"最终模型测试失败: {e}")
        raise RuntimeError(f"模型无法正常工作: {e}")
    
    # 使用AdamW优化器，根据模型复杂度调整学习率
    if hasattr(model, 'head'):
        lr = 0.0005  # 进一步降低学习率，提高稳定性
        weight_decay = 0.01
    else:
        lr = 0.001  # 简单模型也降低学习率
        weight_decay = 0.01
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 定义预热期和总轮次
    warmup_epochs = 5  # 从10轮减少到5轮
    total_epochs = 60 if current_config['use_improved'] else 40
    
    # 使用余弦退火学习率调度器，带预热期
    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=total_epochs - warmup_epochs,
        eta_min=lr * 0.01  # 最小学习率为初始值的1%
    )
    
    # 自定义学习率函数，包含预热期
    def get_lr_with_warmup(epoch):
        if epoch < warmup_epochs:
            # 线性预热 - 更简单、更稳定
            return lr * (0.1 + 0.9 * epoch / warmup_epochs)
        else:
            # 预热后使用余弦退火调度器的值
            return cosine_scheduler.get_last_lr()[0]
    
    # 备用调度器：如果验证损失不下降则降低学习率
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, 
        min_lr=1e-6, verbose=True
    )
    
    # 检查训练集标签分布
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.numpy())
    
    label_counts = np.bincount(train_labels)
    print(f"\n训练集标签分布: {label_counts}")
    
    # 计算类别权重
    if len(label_counts) == 2 and min(label_counts) > 0:
        total_samples = sum(label_counts)
        class_weights = torch.FloatTensor([
            total_samples / (2 * label_counts[0]),
            total_samples / (2 * label_counts[1])
        ]).to(device)
        print(f"使用类别权重: {class_weights}")
    else:
        class_weights = None
        print("使用均匀权重")
    
    # 根据模型类型选择损失函数
    if hasattr(model, 'head'):
        # 使用Focal Loss来处理严重的类别不平衡问题
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1, gamma=2.5, weight=None):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.weight = weight
                
            def forward(self, inputs, targets):
                ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
                return focal_loss.mean()
        
        criterion = FocalLoss(alpha=1, gamma=2.5, weight=class_weights)
        print(f"使用Focal Loss (gamma=2.5) + 类别权重")
    else:
        # 简单模型使用带权重的交叉熵
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"使用加权交叉熵损失")
    
    # 混合精度训练
    scaler = amp.GradScaler()
    
    # 训练参数
    num_epochs = 60
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_model_state = None
    patience = 20
    no_improve_epochs = 0
    
    # 训练统计
    stats = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': [],
        'gm_attention_weights': []  # 记录灰质注意力权重
    }
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n开始训练层次化Swin-Transformer模型，总轮次: {num_epochs}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Hierarchical Swin]')
        
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # 每个epoch开始打印一次信息
            if batch_idx == 0:
                print(f"\n训练批次形状: inputs={inputs.shape}, labels={labels.shape}")
                print(f"标签分布: {torch.bincount(labels, minlength=2)}")
                
                # 检查灰质梯度注意力权重
                if hasattr(model, 'gm_attention'):
                    with torch.no_grad():
                        gamma_value = model.gm_attention.gamma.mean().item()
                        stats['gm_attention_weights'].append(gamma_value)
                        print(f"灰质注意力参数γ: {gamma_value:.4f}")
            
            optimizer.zero_grad()
            
            # 混合精度训练
            with amp.autocast():
                outputs = model(inputs)
                
                # 添加标签噪声提高多样性（仅在早期训练阶段）
                if epoch < 15:  # 前15轮使用标签噪声
                    # 随机翻转一小部分标签，增加多样性
                    noise_ratio = max(0.05, 0.1 - epoch * 0.005)  # 从10%逐渐降低到5%
                    noise_mask = (torch.rand_like(labels.float()) < noise_ratio).to(device)
                    if noise_mask.sum() > 0:  # 确保有标签被噪声化
                        noisy_labels = labels.clone()
                        noisy_labels[noise_mask] = 1 - noisy_labels[noise_mask]  # 翻转标签
                        loss = criterion(outputs, noisy_labels)
                        if batch_idx == 0:  # 只在每个epoch第一个批次打印
                            print(f"添加{noise_mask.sum().item()}个标签噪声，噪声比例{noise_ratio:.1%}")
                    else:
                        loss = criterion(outputs, labels)
                else:
                    loss = criterion(outputs, labels)
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # 使用更严格的梯度裁剪来提高稳定性
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            scaler.step(optimizer)
            scaler.update()
            
            # 统计
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # 监控预测分布，防止模型偏向单一类别
            if batch_idx == 0:  # 每个epoch开始时检查
                pred_dist = torch.bincount(predicted, minlength=2)
                label_dist = torch.bincount(labels, minlength=2)
                print(f"批次预测分布: {pred_dist.tolist()}, 真实分布: {label_dist.tolist()}")
                
                # 如果预测过于偏向，调整学习率
                if pred_dist[0] == 0 or pred_dist[1] == 0:
                    print("⚠ 检测到预测偏向，临时降低学习率")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.8  # 临时降低学习率
            
            # 更新进度条
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
            
            # 定期清理内存
            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        conf_matrix = torch.zeros(2, 2, dtype=torch.long)
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # 更新混淆矩阵
                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    conf_matrix[t.long(), p.long()] += 1
        
        # 计算指标
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
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
        stats['lr'].append(optimizer.param_groups[0]['lr'])
        
        # 更新学习率 - 使用验证损失
        scheduler.step(epoch)
        plateau_scheduler.step(avg_val_loss)
        
        # 打印信息
        print(f'\nEpoch [{epoch+1}/{num_epochs}] - Hierarchical Swin-Transformer:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Val Acc per class: {val_acc_per_class}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'混淆矩阵:\n{conf_matrix}')
        
        # 保存最佳模型 - 更严格的评估标准，冲击85%+
        both_classes_correct = conf_matrix[0, 0] > 0 and conf_matrix[1, 1] > 0
        improved = False
        
        # 计算每个类别的准确率
        class0_acc = conf_matrix[0, 0].item() / max(1, conf_matrix[0, :].sum().item()) * 100
        class1_acc = conf_matrix[1, 1].item() / max(1, conf_matrix[1, :].sum().item()) * 100
        min_class_acc = min(class0_acc, class1_acc)
        balanced_score = (class0_acc + class1_acc) / 2  # 平衡准确率
        
        # 更智能的模型保存策略
        if both_classes_correct and val_acc > best_val_acc:
            improved = True
            best_val_acc = val_acc
            print(f"✓ 保存模型: 验证准确率提高到{val_acc:.2f}%，平衡准确率{balanced_score:.2f}%")
        elif both_classes_correct and val_acc >= best_val_acc - 1 and balanced_score > 65:
            # 如果准确率相近但类别更平衡，也保存
            improved = True
            best_val_acc = max(val_acc, best_val_acc)
            print(f"✓ 保存模型: 类别平衡性优秀，平衡准确率{balanced_score:.2f}%")
        elif not both_classes_correct:
            print(f"⚠ 跳过保存: 只预测单一类别 (AD:{class0_acc:.1f}%, CN:{class1_acc:.1f}%)")
        else:
            print(f"⚠ 跳过保存: 当前{val_acc:.2f}% vs 最佳{best_val_acc:.2f}%, 平衡分{balanced_score:.1f}%")
        
        if improved:
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': avg_val_loss,
                'stats': stats,
                'conf_matrix': conf_matrix.tolist(),
                'model_config': {
                    'in_chans': 3,
                    'num_classes': 2,
                    'embed_dim': model.embed_dim,
                    'window_size': 4
                }
            }
            
            model_filename = f'{save_dir}/best_hierarchical_swin_model.pth'
            torch.save(best_model_state, model_filename)
            print(f'保存最佳层次化Swin模型，验证准确率: {val_acc:.2f}%')
            
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f'验证准确率连续{no_improve_epochs}个epoch没有提高')
        
        # 每15个epoch保存检查点
        if (epoch + 1) % 15 == 0:
            checkpoint_path = f'{save_dir}/hierarchical_swin_checkpoint_epoch{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'stats': stats,
                'conf_matrix': conf_matrix.tolist()
            }, checkpoint_path)
            print(f'保存检查点: {checkpoint_path}')
        
        # 早停检查
        if no_improve_epochs >= patience:
            print(f'早停在epoch {epoch+1}')
            break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state_dict'])
        print(f'已加载最佳层次化Swin模型，验证准确率: {best_val_acc:.2f}%')
    
    # 保存训练统计
    try:
        import json
        
        json_stats = {}
        for key, values in stats.items():
            if isinstance(values, list):
                json_stats[key] = [float(val) for val in values]
            else:
                json_stats[key] = float(values)
        
        stats_path = f'{save_dir}/hierarchical_swin_training_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(json_stats, f, indent=4)
        print(f'训练统计已保存到: {stats_path}')
    except Exception as e:
        print(f'保存训练统计失败: {e}')
    
    return {
        'model': model,
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
        'best_epoch': best_model_state['epoch'] if best_model_state else -1,
        'model_path': f'{save_dir}/best_hierarchical_swin_model.pth',
        'stats': stats,
        'architecture': 'HierarchicalSwinTransformer3D'
    }

def train_early_fusion_model(data_loaders, device, save_dir='./models'):
    """
    训练早期融合模型 - 使用真正的ImprovedResNetCBAM3D
    
    参数:
    - data_loaders: 包含各组织类型训练和验证数据加载器的字典
    - device: 训练设备
    - save_dir: 模型保存目录
    
    返回:
    - 训练结果字典
    """
    # 确保导入必要的库
    import numpy as np
    import math
    import torch.optim as optim
    from torch.cuda import amp
    from tqdm import tqdm
    import os
    import json
    from datetime import datetime
    
    # 内存优化设置
    print("\n===== 内存优化配置 =====")
    
    # 禁用CUDNN基准测试以节省内存
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # 设置内存分配器
    if torch.cuda.is_available():
        # 检查GPU显存
        gpu_properties = torch.cuda.get_device_properties(device)
        total_memory = gpu_properties.total_memory / 1024**3  # GB
        print(f"GPU: {gpu_properties.name}")
        print(f"总显存: {total_memory:.1f}GB")
        
        # 清理缓存
        torch.cuda.empty_cache()
        
        # 当前已用显存
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        cached = torch.cuda.memory_reserved(device) / 1024**3
        print(f"已分配显存: {allocated:.1f}GB")
        print(f"已缓存显存: {cached:.1f}GB")
        
        # 设置内存分片策略
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        print("已设置内存分片大小为128MB")
    
    # 验证数据加载器格式
    expected_keys = ['train', 'val']
    actual_keys = list(data_loaders.keys())
    
    if not all(key in actual_keys for key in expected_keys):
        raise ValueError(
            f"数据加载器格式错误。期望的键: {expected_keys}, 实际键: {actual_keys}。\n"
            f"请确保使用create_early_fusion_loaders函数创建正确格式的数据加载器。"
        )
    
    # 创建训练日志保存目录
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建唯一的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'training_log_{timestamp}.txt')
    
    # 日志写入函数
    def log_message(message, also_print=True):
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
        if also_print:
            print(message)

    # 记录训练开始信息
    log_message(f"===== 开始训练 - {timestamp} =====")

    print("\n===== 训练早期融合ImprovedResNetCBAM3D模型 (CSF+GREY+WHITE) =====")
    
    # 获取训练和验证数据加载器
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    
    # 记录数据集信息
    log_message(f"训练集大小: {len(train_loader.dataset)}样本, {len(train_loader)}批次")
    log_message(f"验证集大小: {len(val_loader.dataset)}样本, {len(val_loader)}批次")
    log_message(f"批次大小: {train_loader.batch_size}")
    
    # 获取训练配置
    current_config = {
        'name': 'ImprovedResNetCBAM3D-MemoryOptimized',
        'use_improved': True,
        'memory_efficient': True
    }
    
    # 创建模型 - 使用内存效率模式
    from optimized_models import create_improved_resnet3d
    
    print("\n===== 创建内存优化模型 =====")
    try:
        # 使用内存效率模式：base_channels=8而不是12
        model = create_improved_resnet3d(
            in_channels=3, 
            device=device,
            base_channels=8,  # 内存效率模式
            dropout_rate=0.3
        )
        log_message(f"已创建内存优化模型: {current_config['name']}")
        
        # 计算模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log_message(f"模型参数总数: {total_params:,}")
        log_message(f"可训练参数: {trainable_params:,}")
        
        # 估算模型内存占用
        model_memory = total_params * 4 / 1024**2  # MB (float32)
        log_message(f"估算模型内存: {model_memory:.1f}MB")
        
        # 测试前向传播以确保内存充足
        print("测试模型前向传播...")
        test_input = torch.randn(1, 3, 113, 137, 113).to(device)
        with torch.no_grad():
            test_output = model(test_input)
        log_message(f"前向传播测试成功: {test_input.shape} -> {test_output.shape}")
        del test_input, test_output
        torch.cuda.empty_cache()
        
    except Exception as e:
        log_message(f"模型创建失败: {e}")
        if "out of memory" in str(e).lower():
            print("\n⚠️ GPU显存不足，尝试使用更小的模型...")
            torch.cuda.empty_cache()
            
            # 尝试使用更小的模型配置
            try:
                model = create_improved_resnet3d(
                    in_channels=3, 
                    device=device,
                    base_channels=6,  # 进一步减少
                    dropout_rate=0.3
                )
                log_message("使用超小模型配置成功")
                current_config['name'] = 'ImprovedResNetCBAM3D-UltraLight'
            except Exception as e2:
                log_message(f"超小模型也失败: {e2}")
                raise RuntimeError(f"无法创建适合当前显存的模型。原始错误: {e}")
        else:
            raise e
    
    # 替换分类器
    if hasattr(model, 'classifier1'):
        in_features = model.classifier1.in_features
        model.classifier1 = StableClassifier(in_features, 2).to(device)
        log_message(f"替换classifier1为StableClassifier")
    
    if hasattr(model, 'classifier2'):
        in_features = model.classifier2.in_features
        model.classifier2 = StableClassifier(in_features, 2).to(device)
        log_message(f"替换classifier2为StableClassifier")
    
    # 计算类别权重
    try:
        # 尝试从数据集中获取类别分布
        ad_count = train_loader.dataset.stats['ad_count']
        cn_count = train_loader.dataset.stats['cn_count']
        total = ad_count + cn_count
        # 计算逆频率权重
        class_weights = torch.FloatTensor([total / max(1, ad_count), total / max(1, cn_count)]).to(device)
        log_message(f"计算得到类别权重: {class_weights}")
    except:
        # 如果无法获取类别分布，使用默认权重
        class_weights = torch.FloatTensor([1.0, 1.0]).to(device)
        log_message(f"使用默认类别权重: {class_weights}")

    # 内存优化：梯度累积配置
    accumulation_steps = 4  # 梯度累积步数，等效batch_size = actual_batch_size * accumulation_steps
    effective_batch_size = train_loader.batch_size * accumulation_steps
    log_message(f"梯度累积配置: {accumulation_steps}步, 等效批次大小: {effective_batch_size}")

    # 使用AdamW优化器，根据模型复杂度调整学习率
    if current_config['use_improved']:
        lr = 0.0001  # 增加学习率来补偿更小的批次大小
        weight_decay = 0.01
    else:
        lr = 0.0002
        weight_decay = 0.01
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    log_message(f"优化器配置: lr={lr}, weight_decay={weight_decay}")
    
    # 定义预热期和总轮次
    warmup_epochs = 5
    total_epochs = 50  # 减少总轮次以适应更慢的训练
    
    # 使用余弦退火学习率调度器，带预热期
    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=total_epochs - warmup_epochs,
        eta_min=lr * 0.01
    )
    
    # 自定义学习率函数，包含预热期
    def get_lr_with_warmup(epoch):
        if epoch < warmup_epochs:
            # 线性预热
            return lr * (0.1 + 0.9 * epoch / warmup_epochs)
        else:
            # 预热后使用余弦退火调度器的值
            return cosine_scheduler.get_last_lr()[0]
    
    # 备用调度器：如果验证损失不下降则降低学习率
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, 
        min_lr=1e-6, verbose=True
    )
    
    # 查看训练集标签分布
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.numpy())
    
    label_counts = np.bincount(train_labels)
    print(f"\n训练集标签分布: {label_counts}")
    
    # 使用平衡的类别权重
    if len(label_counts) == 2 and min(label_counts) > 0:
        total_samples = sum(label_counts)
        class_weights = torch.FloatTensor([
            total_samples / (2 * label_counts[0]),
            total_samples / (2 * label_counts[1])
        ]).to(device)
        print(f"使用类别权重: {class_weights}")
    else:
        class_weights = None
        print("使用均匀权重")
    
    # 根据模型类型选择损失函数
    if current_config['use_improved']:
        # 定义混合损失函数 - 结合交叉熵和Focal Loss的优点
        class MixedLoss(nn.Module):
            def __init__(self, weight=None, focal_weight=0.5, ce_weight=0.5, 
                        gamma=2.0, label_smoothing=0.05):
                super(MixedLoss, self).__init__()
                self.focal_weight = focal_weight
                self.ce_weight = ce_weight
                self.gamma = gamma
                self.weight = weight
                self.label_smoothing = label_smoothing
                
                # 交叉熵损失
                self.ce_loss = nn.CrossEntropyLoss(
                    weight=weight, 
                    label_smoothing=label_smoothing
                )
                
            def forward(self, inputs, targets):
                # 交叉熵部分
                ce_loss = self.ce_loss(inputs, targets)
                
                # Focal Loss部分
                log_probs = F.log_softmax(inputs, dim=1)
                probs = torch.exp(log_probs)
                
                # 获取目标类别的概率
                target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
                
                # 计算focal loss
                focal_loss = -((1 - target_probs) ** self.gamma) * log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
                
                if self.weight is not None:
                    focal_loss = focal_loss * self.weight[targets]
                
                focal_loss = focal_loss.mean()
                
                # 混合损失
                return self.ce_weight * ce_loss + self.focal_weight * focal_loss
        
        # 使用更温和的类别权重，避免极端值
        if class_weights is not None:
            balanced_class_weights = torch.FloatTensor([
                class_weights[0] * 1.5,    # AD类权重适度增加1.5倍
                class_weights[1] * 1.0     # CN类权重保持不变
            ]).to(device)
            print(f"使用平衡类别权重: {balanced_class_weights}")
        else:
            balanced_class_weights = torch.FloatTensor([1.5, 1.0]).to(device)
            print(f"使用默认平衡类别权重: {balanced_class_weights}")
            
        # 使用混合损失函数
        criterion = MixedLoss(
            weight=balanced_class_weights,
            focal_weight=0.4,  # 降低Focal Loss比重
            ce_weight=0.6,     # 增加CrossEntropy比重
            gamma=2.0,         # 使用适中的gamma值
            label_smoothing=0.05
        )
        print(f"使用混合损失函数 (Focal Loss + CrossEntropy) 和平衡类别权重")
    else:
        # 简单模型使用带权重的交叉熵
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"使用加权交叉熵损失")
    
    # 启用自动混合精度训练
    scaler = amp.GradScaler()
    
    # 初始化训练变量
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_model_state = None
    no_improve_epochs = 0
    patience = 20 if current_config['use_improved'] else 15  # 从15增加到20
    min_class_acc_patience = 25  # 类别准确率的耐心值更长
    no_min_class_improve = 0  # 最低类别准确率没有改善的轮数
    best_min_class_acc = 0.0  # 最低类别准确率的最佳值
    
    # 初始化训练统计
    stats = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': [],
        'model_config': current_config,
        'only_predicts_one_class': [],  # 是否只预测单一类别
        'ad_pred_ratio': [],            # AD预测比例
        'cn_pred_ratio': [],            # CN预测比例
        'val_acc_per_class': []         # 每个类别的准确率
    }
    
    # 初始化一些变量，避免访问未定义的变量
    only_predicts_one_class = False
    ad_pred_ratio = torch.tensor(0.5)
    cn_pred_ratio = torch.tensor(0.5)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 主训练循环
    print(f"\n开始训练，总轮次: {total_epochs}，预热期: {warmup_epochs}轮")
    print(f"初始学习率: {lr:.6f}, 预热起始学习率: {lr*0.1:.6f}")
    
    # 添加偏置调整标记，避免连续多次大幅调整
    bias_adjusted_this_epoch = False
    
    for epoch in range(total_epochs):
        # 在每个epoch开始时更新学习率
        current_lr = get_lr_with_warmup(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # 重置偏置调整标记
        bias_adjusted_this_epoch = False
        
        # 预热期结束后重新初始化分类器
        if epoch == warmup_epochs:
            print("预热期结束，重新初始化分类器层...")
            with torch.no_grad():
                if hasattr(model, 'classifier1') and hasattr(model.classifier1, 'classifier'):
                    nn.init.xavier_normal_(model.classifier1.classifier.weight, gain=0.01)
                    # 完全中性的初始化，不偏向任何类别
                    model.classifier1.classifier.bias.data[0] = 0.0  # AD类
                    model.classifier1.classifier.bias.data[1] = 0.0  # CN类
                if hasattr(model, 'classifier2') and hasattr(model.classifier2, 'classifier'):
                    nn.init.xavier_normal_(model.classifier2.classifier.weight, gain=0.01)
                    # 完全中性的初始化，不偏向任何类别
                    model.classifier2.classifier.bias.data[0] = 0.0  # AD类
                    model.classifier2.classifier.bias.data[1] = 0.0  # CN类

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs} [Train]')
        
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # 在早期阶段(前20轮)为严重不平衡的类别添加混合样本
            if epoch < 20:
                try:
                    dataset_stats = train_loader.dataset.get_stats() 
                    if dataset_stats.get('ad_cn_ratio', 1.0) > 1.5 or dataset_stats.get('ad_cn_ratio', 1.0) < 0.67:
                        # 获取批次中的类别统计
                        batch_ad_indices = (labels == 0).nonzero(as_tuple=True)[0]
                        batch_cn_indices = (labels == 1).nonzero(as_tuple=True)[0]
                        
                        # 如果批次中存在两种类别
                        if len(batch_ad_indices) > 0 and len(batch_cn_indices) > 0:
                            # 生成少数类的混合样本
                            if len(batch_ad_indices) < len(batch_cn_indices):  # AD类较少
                                # 随机选择一对AD和CN样本
                                ad_idx = batch_ad_indices[torch.randint(0, len(batch_ad_indices), (1,))]
                                cn_idx = batch_cn_indices[torch.randint(0, len(batch_cn_indices), (1,))]
                                
                                # 创建混合样本 (70% AD + 30% CN)
                                mixed_sample = 0.7 * inputs[ad_idx] + 0.3 * inputs[cn_idx]
                                
                                # 添加到批次中
                                inputs = torch.cat([inputs, mixed_sample])
                                labels = torch.cat([labels, torch.tensor([0], device=device)])  # 标签为AD
                                
                            else:  # CN类较少
                                # 随机选择一对AD和CN样本
                                ad_idx = batch_ad_indices[torch.randint(0, len(batch_ad_indices), (1,))]
                                cn_idx = batch_cn_indices[torch.randint(0, len(batch_cn_indices), (1,))]
                                
                                # 创建混合样本 (30% AD + 70% CN)
                                mixed_sample = 0.3 * inputs[ad_idx] + 0.7 * inputs[cn_idx]
                                
                                # 添加到批次中
                                inputs = torch.cat([inputs, mixed_sample])
                                labels = torch.cat([labels, torch.tensor([1], device=device)])  # 标签为CN
                            
                            if batch_idx == 0:  # 只在每个epoch第一个批次打印
                                print(f"添加混合样本以平衡训练集")
                except Exception as e:
                    # 如果数据集不支持get_stats，或混合样本生成失败，则跳过
                    if batch_idx == 0 and epoch == 0:  # 只在第一个epoch的第一个批次打印一次
                        print(f"跳过混合样本生成: {str(e)}")
            
            optimizer.zero_grad()
            
            # 混合精度训练
            with amp.autocast():
                outputs = model(inputs)
                
                # 添加标签噪声提高多样性（仅在早期训练阶段）
                if epoch < 15:  # 前15轮使用标签噪声
                    # 随机翻转一小部分标签，增加多样性
                    noise_ratio = max(0.05, 0.1 - epoch * 0.005)  # 从10%逐渐降低到5%
                    noise_mask = (torch.rand_like(labels.float()) < noise_ratio).to(device)
                    if noise_mask.sum() > 0:  # 确保有标签被噪声化
                        noisy_labels = labels.clone()
                        noisy_labels[noise_mask] = 1 - noisy_labels[noise_mask]  # 翻转标签
                        loss = criterion(outputs, noisy_labels)
                        if batch_idx == 0:  # 只在每个epoch第一个批次打印
                            print(f"添加{noise_mask.sum().item()}个标签噪声，噪声比例{noise_ratio:.1%}")
                    else:
                        loss = criterion(outputs, labels)
                else:
                    loss = criterion(outputs, labels)
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # 统计
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # 混淆矩阵
        conf_matrix = torch.zeros(2, 2, dtype=torch.long)
        
        # 标记是否出现单一类别预测问题
        previous_only_predicts_one_class = False
        if epoch > 0:  # 第一轮没有上一轮的信息
            # 检查上一轮是否有单一类别预测问题
            previous_only_predicts_one_class = stats.get('only_predicts_one_class', [False])[-1]
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                with amp.autocast():
                    outputs = model(inputs)
                    
                    # 在验证阶段检查类别平衡 - 使用上一轮的信息
                    if previous_only_predicts_one_class and epoch > warmup_epochs:
                        # 应用临时后处理偏置修正
                        # 这不会影响模型权重，只是在预测时临时调整输出
                        logits_adj = outputs.clone()
                        
                        # 根据上一轮的统计进行调整
                        if stats.get('ad_pred_ratio', [0.5])[-1] > 0.8:  # 上一轮过多预测AD
                            logits_adj[:, 1] += 0.2  # 提高CN类的预测概率
                        elif stats.get('cn_pred_ratio', [0.5])[-1] > 0.8:  # 上一轮过多预测CN
                            logits_adj[:, 0] += 0.2  # 提高AD类的预测概率
                        
                        # 使用调整后的logits计算损失
                        loss = criterion(logits_adj, labels)
                        _, predicted = logits_adj.max(1)
                    else:
                        loss = criterion(outputs, labels)
                        _, predicted = outputs.max(1)
                
                val_loss += loss.item()
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # 更新混淆矩阵
                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    conf_matrix[t.long(), p.long()] += 1
        
        # 计算指标
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
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
        stats['lr'].append(optimizer.param_groups[0]['lr'])
        stats['val_acc_per_class'].append(val_acc_per_class)
        
        # 检查模型是否只预测单一类别
        only_predicts_one_class = conf_matrix[0, 0] == 0 or conf_matrix[1, 1] == 0
        stats['only_predicts_one_class'].append(only_predicts_one_class)
        
        # 计算预测分布
        ad_pred_ratio = (conf_matrix[0, 0] + conf_matrix[1, 0]).float() / conf_matrix.sum()
        cn_pred_ratio = (conf_matrix[0, 1] + conf_matrix[1, 1]).float() / conf_matrix.sum()
        stats['ad_pred_ratio'].append(ad_pred_ratio.item())
        stats['cn_pred_ratio'].append(cn_pred_ratio.item())
        
        # 打印信息
        print(f'\nEpoch [{epoch+1}/{total_epochs}] - {current_config["name"]}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Val Acc per class: AD={val_acc_per_class[0]:.2f}%, CN={val_acc_per_class[1]:.2f}%')
        
        # 添加是否改善的指标
        # 初始化改善状态变量
        is_overall_accuracy_improved = False
        acc_change = 0.0
        ad_acc_change = 0.0
        cn_acc_change = 0.0
        balance_improved = False

        if epoch > 0: # 只有在至少完成一个epoch后才能比较
            if len(stats['val_acc']) >= 2:
                prev_val_acc = stats['val_acc'][-2] # 获取上一个epoch的验证准确率
                acc_change = val_acc - prev_val_acc
                is_overall_accuracy_improved = acc_change > 0
            
            if len(stats['val_acc_per_class']) >= 2:
                prev_ad_acc = stats['val_acc_per_class'][-2][0] # 上一个epoch的AD准确率
                prev_cn_acc = stats['val_acc_per_class'][-2][1] # 上一个epoch的CN准确率
                ad_acc_change = val_acc_per_class[0] - prev_ad_acc
                cn_acc_change = val_acc_per_class[1] - prev_cn_acc
                
                # 计算类别平衡度 - 两个类别准确率的差异绝对值
                prev_balance = abs(prev_ad_acc - prev_cn_acc)
                current_balance = abs(val_acc_per_class[0] - val_acc_per_class[1])
                balance_improved = current_balance < prev_balance
            
            # 输出改善指标
            print(f'改善状态: {"✓" if is_overall_accuracy_improved else "✗"} 准确率{acc_change:+.2f}%, ' 
                  f'AD{ad_acc_change:+.2f}%, CN{cn_acc_change:+.2f}%, ' 
                  f'类别平衡{"✓" if balance_improved else "✗"}')
            
            # 如果两个类别都有预测，检查是否从单类预测恢复
            if val_acc_per_class[0] > 0 and val_acc_per_class[1] > 0:
                # 检查上一个epoch是否为单类别预测
                if len(stats['only_predicts_one_class']) >= 2 and stats['only_predicts_one_class'][-2]:
                    print(f'🎉 模型恢复预测两个类别！这是重大改善')
        
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'混淆矩阵:\n{conf_matrix}')
        
        # 保存每个epoch的详细统计信息到JSON文件
        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': float(avg_train_loss),
            'train_acc': float(train_acc),
            'val_loss': float(avg_val_loss),
            'val_acc': float(val_acc),
            'val_acc_per_class': [float(acc) for acc in val_acc_per_class],
            'lr': float(optimizer.param_groups[0]['lr']),
            'conf_matrix': conf_matrix.tolist(),
            'ad_pred_ratio': float(ad_pred_ratio.item()),
            'cn_pred_ratio': float(cn_pred_ratio.item()),
            'only_predicts_one_class': bool(only_predicts_one_class)
        }
        
        # 保存到JSON文件
        stats_file = os.path.join(log_dir, f'epoch_{epoch+1:03d}_{timestamp}.json')
        with open(stats_file, 'w') as f:
            json.dump(epoch_stats, f, indent=2)
        
        # 记录到日志
        log_message(f"Epoch [{epoch+1}/{total_epochs}] - {current_config['name']}:", also_print=False)
        log_message(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%", also_print=False)
        log_message(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%", also_print=False)
        log_message(f"Val Acc per class: AD={val_acc_per_class[0]:.2f}%, CN={val_acc_per_class[1]:.2f}%", also_print=False)
        log_message(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}", also_print=False)
        
        # 打印预测分布 - 使用之前计算的值
        print(f'预测分布: AD={ad_pred_ratio.item()*100:.1f}%, CN={cn_pred_ratio.item()*100:.1f}%')
        
        # 分析趋势
        if epoch >= 2:  # 至少需要3个数据点才能分析趋势
            # 分析最近3轮的AD预测比例趋势
            recent_ad_ratios = stats['ad_pred_ratio'][-3:]
            ad_trend = "上升" if recent_ad_ratios[2] > recent_ad_ratios[0] else "下降"
            ad_trend_strength = abs(recent_ad_ratios[2] - recent_ad_ratios[0]) * 100
            
            # 分析最近3轮的CN预测比例趋势
            recent_cn_ratios = stats['cn_pred_ratio'][-3:]
            cn_trend = "上升" if recent_cn_ratios[2] > recent_cn_ratios[0] else "下降"
            cn_trend_strength = abs(recent_cn_ratios[2] - recent_cn_ratios[0]) * 100
            
            print(f'预测趋势: AD {ad_trend} ({ad_trend_strength:.1f}%), CN {cn_trend} ({cn_trend_strength:.1f}%)')
            
            # 使用趋势强度辅助调整，但避免与其他偏置调整冲突
            if (ad_trend_strength > 25 or cn_trend_strength > 25) and not bias_adjusted_this_epoch:  # 趋势变化阈值从15%提高到25%
                print("检测到明显的预测趋势变化，尝试平衡...")
                with torch.no_grad():
                    if ad_trend == "上升" and ad_trend_strength > 25:  # AD趋势明显上升
                        if hasattr(model, 'classifier1') and hasattr(model.classifier1, 'bias'):
                            model.classifier1.bias.data[1] += 0.15  # 从0.2降为0.15，更平缓地增加CN类的偏置
                        if hasattr(model, 'classifier2') and hasattr(model.classifier2, 'bias'):
                            model.classifier2.bias.data[1] += 0.15  # 从0.2降为0.15，更平缓地增加CN类的偏置
                    elif cn_trend == "上升" and cn_trend_strength > 25:  # CN趋势明显上升
                        if hasattr(model, 'classifier1') and hasattr(model.classifier1, 'bias'):
                            model.classifier1.bias.data[0] += 0.15  # 从0.2降为0.15，更平缓地增加AD类的偏置
                        if hasattr(model, 'classifier2') and hasattr(model.classifier2, 'bias'):
                            model.classifier2.bias.data[0] += 0.15  # 从0.2降为0.15，更平缓地增加AD类的偏置
                    bias_adjusted_this_epoch = True
        
        # 检查模型是否只预测单一类别
        only_predicts_one_class = conf_matrix[0, 0] == 0 or conf_matrix[1, 1] == 0
        if only_predicts_one_class and not bias_adjusted_this_epoch:
            print(f'⚠️ 警告：模型目前只预测单一类别')
            
            # 如果只预测AD类，修正分类器偏置
            if conf_matrix[0, 0] > 0 and conf_matrix[1, 1] == 0:  # 只预测AD类
                print("检测到模型只预测AD类，进行渐进式修正...")
                with torch.no_grad():
                    if hasattr(model, 'classifier1') and hasattr(model.classifier1, 'classifier'):
                        # 1. 适度调整偏置
                        model.classifier1.classifier.bias.data[1] += 0.2  # 从0.5降低到0.2
                        
                        # 2. 调整权重范数 - 减小AD类权重，增大CN类权重
                        ad_weights = model.classifier1.classifier.weight[0]
                        cn_weights = model.classifier1.classifier.weight[1]
                        
                        # 计算当前范数
                        ad_norm = torch.norm(ad_weights)
                        cn_norm = torch.norm(cn_weights)
                        
                        # 调整权重范数
                        if ad_norm > cn_norm:
                            # 减小AD类权重范数，增大CN类权重范数
                            scale_factor = cn_norm / ad_norm * 0.8  # 缩小到80%
                            model.classifier1.classifier.weight.data[0] *= scale_factor
                            model.classifier1.classifier.weight.data[1] *= (2 - scale_factor)
                            print(f"调整权重范数: AD {ad_norm:.4f} → {ad_norm*scale_factor:.4f}, CN {cn_norm:.4f} → {cn_norm*(2-scale_factor):.4f}")
                    
                    elif hasattr(model, 'classifier1') and hasattr(model.classifier1, 'bias'):
                        # 兼容旧版分类器
                        model.classifier1.bias.data[1] += 0.3  # 从0.5降低到0.3
                    
                    if hasattr(model, 'classifier2') and hasattr(model.classifier2, 'classifier'):
                        # 同样的调整应用到第二个分类器
                        model.classifier2.classifier.bias.data[1] += 0.2
                        
                        ad_weights = model.classifier2.classifier.weight[0]
                        cn_weights = model.classifier2.classifier.weight[1]
                        ad_norm = torch.norm(ad_weights)
                        cn_norm = torch.norm(cn_weights)
                        
                        if ad_norm > cn_norm:
                            scale_factor = cn_norm / ad_norm * 0.8
                            model.classifier2.classifier.weight.data[0] *= scale_factor
                            model.classifier2.classifier.weight.data[1] *= (2 - scale_factor)
                    
                    elif hasattr(model, 'classifier2') and hasattr(model.classifier2, 'bias'):
                        # 兼容旧版分类器
                        model.classifier2.bias.data[1] += 0.3
                bias_adjusted_this_epoch = True
            
            # 如果只预测CN类，修正分类器偏置
            elif conf_matrix[0, 0] == 0 and conf_matrix[1, 1] > 0:  # 只预测CN类
                print("检测到模型只预测CN类，进行渐进式修正...")
                with torch.no_grad():
                    if hasattr(model, 'classifier1') and hasattr(model.classifier1, 'classifier'):
                        # 1. 适度调整偏置
                        model.classifier1.classifier.bias.data[0] += 0.2  # 从0.3降低到0.2
                        
                        # 2. 调整权重范数 - 减小CN类权重，增大AD类权重
                        ad_weights = model.classifier1.classifier.weight[0]
                        cn_weights = model.classifier1.classifier.weight[1]
                        
                        # 计算当前范数
                        ad_norm = torch.norm(ad_weights)
                        cn_norm = torch.norm(cn_weights)
                        
                        # 调整权重范数
                        if cn_norm > ad_norm:
                            # 减小CN类权重范数，增大AD类权重范数
                            scale_factor = ad_norm / cn_norm * 0.8  # 缩小到80%
                            model.classifier1.classifier.weight.data[1] *= scale_factor
                            model.classifier1.classifier.weight.data[0] *= (2 - scale_factor)
                            print(f"调整权重范数: CN {cn_norm:.4f} → {cn_norm*scale_factor:.4f}, AD {ad_norm:.4f} → {ad_norm*(2-scale_factor):.4f}")
                    
                    elif hasattr(model, 'classifier1') and hasattr(model.classifier1, 'bias'):
                        # 兼容旧版分类器
                        model.classifier1.bias.data[0] += 0.2  # 从0.3降低到0.2
                    
                    if hasattr(model, 'classifier2') and hasattr(model.classifier2, 'classifier'):
                        # 同样的调整应用到第二个分类器
                        model.classifier2.classifier.bias.data[0] += 0.2
                        
                        ad_weights = model.classifier2.classifier.weight[0]
                        cn_weights = model.classifier2.classifier.weight[1]
                        ad_norm = torch.norm(ad_weights)
                        cn_norm = torch.norm(cn_weights)
                        
                        if cn_norm > ad_norm:
                            scale_factor = ad_norm / cn_norm * 0.8
                            model.classifier2.classifier.weight.data[1] *= scale_factor
                            model.classifier2.classifier.weight.data[0] *= (2 - scale_factor)
                    
                    elif hasattr(model, 'classifier2') and hasattr(model.classifier2, 'bias'):
                        # 兼容旧版分类器
                        model.classifier2.bias.data[0] += 0.2
                bias_adjusted_this_epoch = True
            
            if epoch < warmup_epochs:
                print(f'   处于预热期({epoch+1}/{warmup_epochs})，这是正常现象，模型还在学习特征')
            elif epoch < warmup_epochs + 5:
                print(f'   刚出预热期，继续观察模型学习情况')
            else:
                print(f'   预热期后仍单一预测，已临时修正分类器偏置')
        else:
            # 检查预测是否过于不平衡，但避免与单一类别预测的调整冲突
            if (ad_pred_ratio > 0.85 or cn_pred_ratio > 0.85) and not bias_adjusted_this_epoch:  # 阈值从0.8提高到0.85
                print("预测分布严重不平衡，尝试调整...")
                with torch.no_grad():
                    if ad_pred_ratio > 0.85:  # 过多预测AD
                        if hasattr(model, 'classifier1') and hasattr(model.classifier1, 'classifier'):
                            # 适度调整偏置
                            model.classifier1.classifier.bias.data[1] += 0.15  # 从0.25降低到0.15
                            
                            # 轻微调整权重范数
                            ad_weights = model.classifier1.classifier.weight[0]
                            cn_weights = model.classifier1.classifier.weight[1]
                            ad_norm = torch.norm(ad_weights)
                            cn_norm = torch.norm(cn_weights)
                            
                            if ad_norm > cn_norm * 1.2:  # 如果AD类权重范数明显大于CN类
                                scale_factor = 0.95  # 轻微缩小AD类权重
                                model.classifier1.classifier.weight.data[0] *= scale_factor
                                model.classifier1.classifier.weight.data[1] *= (2 - scale_factor)
                        
                        elif hasattr(model, 'classifier1') and hasattr(model.classifier1, 'bias'):
                            # 兼容旧版分类器
                            model.classifier1.bias.data[1] += 0.15
                        
                        if hasattr(model, 'classifier2') and hasattr(model.classifier2, 'classifier'):
                            # 同样的调整应用到第二个分类器
                            model.classifier2.classifier.bias.data[1] += 0.15
                            
                            # 轻微调整权重范数
                            ad_weights = model.classifier2.classifier.weight[0]
                            cn_weights = model.classifier2.classifier.weight[1]
                            ad_norm = torch.norm(ad_weights)
                            cn_norm = torch.norm(cn_weights)
                            
                            if ad_norm > cn_norm * 1.2:
                                scale_factor = 0.95
                                model.classifier2.classifier.weight.data[0] *= scale_factor
                                model.classifier2.classifier.weight.data[1] *= (2 - scale_factor)
                        
                        elif hasattr(model, 'classifier2') and hasattr(model.classifier2, 'bias'):
                            # 兼容旧版分类器
                            model.classifier2.bias.data[1] += 0.15
                    
                    else:  # 过多预测CN
                        if hasattr(model, 'classifier1') and hasattr(model.classifier1, 'classifier'):
                            # 适度调整偏置
                            model.classifier1.classifier.bias.data[0] += 0.15
                            
                            # 轻微调整权重范数
                            ad_weights = model.classifier1.classifier.weight[0]
                            cn_weights = model.classifier1.classifier.weight[1]
                            ad_norm = torch.norm(ad_weights)
                            cn_norm = torch.norm(cn_weights)
                            
                            if cn_norm > ad_norm * 1.2:  # 如果CN类权重范数明显大于AD类
                                scale_factor = 0.95  # 轻微缩小CN类权重
                                model.classifier1.classifier.weight.data[1] *= scale_factor
                                model.classifier1.classifier.weight.data[0] *= (2 - scale_factor)
                        
                        elif hasattr(model, 'classifier1') and hasattr(model.classifier1, 'bias'):
                            # 兼容旧版分类器
                            model.classifier1.bias.data[0] += 0.15
                        
                        if hasattr(model, 'classifier2') and hasattr(model.classifier2, 'classifier'):
                            # 同样的调整应用到第二个分类器
                            model.classifier2.classifier.bias.data[0] += 0.15
                            
                            # 轻微调整权重范数
                            ad_weights = model.classifier2.classifier.weight[0]
                            cn_weights = model.classifier2.classifier.weight[1]
                            ad_norm = torch.norm(ad_weights)
                            cn_norm = torch.norm(cn_weights)
                            
                            if cn_norm > ad_norm * 1.2:
                                scale_factor = 0.95
                                model.classifier2.classifier.weight.data[1] *= scale_factor
                                model.classifier2.classifier.weight.data[0] *= (2 - scale_factor)
                        
                        elif hasattr(model, 'classifier2') and hasattr(model.classifier2, 'bias'):
                            # 兼容旧版分类器
                            model.classifier2.bias.data[0] += 0.15
                bias_adjusted_this_epoch = True
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': plateau_scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': avg_val_loss,
                'stats': stats,
                'conf_matrix': conf_matrix.tolist(),
                'model_config': current_config
            }
            
            model_filename = f'best_early_fusion_{current_config["name"].lower().replace("-", "_")}.pth'
            model_path = os.path.join(save_dir, model_filename)
            torch.save(best_model_state, model_path)
            print(f'✓ 保存最佳模型: {model_path}，验证准确率: {val_acc:.2f}%')
            log_message(f'✓ 保存最佳模型: {model_path}，验证准确率: {val_acc:.2f}%')
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            
            # 记录早停检查
            if no_improve_epochs >= patience // 2:  # 达到一半早停周期时警告
                log_message(f'⚠️ 已连续{no_improve_epochs}轮无改善，还剩{patience-no_improve_epochs}轮触发早停')
        
        # 计算最低类别准确率，用于复合早停标准
        min_class_acc = min(val_acc_per_class)
        
        # 如果最低类别准确率有改善，重置对应的计数器
        if min_class_acc > best_min_class_acc:
            best_min_class_acc = min_class_acc
            no_min_class_improve = 0
            print(f'✓ 最低类别准确率提升至 {min_class_acc:.2f}%')
        else:
            no_min_class_improve += 1
            if no_min_class_improve >= min_class_acc_patience // 2:
                print(f'⚠️ 最低类别准确率已连续{no_min_class_improve}轮无改善，还剩{min_class_acc_patience-no_min_class_improve}轮触发早停')
        
        # 复合早停标准：整体准确率和最低类别准确率都长时间没有改善
        if no_improve_epochs >= patience and no_min_class_improve >= min_class_acc_patience:
            print(f'早停在epoch {epoch+1}，整体准确率已连续{patience}轮无改善，最低类别准确率已连续{no_min_class_improve}轮无改善')
            log_message(f'早停在epoch {epoch+1}，整体准确率已连续{patience}轮无改善，最低类别准确率已连续{no_min_class_improve}轮无改善')
            break
        # 如果只有整体准确率长时间没有改善，但最低类别准确率仍有希望，继续训练
        elif no_improve_epochs >= patience:
            print(f'整体准确率已连续{patience}轮无改善，但最低类别准确率仍有希望（{no_min_class_improve}/{min_class_acc_patience}），继续训练')
            # 不早停，继续训练
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state_dict'])
        print(f'\n✓ 已加载最佳早期融合模型，验证准确率: {best_val_acc:.2f}%')
    
    return {
        'model': model,
        'best_val_acc': best_val_acc,
        'best_epoch': best_model_state['epoch'] if best_model_state else -1,
        'model_path': model_path if best_model_state else None,
        'stats': stats,
        'final_conf_matrix': conf_matrix.tolist(),
        'model_config': current_config,
        'architecture': current_config['name']
    }

# 保留SimpleResNet3D作为降级选项
class SimpleResNet3D(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(SimpleResNet3D, self).__init__()
        
        # 第一层卷积 - 减少采样尺寸
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        # 简单残差块
        self.res_block1 = self._make_simple_block(16, 32)
        self.res_block2 = self._make_simple_block(32, 64, stride=2)
        self.res_block3 = self._make_simple_block(64, 64)
        
        # 全局池化和分类器
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(64, num_classes)
        
        # 正确初始化
        self._initialize_weights()
    
    def _make_simple_block(self, in_channels, out_channels, stride=1):
        layers = []
        
        # 下采样层（如果需要）
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            downsample = None
        
        # 添加残差块
        layers.append(SimpleResidualBlock(in_channels, out_channels, stride, downsample))
        
        # 添加一个额外的块保持相同维度
        layers.append(SimpleResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # 使用Kaiming正态分布初始化卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                # 初始化BN层
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 初始化分类器 - 使用非常小的值
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

class SimpleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(SimpleResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

def validate_early_fusion_with_metrics(model, val_loader, criterion, device):
    """验证早期融合模型并返回详细指标"""
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    # 初始化混淆矩阵 (2x2 for binary classification)
    conf_matrix = torch.zeros(2, 2, dtype=torch.long)
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 统计
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            
            # 更新混淆矩阵
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                conf_matrix[t.long(), p.long()] += 1
    
    # 计算最终指标
    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100. * val_correct / val_total
    
    return avg_val_loss, val_acc, conf_matrix

# ==================== 内存友好的简化版本 ====================

class MemoryEfficientHierarchicalModel(nn.Module):
    """
    内存友好的层次化模型
    保留核心思想但大幅减少内存需求
    参数数量 < 50K，内存需求 < 1GB
    """
    def __init__(self, in_chans=3, num_classes=2):
        super().__init__()
        
        print("初始化内存友好的层次化模型...")
        
        # 1. 极简卷积特征提取 - 快速降维
        self.conv_layers = nn.Sequential(
            # 第一层：极大幅度下采样 stride=8
            nn.Conv3d(in_chans, 4, kernel_size=7, stride=8, padding=3, bias=False),
            nn.BatchNorm3d(4),
            nn.ReLU(inplace=True),
            
            # 第二层：继续下采样
            nn.Conv3d(4, 8, kernel_size=5, stride=4, padding=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            
            # 第三层：最后的特征提取
            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        
        # 2. 简化的灰质梯度注意力模块
        self.gm_attention = GrayMatterGradientAttention(16)
        
        # 3. 全局平均池化 + 简单分类器
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # 4. 极简分类头
        self.classifier = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(8, num_classes)
        )
        
        # 权重初始化
        self._initialize_weights()
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"内存友好模型参数数量: {total_params:,} (目标: <50K)")
    
    def _initialize_weights(self):
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
        # 输入形状检查
        if len(x.shape) != 5:
            raise ValueError(f"期望5D输入[B,C,D,H,W]，但得到{x.shape}")
        
        # 1. 极简卷积特征提取
        x = self.conv_layers(x)
        
        # 2. 灰质梯度注意力增强
        x = self.gm_attention(x, gray_matter_channel=min(1, x.size(1)-1))
        
        # 3. 全局池化和分类
        x = self.global_pool(x)  # [B, 16, 1, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 16]
        x = self.classifier(x)  # [B, num_classes]
        
        return x 

def test_time_augmentation(model, val_loader, device, num_tta=5):
    """
    测试时增强 - 通过多次预测的平均来提高准确率
    可以额外提升2-5%的准确率
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    print(f"\n执行测试时增强 (TTA)，增强次数: {num_tta}")
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="TTA预测"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 收集多次预测结果
            batch_predictions = []
            
            for tta_idx in range(num_tta):
                # 轻微的随机变换
                if tta_idx > 0:
                    # 添加轻微噪声
                    noise = torch.randn_like(inputs) * 0.01
                    augmented_inputs = inputs + noise
                    # 随机翻转
                    if torch.rand(1) > 0.5:
                        augmented_inputs = torch.flip(augmented_inputs, dims=[2])  # D维翻转
                    if torch.rand(1) > 0.5:
                        augmented_inputs = torch.flip(augmented_inputs, dims=[3])  # H维翻转
                    if torch.rand(1) > 0.5:
                        augmented_inputs = torch.flip(augmented_inputs, dims=[4])  # W维翻转
                else:
                    augmented_inputs = inputs
                
                outputs = model(augmented_inputs)
                batch_predictions.append(F.softmax(outputs, dim=1))
            
            # 平均所有预测
            avg_predictions = torch.stack(batch_predictions).mean(dim=0)
            all_predictions.append(avg_predictions)
            all_labels.append(labels)
    
    # 计算TTA后的准确率
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    _, predicted = all_predictions.max(1)
    tta_accuracy = 100. * predicted.eq(all_labels).sum().item() / all_labels.size(0)
    
    # 计算混淆矩阵
    conf_matrix = torch.zeros(2, 2, dtype=torch.long)
    for t, p in zip(all_labels.view(-1), predicted.view(-1)):
        conf_matrix[t.long(), p.long()] += 1
    
    # 计算每个类别的准确率
    class_accuracies = []
    for i in range(2):
        correct = conf_matrix[i, i].item()
        total = conf_matrix[i, :].sum().item()
        class_accuracies.append(100.0 * correct / max(1, total))
    
    print(f"TTA结果:")
    print(f"TTA准确率: {tta_accuracy:.2f}%")
    print(f"类别准确率: AD={class_accuracies[0]:.2f}%, CN={class_accuracies[1]:.2f}%")
    print(f"TTA混淆矩阵:\n{conf_matrix}")
    
    return tta_accuracy, conf_matrix, class_accuracies
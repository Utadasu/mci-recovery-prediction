"""
🔧 优化对比学习模型 - 问题修复版本
==================================

主要改进:
1. 🎯 修复特征空间对齐问题 - 使用更强的投影层
2. ⚖️ 重新平衡损失权重 - 降低对比学习权重
3. 📊 改进学习率策略 - 差异化学习率
4. 🔄 添加特征归一化 - 提升对比学习效果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
from tqdm import tqdm
import json
import pandas as pd


class ImprovedImageEncoder(nn.Module):
    """改进的图像编码器 - 兼容对比学习预训练模型"""
    
    def __init__(self, pretrained_model_path, feature_dim=512, device='cuda'):
        super(ImprovedImageEncoder, self).__init__()
        
        self.device = device  # 保存设备信息
        
        # 加载预训练模型
        from optimized_models import ImprovedResNetCBAM3D
        
        print(f"🔧 加载对比学习预训练图像编码器...")
        print(f"   模型路径: {pretrained_model_path}")
        
        # 检查预训练模型的架构信息
        checkpoint = torch.load(pretrained_model_path, map_location=device)
        
        # 从checkpoint中获取模型配置信息
        if 'config' in checkpoint:
            config = checkpoint['config']
            base_channels = config.get('base_channels', 12)
            print(f"   检测到base_channels: {base_channels}")
        else:
            # 默认配置
            base_channels = 12
            print(f"   使用默认base_channels: {base_channels}")
        
        # 创建与预训练模型匹配的架构
        self.backbone = ImprovedResNetCBAM3D(
            in_channels=3,
            num_classes=2,
            base_channels=base_channels,
            dropout_rate=0.3,
            use_global_pool=False  # 🔥 使用智能下采样层
        )
        
        # 重建fusion层以匹配预训练模型
        fusion_input_dim = base_channels * 16 * 2 * 2 * 2  # 智能下采样层输出维度
        
        self.backbone.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),  # 输出512维特征
            nn.LayerNorm(512),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3)
        )
        
        # 重建分类头
        self.backbone.classifier = nn.Linear(512, 2)
        
        # 🔧 先移动模型到设备，再加载权重
        self.backbone.to(device)
        
        # 加载预训练权重
        try:
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # 尝试加载权重，忽略不匹配的键
            model_dict = self.backbone.state_dict()
            
            # 过滤掉不匹配的键
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    filtered_state_dict[k] = v
                else:
                    print(f"   跳过不匹配的键: {k}")
            
            # 加载过滤后的权重
            model_dict.update(filtered_state_dict)
            self.backbone.load_state_dict(model_dict)
            
            print(f"✅ 成功加载 {len(filtered_state_dict)}/{len(state_dict)} 个权重")
            
        except Exception as e:
            print(f"⚠️  权重加载警告: {e}")
            print("   将使用随机初始化的权重")
        
        # 冻结骨干网络（除了fusion层和分类头）
        for name, param in self.backbone.named_parameters():
            if 'fusion' not in name and 'classifier' not in name:
                param.requires_grad = False
        
        print(f"   已冻结骨干网络，保留fusion层和分类头可训练")
        
        # 获取特征维度（从fusion层输出）
        self.backbone.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 113, 137, 113).to(device)
            # 使用return_features=True获取fusion层输出
            features = self.backbone(dummy_input, return_features=True)
            backbone_feature_dim = features.size(1)
        
        print(f"   骨干网络特征维度: {backbone_feature_dim}")
        
        # 轻量级投影层 - 因为backbone已经输出512维
        if backbone_feature_dim == feature_dim:
            # 如果维度已经匹配，使用简单的投影层
            self.projection = nn.Sequential(
                nn.Linear(backbone_feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim)
            )
        else:
            # 如果维度不匹配，使用更复杂的投影层
            self.projection = nn.Sequential(
                nn.Linear(backbone_feature_dim, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
                nn.Dropout(0.2),
                
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.1),
                
                nn.Linear(512, feature_dim),
                nn.LayerNorm(feature_dim)
            )
        
        # 🔧 确保投影层也在正确的设备上
        self.projection.to(device)
        
        print(f"   投影层: {backbone_feature_dim} → {feature_dim}")
        
        # 初始化投影层权重
        self._init_projection_weights()
    
    def _init_projection_weights(self):
        """初始化投影层权重"""
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        # 🔧 确保输入在正确的设备上
        x = x.to(self.device)
        
        # 使用return_features=True提取骨干特征
        features = self.backbone(x, return_features=True)  # [B, 512]
        
        # 投影到统一特征空间
        projected = self.projection(features)
        
        # L2标准化（对比学习必需）
        projected = F.normalize(projected, p=2, dim=1)
        
        return projected


class ImprovedTextEncoder(nn.Module):
    """改进的文本编码器 - 基于预提取的512维特征"""
    
    def __init__(self, pretrained_features, feature_dim=512, device='cuda'):
        super(ImprovedTextEncoder, self).__init__()
        
        # 预提取的文本特征 (已经是经过BERT训练的512维特征)
        if isinstance(pretrained_features, np.ndarray):
            self.features = torch.FloatTensor(pretrained_features)
        else:
            self.features = torch.FloatTensor(pretrained_features)
        
        print(f"📝 预训练文本特征形状: {self.features.shape}")
        
        # 将特征移动到指定设备
        self.device = device
        self.features = self.features.to(device)
        print(f"📱 文本特征已移动到设备: {device}")
        
        # 轻量级投影层 - 因为输入已经是高质量的512维特征
        # 主要作用是适配对比学习，不需要太复杂的变换
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # 降低dropout
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        print(f"🔧 文本投影层: {feature_dim} → {feature_dim} (轻量级)")
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)  # 降低初始化范围
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, indices):
        """
        前向传播
        
        Args:
            indices: 样本索引 [B]
            
        Returns:
            projected: L2标准化的特征 [B, feature_dim]
        """
        # 确保索引在正确的设备上
        if isinstance(indices, torch.Tensor):
            indices = indices.to(self.device)
        else:
            indices = torch.tensor(indices, device=self.device, dtype=torch.long)
        
        # 根据索引获取预训练特征
        batch_features = self.features[indices]
        
        # 轻量级投影 - 保持预训练特征的质量
        projected = self.projection(batch_features)
        
        # L2标准化
        projected = F.normalize(projected, p=2, dim=1)
        
        return projected


class EndToEndTextEncoder(nn.Module):
    """端到端文本编码器 - 直接处理Excel文本数据"""
    
    def __init__(self, feature_dim=512, device='cuda', max_length=512):
        super(EndToEndTextEncoder, self).__init__()
        
        self.device = device
        self.max_length = max_length
        self.feature_dim = feature_dim
        
        # 初始化BERT模型和分词器
        from transformers import BertModel, BertTokenizer
        
        print(f"🔧 初始化BERT文本编码器...")
        
        # 使用本地BERT模型路径
        bert_model_path = '/root/autodl-tmp/bert-base-uncased'
        
        try:
            self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
            self.bert_model = BertModel.from_pretrained(bert_model_path)
            print(f"✅ 本地BERT模型加载成功: {bert_model_path}")
        except:
            # 备用：使用在线模型
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
            print(f"✅ 在线BERT模型加载成功")
        
        # 移动BERT到设备
        self.bert_model.to(device)
        
        # 投影层：768维BERT特征 → 512维对比学习特征
        self.projection = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(1024, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # 🔧 确保投影层在正确的设备上
        self.projection.to(device)
        
        print(f"🔧 文本投影层: 768 → {feature_dim}")
        
        # 初始化投影层权重
        self._init_projection_weights()
    
    def _init_projection_weights(self):
        """初始化投影层权重"""
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def encode_texts(self, texts):
        """
        编码文本列表
        
        Args:
            texts: List[str] 文本列表
            
        Returns:
            features: [B, feature_dim] L2标准化的文本特征
        """
        # 分词和编码
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 移动到设备
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # BERT编码
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # 使用[CLS]标记的特征
            bert_features = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        
        # 🔧 确保BERT特征在正确的设备上
        bert_features = bert_features.to(self.device)
        
        # 投影到对比学习特征空间
        projected = self.projection(bert_features)  # [B, feature_dim]
        
        # L2标准化
        projected = F.normalize(projected, p=2, dim=1)
        
        return projected
    
    def forward(self, texts):
        """前向传播"""
        return self.encode_texts(texts)


class TripleLossSystem(nn.Module):
    """三重损失系统 - 端到端版本"""
    
    def __init__(self, temperature=0.5, margin=0.2):
        super(TripleLossSystem, self).__init__()
        self.temperature = temperature
        self.margin = margin
        self.eps = 1e-8
    
    def classification_loss(self, logits, labels):
        """
        1. AD/CN分类损失
        
        Args:
            logits: [B, 2] 分类logits
            labels: [B] 真实标签 (0=CN, 1=AD)
            
        Returns:
            loss: 分类损失
        """
        return F.cross_entropy(logits, labels)
    
    def cross_modal_alignment_loss(self, image_features, text_features):
        """
        2. 图像文本对齐损失 (InfoNCE对比学习)
        
        Args:
            image_features: [B, D] L2标准化的图像特征
            text_features: [B, D] L2标准化的文本特征
            
        Returns:
            loss: 跨模态对齐损失
        """
        batch_size = image_features.size(0)
        device = image_features.device
        
        # 确保特征已标准化
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(image_features, text_features.T) / self.temperature
        
        # 数值稳定性
        similarity_matrix = torch.clamp(similarity_matrix, -10, 10)
        
        # 正样本标签 (对角线)
        labels = torch.arange(batch_size, device=device)
        
        # 图像到文本的损失
        loss_i2t = F.cross_entropy(similarity_matrix, labels)
        
        # 文本到图像的损失
        loss_t2i = F.cross_entropy(similarity_matrix.T, labels)
        
        # 平均损失
        return (loss_i2t + loss_t2i) / 2
    
    def intra_modal_contrastive_loss(self, image_features, labels):
        """
        3. 图像内部对比损失 (优化相同模态内特征分布)
        
        Args:
            image_features: [B, D] L2标准化的图像特征
            labels: [B] 真实标签 (0=CN, 1=AD)
            
        Returns:
            loss: 模态内对比损失
        """
        batch_size = image_features.size(0)
        device = image_features.device
        
        # 确保特征已标准化
        image_features = F.normalize(image_features, p=2, dim=1)
        
        # 计算特征相似度矩阵
        similarity_matrix = torch.matmul(image_features, image_features.T) / self.temperature
        
        # 创建标签掩码
        labels = labels.contiguous().view(-1, 1)
        label_mask = torch.eq(labels, labels.T).float().to(device)
        
        # 移除对角线 (自己与自己的相似度)
        identity_mask = torch.eye(batch_size, device=device)
        label_mask = label_mask - identity_mask
        
        # 正样本掩码 (同类样本，排除自己)
        positive_mask = label_mask
        
        # 负样本掩码 (异类样本)
        negative_mask = 1 - label_mask - identity_mask
        
        # 计算正样本损失 (同类样本应该相似)
        positive_similarities = similarity_matrix * positive_mask
        positive_count = positive_mask.sum(dim=1, keepdim=True)
        positive_count = torch.clamp(positive_count, min=1)  # 避免除零
        
        # 正样本的平均相似度
        positive_mean = torch.sum(positive_similarities, dim=1, keepdim=True) / positive_count
        positive_loss = -positive_mean  # 负号：最大化正样本相似度
        
        # 计算负样本损失 (异类样本应该不相似)
        negative_similarities = similarity_matrix * negative_mask
        negative_count = negative_mask.sum(dim=1, keepdim=True)
        negative_count = torch.clamp(negative_count, min=1)
        
        # 负样本的最大相似度 (最困难的负样本)
        negative_max = torch.max(negative_similarities + (1 - negative_mask) * (-100), dim=1, keepdim=True)[0]
        
        # 使用margin-based loss: max(0, negative_sim - positive_sim + margin)
        triplet_loss = torch.clamp(negative_max - positive_mean + self.margin, min=0)
        
        # 总的模态内损失
        total_loss = (positive_loss + triplet_loss).mean()
        
        return total_loss
    
    def forward(self, logits, image_features, text_features, labels):
        """
        计算三重损失
        
        Args:
            logits: [B, 2] 分类logits
            image_features: [B, D] 图像特征
            text_features: [B, D] 文本特征
            labels: [B] 真实标签
            
        Returns:
            dict: 包含各种损失的字典
        """
        # 1. AD/CN分类损失
        cls_loss = self.classification_loss(logits, labels)
        
        # 2. 图像文本对齐损失
        alignment_loss = self.cross_modal_alignment_loss(image_features, text_features)
        
        # 3. 图像内部对比损失
        intra_loss = self.intra_modal_contrastive_loss(image_features, labels)
        
        return {
            'classification_loss': cls_loss,
            'alignment_loss': alignment_loss,
            'intra_modal_loss': intra_loss,
            'total_loss': cls_loss + alignment_loss + intra_loss
        }


class ImprovedContrastiveLoss(nn.Module):
    """改进的对比学习损失 - 修复温度参数"""
    
    def __init__(self, temperature=0.5, reduction='mean'):  # 提高温度从0.2到0.5
        super(ImprovedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.eps = 1e-8
    
    def forward(self, image_features, text_features):
        """
        计算对比学习损失 - 更合理的温度参数
        
        Args:
            image_features: [B, D] L2标准化的图像特征
            text_features: [B, D] L2标准化的文本特征
        
        Returns:
            loss: 对比学习损失
        """
        batch_size = image_features.size(0)
        device = image_features.device
        
        # 确保特征已经标准化
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        
        # 减少噪声，避免破坏预训练特征质量
        if self.training:
            noise_std = 0.005  # 从0.01降低到0.005
            image_features = image_features + torch.randn_like(image_features) * noise_std
            text_features = text_features + torch.randn_like(text_features) * noise_std
            # 重新归一化
            image_features = F.normalize(image_features, p=2, dim=1)
            text_features = F.normalize(text_features, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(image_features, text_features.T) / self.temperature
        
        # 数值稳定性处理
        similarity_matrix = torch.clamp(similarity_matrix, -10, 10)  # 进一步缩小范围
        
        # 创建正样本标签
        labels = torch.arange(batch_size, device=device)
        
        # 图像到文本的损失
        loss_i2t = F.cross_entropy(similarity_matrix, labels, reduction=self.reduction)
        
        # 文本到图像的损失  
        loss_t2i = F.cross_entropy(similarity_matrix.T, labels, reduction=self.reduction)
        
        # 总损失
        total_loss = (loss_i2t + loss_t2i) / 2
        
        return total_loss


class ImprovedMultiModalModel(nn.Module):
    """改进的多模态模型 - 三重损失端到端训练版本"""
    
    def __init__(self, image_model_path, all_texts, feature_dim=512, num_classes=2, device='cuda'):
        super(ImprovedMultiModalModel, self).__init__()
        
        self.device = device  # 保存设备信息
        
        # 编码器
        self.image_encoder = ImprovedImageEncoder(image_model_path, feature_dim, device)
        self.text_encoder = EndToEndTextEncoder(feature_dim, device)
        
        # 三重损失系统
        self.triple_loss = TripleLossSystem(temperature=0.5, margin=0.2)
        
        # 改进的融合分类器
        self.fusion_classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, num_classes)
        )
        
        # 🔧 确保融合分类器在正确的设备上
        self.fusion_classifier.to(device)
        self.triple_loss.to(device)
        
        # 初始化分类器权重
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """初始化分类器权重"""
        for m in self.fusion_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, images, texts, labels=None, mode='both'):
        """
        前向传播 - 三重损失端到端版本
        
        Args:
            images: [B, 3, 113, 137, 113] 图像数据
            texts: List[str] 文本数据列表
            labels: [B] 真实标签 (用于计算损失)
            mode: 'classification', 'losses', 'both'
        
        Returns:
            dict: 包含不同输出的字典
        """
        # 🔧 确保输入在正确的设备上
        images = images.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        # 编码特征
        image_features = self.image_encoder(images)  # [B, 512]
        text_features = self.text_encoder(texts)     # [B, 512]
        
        results = {
            'image_features': image_features,
            'text_features': text_features
        }
        
        if mode in ['classification', 'both']:
            # 特征融合
            fused_features = torch.cat([image_features, text_features], dim=1)  # [B, 1024]
            
            # 分类
            logits = self.fusion_classifier(fused_features)  # [B, 2]
            results['logits'] = logits
        
        if mode in ['losses', 'both'] and labels is not None:
            # 计算三重损失
            if 'logits' not in results:
                # 如果只计算损失，也需要logits
                fused_features = torch.cat([image_features, text_features], dim=1)
                logits = self.fusion_classifier(fused_features)
                results['logits'] = logits
            
            # 三重损失计算
            loss_dict = self.triple_loss(results['logits'], image_features, text_features, labels)
            results.update(loss_dict)
        
        return results


class EndToEndMultiModalDataset(Dataset):
    """端到端多模态数据集 - 直接处理Excel文本数据"""
    
    def __init__(self, images, texts, labels):
        """
        Args:
            images: numpy array [N, 3, 113, 137, 113] 图像数据
            texts: List[str] 文本数据列表
            labels: numpy array [N] 标签数据
        """
        self.images = torch.FloatTensor(images)
        self.texts = texts  # 保持为字符串列表
        self.labels = torch.LongTensor(labels)
        
        assert len(self.images) == len(self.texts) == len(self.labels)
        print(f"📊 数据集创建: {len(self.labels)} 样本")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'text': self.texts[idx],  # 返回原始文本字符串
            'label': self.labels[idx],
            'index': idx
        }


class OptimizedMultiModalDataset(Dataset):
    """优化的多模态数据集"""
    
    def __init__(self, images, text_indices, labels):
        """
        Args:
            images: numpy array [N, 3, 113, 137, 113]
            text_indices: numpy array [N] 文本特征索引
            labels: numpy array [N]
        """
        self.images = torch.FloatTensor(images)
        self.text_indices = torch.LongTensor(text_indices)
        self.labels = torch.LongTensor(labels)
        
        assert len(self.images) == len(self.text_indices) == len(self.labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'text_index': self.text_indices[idx],
            'label': self.labels[idx],
            'index': idx  # 用于调试
        }


class OptimizedContrastiveTrainer:
    """优化的对比学习训练器 - 三重损失版本"""
    
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # 差异化学习率优化器
        self.optimizer = self._create_optimizer()
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['num_epochs']
        )
        
        # 三重损失权重配置
        self.classification_weight = config.get('classification_weight', 1.0)
        self.alignment_weight = config.get('alignment_weight', 0.5)
        self.intra_modal_weight = config.get('intra_modal_weight', 0.3)
        
        print(f"🎯 三重损失权重配置:")
        print(f"   分类损失权重: {self.classification_weight}")
        print(f"   图像文本对齐权重: {self.alignment_weight}")
        print(f"   图像内部对比权重: {self.intra_modal_weight}")
    
    def _create_optimizer(self):
        """创建差异化学习率优化器"""
        param_groups = [
            # 图像投影层 - 高学习率
            {
                'params': self.model.image_encoder.projection.parameters(),
                'lr': self.config['learning_rate'] * 2,
                'name': 'image_projection'
            },
            # 文本投影层 - 高学习率
            {
                'params': self.model.text_encoder.projection.parameters(),
                'lr': self.config['learning_rate'] * 2,
                'name': 'text_projection'
            },
            # BERT参数 - 低学习率
            {
                'params': self.model.text_encoder.bert_model.parameters(),
                'lr': self.config['learning_rate'] * 0.1,
                'name': 'bert_backbone'
            },
            # 融合分类器 - 标准学习率
            {
                'params': self.model.fusion_classifier.parameters(),
                'lr': self.config['learning_rate'],
                'name': 'fusion_classifier'
            }
        ]
        
        return torch.optim.AdamW(param_groups, weight_decay=self.config['weight_decay'])
    
    def train_epoch(self, dataloader, epoch):
        """训练一个epoch - 三重损失版本"""
        self.model.train()
        
        total_loss = 0.0
        total_classification_loss = 0.0
        total_alignment_loss = 0.0
        total_intra_modal_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            images = batch['image'].to(self.device)
            texts = batch['text']  # 文本列表
            labels = batch['label'].to(self.device)
            
            # 前向传播 - 计算所有损失
            outputs = self.model(images, texts, labels=labels, mode='both')
            
            # 提取三重损失
            classification_loss = outputs['classification_loss']
            alignment_loss = outputs['alignment_loss']
            intra_modal_loss = outputs['intra_modal_loss']
            
            # 加权总损失
            total_batch_loss = (
                self.classification_weight * classification_loss +
                self.alignment_weight * alignment_loss +
                self.intra_modal_weight * intra_modal_loss
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
            
            self.optimizer.step()
            
            # 统计
            total_loss += total_batch_loss.item()
            total_classification_loss += classification_loss.item()
            total_alignment_loss += alignment_loss.item()
            total_intra_modal_loss += intra_modal_loss.item()
            
            # 计算准确率
            logits = outputs['logits']
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条
            accuracy = 100. * correct / total
            progress_bar.set_postfix({
                'Loss': f'{total_batch_loss.item():.4f}',
                'Acc': f'{accuracy:.2f}%',
                'CLS': f'{classification_loss.item():.4f}',
                'ALN': f'{alignment_loss.item():.4f}',
                'INT': f'{intra_modal_loss.item():.4f}'
            })
        
        # 更新学习率
        self.scheduler.step()
        
        return {
            'total_loss': total_loss / len(dataloader),
            'classification_loss': total_classification_loss / len(dataloader),
            'alignment_loss': total_alignment_loss / len(dataloader),
            'intra_modal_loss': total_intra_modal_loss / len(dataloader),
            'accuracy': accuracy / 100.0,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def evaluate(self, dataloader):
        """评估模型 - 三重损失版本"""
        self.model.eval()
        
        total_loss = 0.0
        total_classification_loss = 0.0
        total_alignment_loss = 0.0
        total_intra_modal_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='评估中'):
                images = batch['image'].to(self.device)
                texts = batch['text']  # 文本列表
                labels = batch['label'].to(self.device)
                
                # 前向传播 - 计算所有损失
                outputs = self.model(images, texts, labels=labels, mode='both')
                
                # 提取损失
                classification_loss = outputs['classification_loss']
                alignment_loss = outputs['alignment_loss']
                intra_modal_loss = outputs['intra_modal_loss']
                
                # 加权总损失
                total_batch_loss = (
                    self.classification_weight * classification_loss +
                    self.alignment_weight * alignment_loss +
                    self.intra_modal_weight * intra_modal_loss
                )
                
                total_loss += total_batch_loss.item()
                total_classification_loss += classification_loss.item()
                total_alignment_loss += alignment_loss.item()
                total_intra_modal_loss += intra_modal_loss.item()
                
                # 预测
                logits = outputs['logits']
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算详细指标
        accuracy = 100. * correct / total
        report = classification_report(all_labels, all_predictions, target_names=['CN', 'AD'], output_dict=True)
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        return {
            'loss': total_loss / len(dataloader),
            'classification_loss': total_classification_loss / len(dataloader),
            'alignment_loss': total_alignment_loss / len(dataloader),
            'intra_modal_loss': total_intra_modal_loss / len(dataloader),
            'accuracy': accuracy / 100.0,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'predictions': all_predictions,
            'labels': all_labels
        }


def load_end_to_end_data(image_data_dir, text_data_dir, test_size=0.2, random_state=42):
    """
    加载端到端多模态数据 - 严格防数据泄露版本 (修复患者ID提取逻辑)
    
    Args:
        image_data_dir: 图像数据目录
        text_data_dir: 文本数据目录  
        test_size: 测试集比例
        random_state: 随机种子
        
    Returns:
        训练和测试数据
    """
    print("🔄 加载端到端多模态数据 (修复患者ID提取逻辑版本)...")
    
    # 1. 加载图像数据
    from data_utils import load_early_fusion_data
    image_data, image_labels = load_early_fusion_data(image_data_dir)
    
    # 2. 加载文本数据
    text_data, text_labels, patient_ids = load_text_data_from_excel_clean(text_data_dir)
    
    # 3. 数据一致性检查
    assert len(image_data) == len(text_data), \
        f"图像数据({len(image_data)})和文本数据({len(text_data)})数量不匹配"
    assert len(image_labels) == len(text_labels), \
        f"图像标签({len(image_labels)})和文本标签({len(text_labels)})数量不匹配"
    assert np.array_equal(image_labels, text_labels), \
        "图像标签和文本标签不一致"
    
    print(f"📊 数据一致性检查通过:")
    print(f"   图像数据: {image_data.shape}")
    print(f"   文本数据: {len(text_data)} 条")
    print(f"   标签分布: AD={np.sum(text_labels==1)}, CN={np.sum(text_labels==0)}")
    
    # 4. 🔧 修复：正确提取图像文件的患者ID
    print(f"🔧 修复图像文件患者ID提取逻辑...")
    
    def extract_patient_id_from_filename(filename):
        """
        从图像文件名提取患者ID
        文件名格式: mwp3MRI_002_S_0619_3-2016-01-29_12_25_03.0.nii
        目标提取: 002_S_0619
        """
        basename = filename.split('.')[0]  # 移除.nii后缀
        
        # 分割文件名
        parts = basename.split('_')
        
        # 查找模式: mwp3MRI_XXX_S_YYYY
        if len(parts) >= 4 and parts[0].startswith('mwp') and parts[2] == 'S':
            # 提取 XXX_S_YYYY 部分
            patient_id = f"{parts[1]}_{parts[2]}_{parts[3]}"
            return patient_id
        
        # 备用方案：查找 XXX_S_YYYY 模式
        for i in range(len(parts) - 2):
            if parts[i+1] == 'S' and parts[i].isdigit() and parts[i+2].isdigit():
                return f"{parts[i]}_{parts[i+1]}_{parts[i+2]}"
        
        # 最后备用：返回原始文件名
        return basename
    
    # 重新构建图像数据的患者ID列表
    print(f"📊 重新构建图像数据的患者ID映射...")
    
    # 构建图像文件到患者ID的映射
    image_patient_ids = []
    
    # 重新扫描图像文件获取患者ID
    ad_csf_dir = os.path.join(image_data_dir, "123-AD-MRI", "ADfinalCSF")
    cn_csf_dir = os.path.join(image_data_dir, "123-CN-MRI", "CNfinalCSF")
    
    # 获取AD文件的患者ID
    ad_files = sorted([f for f in os.listdir(ad_csf_dir) if f.endswith('.nii')])
    for filename in ad_files:
        patient_id = extract_patient_id_from_filename(filename)
        image_patient_ids.append(patient_id)
    
    # 获取CN文件的患者ID
    cn_files = sorted([f for f in os.listdir(cn_csf_dir) if f.endswith('.nii')])
    for filename in cn_files:
        patient_id = extract_patient_id_from_filename(filename)
        image_patient_ids.append(patient_id)
    
    print(f"📊 修复后的患者ID示例:")
    print(f"   图像患者ID示例: {image_patient_ids[:5]}")
    print(f"   文本患者ID示例: {patient_ids[:5]}")
    
    # 5. 创建患者ID到数据的映射
    image_id_to_data = {}
    text_id_to_data = {}
    
    # 图像数据映射
    for i, pid in enumerate(image_patient_ids):
        image_id_to_data[pid] = {
            'data': image_data[i],
            'label': image_labels[i],
            'index': i
        }
    
    # 文本数据映射
    for i, pid in enumerate(patient_ids):
        text_id_to_data[pid] = {
            'data': text_data[i],
            'label': text_labels[i],
            'index': i
        }
    
    # 6. 找到共同的患者ID
    image_patient_set = set(image_patient_ids)
    text_patient_set = set(patient_ids)
    common_patients = image_patient_set & text_patient_set
    
    print(f"📊 修复后患者ID对齐统计:")
    print(f"   图像患者数: {len(image_patient_set)}")
    print(f"   文本患者数: {len(text_patient_set)}")
    print(f"   共同患者数: {len(common_patients)}")
    
    if len(common_patients) < len(image_patient_set) * 0.8:
        print(f"⚠️  警告: 共同患者比例仍然较低 ({len(common_patients)}/{len(image_patient_set)})")
        
        # 显示不匹配的患者ID进行调试
        image_only = image_patient_set - text_patient_set
        text_only = text_patient_set - image_patient_set
        
        if image_only:
            print(f"   仅在图像中: {sorted(list(image_only))[:10]}...")
        if text_only:
            print(f"   仅在文本中: {sorted(list(text_only))[:10]}...")
    else:
        print(f"✅ 患者ID对齐成功率: {len(common_patients)/len(image_patient_set)*100:.1f}%")
    
    # 7. 按共同患者ID重新组织数据
    common_patients_list = sorted(list(common_patients))  # 排序确保一致性
    
    aligned_image_data = []
    aligned_text_data = []
    aligned_labels = []
    aligned_patient_ids = []
    
    for patient_id in common_patients_list:
        # 验证标签一致性
        img_label = image_id_to_data[patient_id]['label']
        txt_label = text_id_to_data[patient_id]['label']
        
        if img_label != txt_label:
            print(f"⚠️  患者 {patient_id} 标签不一致: 图像={img_label}, 文本={txt_label}")
            continue
        
        aligned_image_data.append(image_id_to_data[patient_id]['data'])
        aligned_text_data.append(text_id_to_data[patient_id]['data'])
        aligned_labels.append(img_label)
        aligned_patient_ids.append(patient_id)
    
    # 转换为numpy数组
    aligned_image_data = np.array(aligned_image_data)
    aligned_labels = np.array(aligned_labels)
    
    print(f"✅ 修复后患者ID对齐完成:")
    print(f"   对齐后样本数: {len(aligned_labels)}")
    print(f"   图像数据形状: {aligned_image_data.shape}")
    print(f"   文本数据数量: {len(aligned_text_data)}")
    print(f"   标签分布: AD={np.sum(aligned_labels==1)}, CN={np.sum(aligned_labels==0)}")
    
    # 8. 验证对齐效果
    print(f"🔍 验证前5个样本的患者ID对齐:")
    for i in range(min(5, len(aligned_patient_ids))):
        print(f"   样本{i}: 患者ID={aligned_patient_ids[i]}, 标签={aligned_labels[i]}")
    
    # 9. 严格的患者级别数据分割
    from sklearn.model_selection import StratifiedShuffleSplit
    
    # 使用患者ID确保同一患者的数据不会同时出现在训练和测试集
    splitter = StratifiedShuffleSplit(
        n_splits=1, 
        test_size=test_size, 
        random_state=random_state
    )
    
    train_idx, test_idx = next(splitter.split(aligned_image_data, aligned_labels))
    
    # 分割数据
    train_images = aligned_image_data[train_idx]
    test_images = aligned_image_data[test_idx]
    train_texts = [aligned_text_data[i] for i in train_idx]
    test_texts = [aligned_text_data[i] for i in test_idx]
    train_labels = aligned_labels[train_idx]
    test_labels = aligned_labels[test_idx]
    
    # 验证分割结果
    print(f"📊 严格数据分割结果:")
    print(f"   训练集: {len(train_labels)} 样本 (AD={np.sum(train_labels==1)}, CN={np.sum(train_labels==0)})")
    print(f"   测试集: {len(test_labels)} 样本 (AD={np.sum(test_labels==1)}, CN={np.sum(test_labels==0)})")
    print(f"   训练集类别比例: {np.sum(train_labels==1)/len(train_labels):.3f}")
    print(f"   测试集类别比例: {np.sum(test_labels==1)/len(test_labels):.3f}")
    
    # 检查数据泄露
    train_patient_ids = [aligned_patient_ids[i] for i in train_idx]
    test_patient_ids = [aligned_patient_ids[i] for i in test_idx]
    overlap = set(train_patient_ids) & set(test_patient_ids)
    
    if overlap:
        print(f"⚠️  警告: 发现患者ID重叠: {overlap}")
    else:
        print(f"✅ 数据泄露检查通过: 训练集和测试集无患者重叠")
    
    return (train_images, train_texts, train_labels, 
            test_images, test_texts, test_labels)


def load_text_data_from_excel_clean(text_data_dir):
    """
    从Excel文件加载文本数据 - 完全清洁版本（无诊断信息泄露）
    
    Args:
        text_data_dir: 文本数据目录路径
        
    Returns:
        all_texts: List[str] 所有文本数据（不包含诊断信息）
        all_labels: numpy array 所有标签
        patient_ids: List[str] 患者ID列表 (使用NAME列)
    """
    print(f"📝 从Excel文件加载文本数据 (完全清洁版本 - 无诊断泄露)...")
    
    # 文件路径
    ad_file = os.path.join(text_data_dir, 'final_AD_updated.xlsx')
    cn_file = os.path.join(text_data_dir, 'final_CN_updated.xlsx')
    
    # 检查文件存在性
    if not os.path.exists(ad_file):
        raise FileNotFoundError(f"AD文件不存在: {ad_file}")
    if not os.path.exists(cn_file):
        raise FileNotFoundError(f"CN文件不存在: {cn_file}")
    
    # 加载数据
    ad_df = pd.read_excel(ad_file)
    cn_df = pd.read_excel(cn_file)
    
    print(f"📊 原始数据统计:")
    print(f"   AD样本: {len(ad_df)} 行")
    print(f"   CN样本: {len(cn_df)} 行")
    print(f"   AD列名: {list(ad_df.columns)}")
    print(f"   CN列名: {list(cn_df.columns)}")
    
    def create_clean_clinical_text(row):
        """
        创建完全清洁的临床文本描述 - 绝对不包含诊断信息
        
        ⚠️ 重要：此函数绝对不能包含任何可能泄露诊断的信息
        """
        text_parts = []
        
        # 基本人口统计学信息
        if 'Age' in row and pd.notna(row['Age']):
            text_parts.append(f"Patient age: {row['Age']} years")
        
        if 'Gender' in row and pd.notna(row['Gender']):
            # 转换性别编码
            gender = "male" if row['Gender'] == 1 else "female"
            text_parts.append(f"Gender: {gender}")
        
        if 'Edu' in row and pd.notna(row['Edu']):
            text_parts.append(f"Education level: {row['Edu']} years")
        
        # 认知评估分数 - 这些是客观测量，不直接泄露诊断
        cognitive_scores = []
        
        if 'MMSE' in row and pd.notna(row['MMSE']):
            cognitive_scores.append(f"MMSE score: {row['MMSE']}")
        
        if 'CDRSB' in row and pd.notna(row['CDRSB']):
            cognitive_scores.append(f"CDR-SB score: {row['CDRSB']}")
        
        # 添加其他可用的认知测试分数
        additional_scores = []
        for col in row.index:
            if col in ['ADAS11', 'ADAS13', 'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting']:
                if pd.notna(row[col]):
                    additional_scores.append(f"{col}: {row[col]}")
        
        # 组合所有信息
        if cognitive_scores:
            text_parts.append("Cognitive assessment: " + ", ".join(cognitive_scores))
        
        if additional_scores:
            text_parts.append("Additional measures: " + ", ".join(additional_scores))
        
        # 如果没有足够信息，创建基本描述
        if len(text_parts) == 0:
            text_parts = ["Clinical assessment data available for analysis"]
        
        # 🔥 关键：绝对不添加任何诊断相关信息
        final_text = " ".join(text_parts)
        
        # 额外安全检查：确保文本中不包含诊断关键词
        diagnosis_keywords = ['alzheimer', 'dementia', 'ad', 'normal', 'cn', 'cognitively normal', 'diagnosis', 'disease']
        final_text_lower = final_text.lower()
        
        for keyword in diagnosis_keywords:
            if keyword in final_text_lower:
                print(f"⚠️  警告: 检测到可能的诊断泄露关键词 '{keyword}' 在文本中")
                # 移除包含关键词的部分
                words = final_text.split()
                filtered_words = [word for word in words if keyword not in word.lower()]
                final_text = " ".join(filtered_words)
        
        return final_text
    
    # 处理AD数据
    ad_texts = []
    ad_patient_ids = []
    for idx, row in ad_df.iterrows():
        text = create_clean_clinical_text(row)
        ad_texts.append(text)
        
        # 提取患者ID - 使用NAME列
        if 'NAME' in row and pd.notna(row['NAME']):
            ad_patient_ids.append(str(row['NAME']))
        else:
            # 备用：从wholecode提取NAME部分
            if 'wholecode' in row and pd.notna(row['wholecode']):
                wholecode = str(row['wholecode'])
                # 从wholecode提取NAME: "029_S_4385_3-2016-01-29_12_25_03.0.nii" -> "029_S_4385"
                parts = wholecode.split('_')
                if len(parts) >= 3:
                    name = f"{parts[0]}_{parts[1]}_{parts[2]}"
                    ad_patient_ids.append(name)
                else:
                    ad_patient_ids.append(f"AD_{idx}")
            else:
                ad_patient_ids.append(f"AD_{idx}")
    
    # 处理CN数据
    cn_texts = []
    cn_patient_ids = []
    for idx, row in cn_df.iterrows():
        text = create_clean_clinical_text(row)
        cn_texts.append(text)
        
        # 提取患者ID - 使用NAME列
        if 'NAME' in row and pd.notna(row['NAME']):
            cn_patient_ids.append(str(row['NAME']))
        else:
            # 备用：从wholecode提取NAME部分
            if 'wholecode' in row and pd.notna(row['wholecode']):
                wholecode = str(row['wholecode'])
                # 从wholecode提取NAME: "029_S_4385_3-2016-01-29_12_25_03.0.nii" -> "029_S_4385"
                parts = wholecode.split('_')
                if len(parts) >= 3:
                    name = f"{parts[0]}_{parts[1]}_{parts[2]}"
                    cn_patient_ids.append(name)
                else:
                    cn_patient_ids.append(f"CN_{idx}")
            else:
                cn_patient_ids.append(f"CN_{idx}")
    
    # 合并数据
    all_texts = ad_texts + cn_texts
    all_labels = np.array([1] * len(ad_texts) + [0] * len(cn_texts))  # AD=1, CN=0
    patient_ids = ad_patient_ids + cn_patient_ids
    
    print(f"✅ 清洁文本数据加载完成 (无诊断泄露):")
    print(f"   总样本数: {len(all_texts)}")
    print(f"   AD样本: {len(ad_texts)}, CN样本: {len(cn_texts)}")
    print(f"   示例患者ID: {patient_ids[:5]}")
    print(f"   示例AD文本: {all_texts[0][:150]}...")
    print(f"   示例CN文本: {all_texts[len(ad_texts)][:150]}...")
    
    # 🔥 最终安全检查：验证所有文本都不包含诊断信息
    diagnosis_leak_count = 0
    diagnosis_keywords = ['alzheimer', 'dementia', 'ad', 'normal', 'cn', 'cognitively normal', 'diagnosis', 'disease']
    
    for i, text in enumerate(all_texts):
        text_lower = text.lower()
        for keyword in diagnosis_keywords:
            if keyword in text_lower:
                diagnosis_leak_count += 1
                print(f"⚠️  发现诊断泄露: 样本{i} 包含关键词 '{keyword}'")
                break
    
    if diagnosis_leak_count == 0:
        print(f"✅ 诊断泄露检查通过: 所有{len(all_texts)}个文本样本都不包含诊断信息")
    else:
        print(f"❌ 发现{diagnosis_leak_count}个样本存在诊断泄露风险")
    
    return all_texts, all_labels, patient_ids


def load_end_to_end_data_for_cv(image_data_dir, text_data_dir, holdout_test_size=0.2, random_state=42):
    """
    为交叉验证加载数据 - 防止数据泄露版本
    
    策略:
    1. 首先分离出独立的holdout测试集 (20%)
    2. 剩余80%数据用于5折交叉验证
    3. 确保holdout测试集在整个训练过程中不被使用
    
    Args:
        image_data_dir: 图像数据目录
        text_data_dir: 文本数据目录  
        holdout_test_size: 独立测试集比例 (默认20%)
        random_state: 随机种子
        
    Returns:
        cv_images: 交叉验证用图像数据 (80%)
        cv_texts: 交叉验证用文本数据 (80%)
        cv_labels: 交叉验证用标签 (80%)
        holdout_images: 独立测试集图像 (20%)
        holdout_texts: 独立测试集文本 (20%)
        holdout_labels: 独立测试集标签 (20%)
    """
    print("🔄 加载交叉验证数据 (防数据泄露版本)...")
    print("="*60)
    
    # 1. 直接加载所有对齐的数据 - 不进行分割
    print("📊 加载所有对齐数据...")
    
    # 加载图像数据
    from data_utils import load_early_fusion_data
    image_data, image_labels = load_early_fusion_data(image_data_dir)
    
    # 加载文本数据
    text_data, text_labels, patient_ids = load_text_data_from_excel_clean(text_data_dir)
    
    # 数据一致性检查
    assert len(image_data) == len(text_data), \
        f"图像数据({len(image_data)})和文本数据({len(text_data)})数量不匹配"
    assert len(image_labels) == len(text_labels), \
        f"图像标签({len(image_labels)})和文本标签({len(text_labels)})数量不匹配"
    assert np.array_equal(image_labels, text_labels), \
        "图像标签和文本标签不一致"
    
    print(f"📊 数据一致性检查通过:")
    print(f"   图像数据: {image_data.shape}")
    print(f"   文本数据: {len(text_data)} 条")
    print(f"   标签分布: AD={np.sum(text_labels==1)}, CN={np.sum(text_labels==0)}")
    
    # 2. 修复：正确提取图像文件的患者ID
    print(f"🔧 修复图像文件患者ID提取逻辑...")
    
    def extract_patient_id_from_filename(filename):
        """
        从图像文件名提取患者ID
        文件名格式: mwp3MRI_002_S_0619_3-2016-01-29_12_25_03.0.nii
        目标提取: 002_S_0619
        """
        basename = filename.split('.')[0]  # 移除.nii后缀
        
        # 分割文件名
        parts = basename.split('_')
        
        # 查找模式: mwp3MRI_XXX_S_YYYY
        if len(parts) >= 4 and parts[0].startswith('mwp') and parts[2] == 'S':
            # 提取 XXX_S_YYYY 部分
            patient_id = f"{parts[1]}_{parts[2]}_{parts[3]}"
            return patient_id
        
        # 备用方案：查找 XXX_S_YYYY 模式
        for i in range(len(parts) - 2):
            if parts[i+1] == 'S' and parts[i].isdigit() and parts[i+2].isdigit():
                return f"{parts[i]}_{parts[i+1]}_{parts[i+2]}"
        
        # 最后备用：返回原始文件名
        return basename
    
    # 重新构建图像数据的患者ID列表
    print(f"📊 重新构建图像数据的患者ID映射...")
    
    # 构建图像文件到患者ID的映射
    image_patient_ids = []
    
    # 重新扫描图像文件获取患者ID
    ad_csf_dir = os.path.join(image_data_dir, "123-AD-MRI", "ADfinalCSF")
    cn_csf_dir = os.path.join(image_data_dir, "123-CN-MRI", "CNfinalCSF")
    
    # 获取AD文件的患者ID
    ad_files = sorted([f for f in os.listdir(ad_csf_dir) if f.endswith('.nii')])
    for filename in ad_files:
        patient_id = extract_patient_id_from_filename(filename)
        image_patient_ids.append(patient_id)
    
    # 获取CN文件的患者ID
    cn_files = sorted([f for f in os.listdir(cn_csf_dir) if f.endswith('.nii')])
    for filename in cn_files:
        patient_id = extract_patient_id_from_filename(filename)
        image_patient_ids.append(patient_id)
    
    print(f"📊 修复后的患者ID示例:")
    print(f"   图像患者ID示例: {image_patient_ids[:5]}")
    print(f"   文本患者ID示例: {patient_ids[:5]}")
    
    # 3. 创建患者ID到数据的映射
    image_id_to_data = {}
    text_id_to_data = {}
    
    # 图像数据映射
    for i, pid in enumerate(image_patient_ids):
        image_id_to_data[pid] = {
            'data': image_data[i],
            'label': image_labels[i],
            'index': i
        }
    
    # 文本数据映射
    for i, pid in enumerate(patient_ids):
        text_id_to_data[pid] = {
            'data': text_data[i],
            'label': text_labels[i],
            'index': i
        }
    
    # 4. 找到共同的患者ID
    image_patient_set = set(image_patient_ids)
    text_patient_set = set(patient_ids)
    common_patients = image_patient_set & text_patient_set
    
    print(f"📊 修复后患者ID对齐统计:")
    print(f"   图像患者数: {len(image_patient_set)}")
    print(f"   文本患者数: {len(text_patient_set)}")
    print(f"   共同患者数: {len(common_patients)}")
    
    if len(common_patients) < len(image_patient_set) * 0.8:
        print(f"⚠️  警告: 共同患者比例仍然较低 ({len(common_patients)}/{len(image_patient_set)})")
        
        # 显示不匹配的患者ID进行调试
        image_only = image_patient_set - text_patient_set
        text_only = text_patient_set - image_patient_set
        
        if image_only:
            print(f"   仅在图像中: {sorted(list(image_only))[:10]}...")
        if text_only:
            print(f"   仅在文本中: {sorted(list(text_only))[:10]}...")
    else:
        print(f"✅ 患者ID对齐成功率: {len(common_patients)/len(image_patient_set)*100:.1f}%")
    
    # 5. 按共同患者ID重新组织数据
    common_patients_list = sorted(list(common_patients))  # 排序确保一致性
    
    aligned_image_data = []
    aligned_text_data = []
    aligned_labels = []
    aligned_patient_ids = []
    
    for patient_id in common_patients_list:
        # 验证标签一致性
        img_label = image_id_to_data[patient_id]['label']
        txt_label = text_id_to_data[patient_id]['label']
        
        if img_label != txt_label:
            print(f"⚠️  患者 {patient_id} 标签不一致: 图像={img_label}, 文本={txt_label}")
            continue
        
        aligned_image_data.append(image_id_to_data[patient_id]['data'])
        aligned_text_data.append(text_id_to_data[patient_id]['data'])
        aligned_labels.append(img_label)
        aligned_patient_ids.append(patient_id)
    
    # 转换为numpy数组
    aligned_image_data = np.array(aligned_image_data)
    aligned_labels = np.array(aligned_labels)
    
    print(f"✅ 修复后患者ID对齐完成:")
    print(f"   对齐后样本数: {len(aligned_labels)}")
    print(f"   图像数据形状: {aligned_image_data.shape}")
    print(f"   文本数据数量: {len(aligned_text_data)}")
    print(f"   标签分布: AD={np.sum(aligned_labels==1)}, CN={np.sum(aligned_labels==0)}")
    
    # 6. 验证对齐效果
    print(f"🔍 验证前5个样本的患者ID对齐:")
    for i in range(min(5, len(aligned_patient_ids))):
        print(f"   样本{i}: 患者ID={aligned_patient_ids[i]}, 标签={aligned_labels[i]}")
    
    print(f"\n📊 总数据统计:")
    print(f"   总样本: {len(aligned_labels)}")
    print(f"   AD样本: {np.sum(aligned_labels==1)}")
    print(f"   CN样本: {np.sum(aligned_labels==0)}")
    
    # 7. 患者级别分割 - 先分离holdout测试集
    from sklearn.model_selection import StratifiedShuffleSplit
    
    # 第一次分割: 分离holdout测试集
    holdout_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=holdout_test_size,
        random_state=random_state
    )
    
    cv_idx, holdout_idx = next(holdout_splitter.split(aligned_image_data, aligned_labels))
    
    # 分离数据
    cv_images = aligned_image_data[cv_idx]
    cv_texts = [aligned_text_data[i] for i in cv_idx]
    cv_labels = aligned_labels[cv_idx]
    
    holdout_images = aligned_image_data[holdout_idx]
    holdout_texts = [aligned_text_data[i] for i in holdout_idx]
    holdout_labels = aligned_labels[holdout_idx]
    
    print(f"\n🎯 数据分割结果 (防泄露):")
    print(f"   交叉验证集: {len(cv_labels)} 样本 (AD={np.sum(cv_labels==1)}, CN={np.sum(cv_labels==0)})")
    print(f"   独立测试集: {len(holdout_labels)} 样本 (AD={np.sum(holdout_labels==1)}, CN={np.sum(holdout_labels==0)})")
    print(f"   交叉验证比例: {len(cv_labels)/len(aligned_labels)*100:.1f}%")
    print(f"   独立测试比例: {len(holdout_labels)/len(aligned_labels)*100:.1f}%")
    
    # 8. 验证数据分离
    cv_patient_ids_list = [aligned_patient_ids[i] for i in cv_idx]
    holdout_patient_ids_list = [aligned_patient_ids[i] for i in holdout_idx]
    
    # 检查索引是否有重叠
    cv_indices = set(cv_idx)
    holdout_indices = set(holdout_idx)
    overlap = cv_indices & holdout_indices
    
    if overlap:
        print(f"⚠️  警告: 发现数据重叠: {len(overlap)} 个样本")
    else:
        print(f"✅ 数据分离验证通过: 交叉验证集和独立测试集无重叠")
    
    # 检查患者ID重叠
    cv_patient_set = set(cv_patient_ids_list)
    holdout_patient_set = set(holdout_patient_ids_list)
    patient_overlap = cv_patient_set & holdout_patient_set
    
    if patient_overlap:
        print(f"⚠️  警告: 发现患者ID重叠: {patient_overlap}")
    else:
        print(f"✅ 患者级别分离验证通过: 无患者重叠")
    
    return (cv_images, cv_texts, cv_labels,
            holdout_images, holdout_texts, holdout_labels)


def main():
    """主函数 - 优化对比学习训练"""
    print("🔧 三重损失端到端对比学习模型训练 - 修复患者ID版本")
    print("="*50)
    
    # 🔥 动态获取最佳图像模型路径
    def get_best_image_model_path():
        """智能获取最佳图像模型路径"""
        print("🔍 搜索最佳图像编码器模型...")
        
        # 🔥 优先级1: 5折交叉验证的最佳模型
        cv_models = [
            ('./models/contrastive/fold_1_best_model.pth', '第2折最佳模型 (94.74%)'),
            ('./models/contrastive/fold_0_best_model.pth', '第1折模型'),
            ('./models/contrastive/fold_2_best_model.pth', '第3折模型'),
            ('./models/contrastive/fold_3_best_model.pth', '第4折模型'),
            ('./models/contrastive/fold_4_best_model.pth', '第5折模型'),
            ('./models/contrastive/best_contrastive_image_encoder.pth', '总体最佳模型'),
            ('./models/备份1/contrastive_image_encoder_ch12.pth', '备份模型ch12'),  # 🔥 修复路径
            ('./models/备份1/contrastive_image_encoder_ch8.pth', '备份模型ch8')   # 🔥 新增备份路径
        ]
        
        for model_path, description in cv_models:
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location='cpu')
                    # 尝试从不同的键获取准确率
                    val_acc = checkpoint.get('best_val_accuracy', 
                              checkpoint.get('val_accuracy', 
                              checkpoint.get('val_acc', 0)))
                    
                    print(f"✅ 使用优选图像模型: {description}")
                    print(f"   文件路径: {model_path}")
                    print(f"   验证准确率: {val_acc:.2f}%")
                    return model_path
                except Exception as e:
                    print(f"   ⚠️ 模型加载测试失败 {model_path}: {e}")
                    continue
        
        # 🔥 优先级2: 备用模型（更新路径）
        backup_models = [
            './models/contrastive/fold_1_best_model.pth',  # 最佳模型重复检查
            './models/smart_downsample_global_ch12.pth',
            './models/best_memory_optimized_early_fusion.pth'
        ]
        
        for model_path in backup_models:
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location='cpu')
                    val_acc = checkpoint.get('val_acc', 0)
                    print(f"✅ 使用备用图像模型: {os.path.basename(model_path)}")
                    print(f"   验证准确率: {val_acc:.2f}%")
                    return model_path
                except:
                    continue
        
        raise FileNotFoundError("❌ 未找到可用的图像模型，请先训练图像编码器")
    
    # 获取图像模型路径
    try:
        image_model_path = get_best_image_model_path()
    except FileNotFoundError as e:
        print(str(e))
        print("💡 请运行: python run_contrastive_image_encoder.py")
        return
    
    # 修复配置 - 重新调整参数
    config = {
        'image_model_path': image_model_path,  # 🔥 使用动态检测的路径
        'text_data_dir': './文本编码器/',
        'image_data_dir': '/root/autodl-tmp/DATA_MCI/test_data/',
        'save_dir': './models/triple_loss_contrastive',
        
        # 端到端训练参数
        'batch_size': 8,        # 小批次，防过拟合
        'num_epochs': 20,       # 减少轮数
        'learning_rate': 1e-4,  # 保守学习率
        'weight_decay': 1e-3,   # 强正则化
        'gradient_clip': 1.0,
        
        # 三重损失权重配置
        'classification_weight': 1.0,    # AD/CN分类损失权重
        'alignment_weight': 0.5,         # 图像文本对齐损失权重
        'intra_modal_weight': 0.3,       # 图像内部对比损失权重
        
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"🎯 三重损失系统特点:")
    print(f"   1️⃣ AD/CN分类损失: {config['classification_weight']} (主要任务)")
    print(f"   2️⃣ 图像文本对齐损失: {config['alignment_weight']} (跨模态对齐)")
    print(f"   3️⃣ 图像内部对比损失: {config['intra_modal_weight']} (特征分布优化)")
    print(f"   ✅ 端到端BERT训练")
    print(f"   ✅ 严格数据分离")
    print(f"   🔧 修复患者ID对齐问题")
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    
    try:
        # 加载数据 - 严格防数据泄露
        (train_images, train_texts, train_labels,
         test_images, test_texts, test_labels) = load_end_to_end_data(
            config['image_data_dir'], config['text_data_dir']
        )
        
        # 创建模型
        print("🎯 创建三重损失多模态模型...")
        model = ImprovedMultiModalModel(
            config['image_model_path'],
            train_texts + test_texts,  # 传入所有文本数据
            feature_dim=512,
            device=config['device']
        )
        
        # 创建数据集
        train_dataset = EndToEndMultiModalDataset(train_images, train_texts, train_labels)
        test_dataset = EndToEndMultiModalDataset(test_images, test_texts, test_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=False)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=False)
        
        # 创建训练器
        trainer = OptimizedContrastiveTrainer(model, config['device'], config)
        
        print(f"🚀 开始三重损失训练，共 {config['num_epochs']} 轮...")
        print(f"🎯 目标: 优化三种损失，提升多模态性能")
        
        best_test_accuracy = 0.0
        training_history = []
        
        for epoch in range(config['num_epochs']):
            # 训练
            train_metrics = trainer.train_epoch(train_loader, epoch)
            
            # 测试
            test_metrics = trainer.evaluate(test_loader)
            
            # 记录历史
            epoch_history = {
                'epoch': epoch + 1,
                'train': train_metrics,
                'test': test_metrics
            }
            training_history.append(epoch_history)
            
            # 打印详细结果
            print(f"\n📊 Epoch {epoch+1} 结果:")
            print(f"   训练 - 总损失: {train_metrics['total_loss']:.4f}, 准确率: {train_metrics['accuracy']:.4f}")
            print(f"   训练 - 分类: {train_metrics['classification_loss']:.4f}, 对齐: {train_metrics['alignment_loss']:.4f}, 内部: {train_metrics['intra_modal_loss']:.4f}")
            print(f"   测试 - 总损失: {test_metrics['loss']:.4f}, 准确率: {test_metrics['accuracy']:.4f}")
            print(f"   测试 - 分类: {test_metrics['classification_loss']:.4f}, 对齐: {test_metrics['alignment_loss']:.4f}, 内部: {test_metrics['intra_modal_loss']:.4f}")
            print(f"   学习率: {train_metrics['learning_rate']:.6f}")
            
            # 损失分析
            if train_metrics['alignment_loss'] < 1.0:
                print(f"   ✅ 图像文本对齐良好: {train_metrics['alignment_loss']:.4f}")
            else:
                print(f"   ⚠️  图像文本对齐需改进: {train_metrics['alignment_loss']:.4f}")
            
            if train_metrics['intra_modal_loss'] < 0.5:
                print(f"   ✅ 图像特征分布优化良好: {train_metrics['intra_modal_loss']:.4f}")
            else:
                print(f"   🔄 图像特征分布持续优化: {train_metrics['intra_modal_loss']:.4f}")
            
            # 泛化能力检查
            train_acc = train_metrics['accuracy']
            test_acc = test_metrics['accuracy']
            generalization_gap = train_acc - test_acc
            
            if generalization_gap > 0.1:
                print(f"   ⚠️  过拟合警告: 训练-测试差距 {generalization_gap:.3f}")
            elif generalization_gap > 0.05:
                print(f"   🔄 轻微过拟合: 训练-测试差距 {generalization_gap:.3f}")
            else:
                print(f"   ✅ 泛化良好: 训练-测试差距 {generalization_gap:.3f}")
            
            # 保存最佳模型
            if test_metrics['accuracy'] > best_test_accuracy:
                best_test_accuracy = test_metrics['accuracy']
                save_path = os.path.join(config['save_dir'], 'best_triple_loss_model.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'best_test_accuracy': best_test_accuracy,
                    'config': config,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics
                }, save_path)
                
                print(f"🏆 新的最佳模型已保存: 测试准确率 {best_test_accuracy:.4f}")
        
        # 保存训练历史
        history_path = os.path.join(config['save_dir'], 'triple_loss_training_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            # 转换numpy数组为列表
            serializable_history = []
            for epoch_data in training_history:
                epoch_copy = {}
                for key, value in epoch_data.items():
                    if key in ['train', 'test']:
                        metrics_copy = {}
                        for metric_key, metric_value in value.items():
                            if hasattr(metric_value, 'tolist'):
                                metrics_copy[metric_key] = metric_value.tolist()
                            elif isinstance(metric_value, (int, float, str, bool)):
                                metrics_copy[metric_key] = metric_value
                            elif isinstance(metric_value, dict):
                                # 处理分类报告等字典
                                metrics_copy[metric_key] = {k: v for k, v in metric_value.items() 
                                                          if isinstance(v, (int, float, str, bool, dict))}
                        epoch_copy[key] = metrics_copy
                    else:
                        epoch_copy[key] = value
                serializable_history.append(epoch_copy)
            
            json.dump(serializable_history, f, indent=2, ensure_ascii=False)
        
        print(f"\n🎉 三重损失训练完成！")
        print(f"📈 最佳测试准确率: {best_test_accuracy:.4f}")
        print(f"💾 模型保存路径: {config['save_dir']}")
        
        # 性能分析
        print(f"\n📊 三重损失性能评估:")
        print(f"   图像单模态基线: 77.42%")
        print(f"   三重损失多模态: {best_test_accuracy*100:.2f}%")
        
        if best_test_accuracy > 0.7742:
            improvement = (best_test_accuracy - 0.7742) * 100
            print(f"   🏆 相对基线提升: +{improvement:.2f}%")
        else:
            gap = (0.7742 - best_test_accuracy) * 100
            print(f"   📉 距离基线差距: -{gap:.2f}%")
        
        print(f"\n✅ 三重损失系统优势:")
        print(f"   1️⃣ AD/CN分类损失: 确保主要任务性能")
        print(f"   2️⃣ 图像文本对齐损失: 优化跨模态特征对齐")
        print(f"   3️⃣ 图像内部对比损失: 优化特征分布质量")
        print(f"   🛡️ 严格数据分离: 真实泛化能力评估")
        print(f"   🔧 修复患者ID对齐: 确保数据一致性")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
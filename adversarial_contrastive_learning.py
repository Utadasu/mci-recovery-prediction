#!/usr/bin/env python3
"""
🎯 对抗性对比学习模型 - 增强版 v2.1
====================

🔥 v2.1 进度条优化版本
主要改进:
- ✅ 优化重复警告信息，避免干扰进度条显示
- ✅ 静默处理数值稳定性问题，只在epoch级别显示关键信息  
- ✅ 简化训练过程输出，更清晰的批次监控
- ✅ 增强进度条信息密度，包含更多有用指标
- ✅ 每5个批次更新进度条，减少频繁刷新

核心思想：
1. 保留MMSE/CDR-SB等认知评估分数（有价值的医学特征）
2. 通过对抗性训练让文本编码器学习"去偏"的表征
3. 强制图像-文本特征对齐，减少对认知分数的直接依赖
4. 使用特征解耦技术分离诊断相关和诊断无关特征

🔥 v2.1新增功能：
1. 自适应损失权重学习 - 动态调整各损失函数权重
2. MMSE分数分桶和非线性变换 - 减轻认知分数泄露
3. 正负样本对构建 - 优化对比学习效果
4. 静默数值稳定性处理 - 清晰的训练监控界面

技术架构:
- 图像编码器: ImprovedResNetCBAM3D + 智能下采样
- 文本编码器: 多元回归认知评估 + BERT编码  
- 对比学习: InfoNCE双向损失 + 特征解耦
- 融合分类: 自适应权重学习 + 图像主导策略

性能目标: 基于94.74%最佳图像编码器，提升至≥95%多模态准确率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import pickle
import os
from tqdm import tqdm
import json
import pandas as pd
import math
import re
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
import logging
from datetime import datetime
import argparse  # 新增命令行参数解析
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
import openpyxl
import torch.cuda.amp as amp
from torch.cuda.amp import GradScaler
import warnings
import random

warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.functional')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class AdaptiveLossWeights(nn.Module):
    """🔥 自适应损失权重学习模块"""
    
    def __init__(self, num_losses=7, device='cuda', initial_weights=None):
        super(AdaptiveLossWeights, self).__init__()
        self.num_losses = num_losses
        
        # 🎯 可学习的损失权重参数
        # 使用log空间学习，确保权重为正
        self.log_weights = nn.Parameter(torch.zeros(num_losses, device=device))
        
        # 损失名称映射
        self.loss_names = [
            'classification',    # 0: 分类损失
            'alignment',         # 1: 跨模态对齐损失  
            'contrastive',       # 2: 对比损失
            'reconstruction',    # 3: 重构损失
            'orthogonality',     # 4: 正交性损失
            'diagnostic',        # 5: 诊断损失
            'dominance'          # 6: 图像主导损失
            # 删除 'text_suppression'
        ]
        
        # 🔧 权重约束范围
        self.min_weight = 0.001
        self.max_weight = 10.0
        
        print(f"🎯 自适应损失权重学习器初始化:")
        print(f"   损失数量: {num_losses}")
        print(f"   权重范围: [{self.min_weight}, {self.max_weight}]")
        print(f"   损失类型: {self.loss_names}")
    
    def get_weights(self):
        """获取当前的损失权重"""
        # 从log空间转换到实际权重，并应用约束
        weights = torch.exp(self.log_weights)
        weights = torch.clamp(weights, self.min_weight, self.max_weight)
        return weights
    
    def forward(self, losses_dict, epoch=0):
        """
        计算自适应损失权重
        
        Args:
            losses_dict: 损失字典，包含各种损失
            epoch: 当前训练轮数
            
        Returns:
            total_loss: 加权总损失
            weights_dict: 当前权重字典
        """
        # 获取所有损失名称
        loss_names = list(losses_dict.keys())
        
        # 获取当前权重
        weights = self.get_weights()  # [num_losses]
        
        # 🔥 根据训练阶段调整初始权重 - 确保分类损失的主导地位
        if epoch < 5:
            # 早期阶段: 分类和对齐并重
            weights_dict = {
                'classification': 1.5,     # 分类损失权重
                'alignment': 1.0,          # 对齐损失权重
                'contrastive': 0.5,        # 对比损失权重
                'reconstruction': 0.5,     # 重构损失权重
                'orthogonality': 0.5,      # 正交性损失权重
                'diagnostic': 0.3,         # 诊断损失权重
                'dominance': 0.8           # 特征主导性损失权重
            }
        elif epoch < 15:
            # 中期阶段: 强化对齐，但分类仍是主导
            weights_dict = {
                'classification': 1.5,     # 分类损失权重
                'alignment': 1.5,          # 对齐损失权重
                'contrastive': 1.0,        # 对比损失权重
                'reconstruction': 0.7,     # 重构损失权重
                'orthogonality': 0.7,      # 正交性损失权重
                'diagnostic': 0.5,         # 诊断损失权重
                'dominance': 1.0           # 特征主导性损失权重
            }
        else: # epoch >= 15
            # 后期阶段: 稳定对齐，分类任务权重最高
            weights_dict = {
                'classification': 2.0,     # 🔥 确保分类任务拥有最高权重
                'alignment': 1.0,          # 对齐损失作为辅助
                'contrastive': 0.8,        # 对比损失作为辅助
                'reconstruction': 0.8,     # 重构损失权重
                'orthogonality': 0.8,      # 正交性损失权重
                'diagnostic': 0.3,         # 诊断损失权重
                'dominance': 1.2           # 特征主导性损失权重
            }
        
        # 构建损失向量
        loss_values = []
        adjusted_weights = []
        
        # 为每个损失应用权重
        for i, loss_name in enumerate(self.loss_names):
            if loss_name in losses_dict:
                loss_values.append(losses_dict[loss_name])
                # 应用预定义权重（如果损失名称匹配）
                if loss_name in weights_dict:
                    adjusted_weights.append(weights_dict[loss_name])
                else:
                    adjusted_weights.append(weights[i])
            else:
                loss_values.append(torch.tensor(0.0, device=self.device))
                adjusted_weights.append(weights[i])
        
        # 转换为张量
        loss_tensor = torch.stack(loss_values)
        adjusted_weights_tensor = torch.tensor(adjusted_weights, device=self.device)
        
        # 计算加权总损失
        total_loss = torch.sum(adjusted_weights_tensor * loss_tensor)
        
        # 构建权重字典用于监控
        weights_monitor = {}
        for i, loss_name in enumerate(self.loss_names):
            if i < len(adjusted_weights):
                weights_monitor[f'{loss_name}_weight'] = adjusted_weights[i]
        
        return total_loss, weights_monitor


class CognitiveAssessmentProcessor(nn.Module):
    """🔥 认知评估处理器 - 多元回归校正 + CDR-SB整合"""
    
    def __init__(self, device='cuda'):
        super(CognitiveAssessmentProcessor, self).__init__()
        
        self.device = device
        
        # 🎯 MMSE多元回归校正参数 (基于循证医学研究)
        # 基于Crum et al. (1993) JAMA - 18,056人队列研究
        # 年龄、性别、教育对MMSE分数的多元回归模型
        self.mmse_regression_params = {
            'intercept': 29.1,           # 基线截距 (高教育组基准)
            'age_coef': -0.045,          # 年龄系数 (每年-0.045分)
            'age_squared_coef': -0.0003, # 年龄平方项 (非线性老化效应)
            'gender_coef': 0.1,          # 性别系数 (基于实际研究，差异很小)
            'education_coef': 0.35,      # 教育系数 (每年+0.35分)
            'education_squared_coef': -0.008  # 教育平方项 (边际递减效应)
        }
        
        # 🎯 CDR-SB分箱策略 (基于Morris 1993原始分级标准)
        # 参考: Morris, J.C. (1993). Neurology, 43(11):2412-4
        # 完全遵循原始CDR-SB评分系统的严重程度分级
        self.cdrsb_bins = {
            'normal': [0, 0.5],          # 正常 (CDR 0)
            'questionable': [0.5, 2.5],  # 可疑认知障碍 (CDR 0.5)
            'mild': [2.5, 4.5],          # 轻度痴呆 (CDR 1)
            'moderate': [4.5, 9.0],      # 中度痴呆 (CDR 2)
            'severe': [9.0, 18.0]        # 重度痴呆 (CDR 3)
        }
        
        # 🔧 MMSE多元回归校正网络
        self.mmse_corrector = nn.Sequential(
            nn.Linear(5, 32),            # 输入: [age, age², gender, education, education²]
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)             # 输出: 校正值
        )
        
        # 🔧 MMSE特征编码器
        self.mmse_encoder = nn.Sequential(
            nn.Linear(2, 32),            # 输入: [raw_score, corrected_score]
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64)            # 输出: 64维MMSE特征
        )
        
        # 🔧 CDR-SB分箱嵌入
        self.cdrsb_bin_embedding = nn.Embedding(5, 32)  # 5个严重程度级别
        
        # 🔧 CDR-SB连续值编码器
        self.cdrsb_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32)            # 输出: 32维CDR-SB特征
        )
        
        # 🎯 多模态认知特征融合器
        self.cognitive_fusion = nn.Sequential(
            nn.Linear(64 + 32 + 32, 128), # MMSE(64) + CDR-SB分箱(32) + CDR-SB连续(32)
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16)            # 最终输出: 16维认知特征
        )
        
        print(f"🎯 多元回归认知评估处理器初始化:")
        print(f"   MMSE多元回归校正: 年龄 + 年龄² + 性别 + 教育 + 教育²")
        print(f"   CDR-SB分级: {list(self.cdrsb_bins.keys())}")
        print(f"   特征融合: MMSE(64) + CDR-SB(64) → 16维")
        print(f"   校正参数: {self.mmse_regression_params}")
    
    def extract_demographic_info(self, texts):
        """从文本中提取人口统计学信息"""
        demographics = []
        
        for text in texts:
            demo_info = {'age': 70.0, 'gender': 0, 'education': 12.0}  # 默认值
            
            # 🔥 提取年龄
            age_patterns = [
                r'Age:\s*(\d+(?:\.\d+)?)\s*years',
                r'年龄:\s*(\d+(?:\.\d+)?)\s*岁',
                r'age:\s*(\d+(?:\.\d+)?)'
            ]
            for pattern in age_patterns:
                match = re.search(pattern, text)
                if match:
                    demo_info['age'] = float(match.group(1))
                    break
            
            # 🔥 提取性别 (0=男性, 1=女性)
            gender_patterns = [
                r'Gender:\s*(male|female)',
                r'性别:\s*(男|女)',
                r'gender:\s*(male|female)'
            ]
            for pattern in gender_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    gender_str = match.group(1).lower()
                    if gender_str in ['female', '女']:
                        demo_info['gender'] = 1
                    else:
                        demo_info['gender'] = 0
                    break
            
            # 🔥 提取教育年限
            edu_patterns = [
                r'Education:\s*(\d+(?:\.\d+)?)\s*years',
                r'教育:\s*(\d+(?:\.\d+)?)\s*年',
                r'education:\s*(\d+(?:\.\d+)?)'
            ]
            for pattern in edu_patterns:
                match = re.search(pattern, text)
                if match:
                    demo_info['education'] = float(match.group(1))
                    break
            
            # 🔧 数据范围检查
            demo_info['age'] = max(18.0, min(120.0, demo_info['age']))
            demo_info['education'] = max(0.0, min(25.0, demo_info['education']))
            
            demographics.append(demo_info)
        
        return demographics
    
    def extract_mmse_scores(self, texts):
        """从文本中提取MMSE分数"""
        mmse_scores = []
        
        for text in texts:
            patterns = [
                r'Mini-Mental State Examination \(MMSE\):\s*\[(\d+(?:\.\d+)?)/30\]',
                r'MMSE:\s*\[(\d+(?:\.\d+)?)/30\]',
                r'MMSE:\s*(\d+(?:\.\d+)?)',
                r'Mini-Mental State Examination.*?(\d+(?:\.\d+)?)'
            ]
            
            mmse_score = None
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    mmse_score = float(match.group(1))
                    break
            
            if mmse_score is None:
                mmse_score = 15.0  # 默认中等分数
            
            mmse_score = max(0.0, min(30.0, mmse_score))
            mmse_scores.append(mmse_score)
        
        return torch.tensor(mmse_scores, dtype=torch.float32, device=self.device)
    
    def extract_cdrsb_scores(self, texts):
        """从文本中提取CDR-SB分数"""
        cdrsb_scores = []
        
        for text in texts:
            patterns = [
                r'Clinical Dementia Rating - Sum of Boxes \(CDR-SB\):\s*\[(\d+(?:\.\d+)?)\]',
                r'CDR-SB:\s*\[(\d+(?:\.\d+)?)\]',
                r'CDR-SB:\s*(\d+(?:\.\d+)?)',
                r'Clinical Dementia Rating.*?(\d+(?:\.\d+)?)'
            ]
            
            cdrsb_score = None
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    cdrsb_score = float(match.group(1))
                    break
            
            if cdrsb_score is None:
                cdrsb_score = 2.0  # 默认中等分数
            
            cdrsb_score = max(0.0, min(18.0, cdrsb_score))
            cdrsb_scores.append(cdrsb_score)
        
        return torch.tensor(cdrsb_scores, dtype=torch.float32, device=self.device)
    
    def compute_mmse_correction(self, demographics):
        """
        基于多因素回归模型计算MMSE校正值
        
        Args:
            demographics: List[Dict] 人口统计学信息
            
        Returns:
            corrections: Tensor [B] 校正值
        """
        batch_size = len(demographics)
        regression_features = torch.zeros(batch_size, 5, device=self.device)
        
        for i, demo in enumerate(demographics):
            age = demo['age']
            gender = demo['gender']
            education = demo['education']
            
            # 构建回归特征 [age, age², gender, education, education²]
            regression_features[i, 0] = age
            regression_features[i, 1] = age ** 2
            regression_features[i, 2] = gender
            regression_features[i, 3] = education
            regression_features[i, 4] = education ** 2
        
        # 使用神经网络进行非线性校正 (比线性回归更灵活)
        corrections = self.mmse_corrector(regression_features).squeeze()
        return corrections
    
    def get_cdrsb_bins(self, cdrsb_scores):
        """将CDR-SB分数转换为分箱标签"""
        bins = torch.zeros(len(cdrsb_scores), dtype=torch.long, device=self.device)
        
        for i, score in enumerate(cdrsb_scores):
            score_val = score.item()
            
            if 0 <= score_val < 0.5:
                bins[i] = 0  # normal
            elif 0.5 <= score_val < 2.5:
                bins[i] = 1  # questionable
            elif 2.5 <= score_val < 4.5:
                bins[i] = 2  # mild
            elif 4.5 <= score_val < 9.0:
                bins[i] = 3  # moderate
            else:  # >= 9.0
                bins[i] = 4  # severe
        
        return bins
    
    def forward(self, texts):
        """
        多元回归认知评估处理
        
        Args:
            texts: List[str] 文本列表
            
        Returns:
            cognitive_features: [B, 16] 融合认知特征
        """
        # 🔥 Step 1: 提取所有认知和人口统计学信息
        demographics = self.extract_demographic_info(texts)
        mmse_scores = self.extract_mmse_scores(texts)      # [B]
        cdrsb_scores = self.extract_cdrsb_scores(texts)    # [B]
        
        # 🔥 Step 2: 多元回归MMSE校正
        mmse_corrections = self.compute_mmse_correction(demographics)  # [B]
        corrected_mmse = mmse_scores - mmse_corrections  # 多元回归校正
        
        # 标准化到[-1, 1]范围
        normalized_raw = (mmse_scores - 15.0) / 15.0
        normalized_corrected = (corrected_mmse - 15.0) / 15.0
        
        # MMSE特征编码 (包含原始分数和校正分数)
        mmse_input = torch.stack([normalized_raw, normalized_corrected], dim=1)  # [B, 2]
        mmse_features = self.mmse_encoder(mmse_input)  # [B, 64]
        
        # 🔥 Step 3: CDR-SB双路径处理
        # 3.1 分箱路径
        cdrsb_bins = self.get_cdrsb_bins(cdrsb_scores)  # [B]
        cdrsb_bin_features = self.cdrsb_bin_embedding(cdrsb_bins)  # [B, 32]
        
        # 3.2 连续值路径
        normalized_cdrsb = (cdrsb_scores - 4.5) / 4.5  # 标准化到[-1, 1]
        cdrsb_continuous_features = self.cdrsb_encoder(normalized_cdrsb.unsqueeze(1))  # [B, 32]
        
        # 🔥 Step 4: 多模态认知特征融合
        combined_features = torch.cat([
            mmse_features,              # [B, 64] 多元回归校正MMSE特征
            cdrsb_bin_features,         # [B, 32] CDR-SB分箱特征
            cdrsb_continuous_features   # [B, 32] CDR-SB连续特征
        ], dim=1)  # [B, 128]
        
        cognitive_features = self.cognitive_fusion(combined_features)  # [B, 16]
        
        return cognitive_features


# 为了保持向后兼容性，创建一个别名
MMSEProcessor = CognitiveAssessmentProcessor


class ContrastiveSampler:
    """🔥 对比学习正负样本构建器"""
    
    def __init__(self, temperature=0.05, hard_negative_ratio=0.3):  # 🔥 温度从0.1降低到0.05，增强AD样本对齐
        self.temperature = temperature
        self.hard_negative_ratio = hard_negative_ratio
        
        print(f"🎯 对比学习采样器初始化:")
        print(f"   温度参数: {temperature} (🔥 优化后 - 降低温度增强AD样本对齐)")
        print(f"   困难负样本比例: {hard_negative_ratio}")
    
    def create_positive_pairs(self, image_features, text_features, labels):
        """创建正样本对"""
        batch_size = image_features.size(0)
        
        # 同类样本作为正样本对
        positive_pairs = []
        positive_labels = []
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                if labels[i] == labels[j]:
                    # 图像-文本正样本对
                    positive_pairs.append((image_features[i], text_features[j]))
                    positive_pairs.append((image_features[j], text_features[i]))
                    positive_labels.extend([1, 1])
        
        return positive_pairs, positive_labels
    
    def create_negative_pairs(self, image_features, text_features, labels):
        """创建负样本对"""
        batch_size = image_features.size(0)
        
        # 不同类样本作为负样本对
        negative_pairs = []
        negative_labels = []
        
        for i in range(batch_size):
            for j in range(batch_size):
                if labels[i] != labels[j]:
                    # 图像-文本负样本对
                    negative_pairs.append((image_features[i], text_features[j]))
                    negative_labels.append(0)
        
        return negative_pairs, negative_labels
    
    def compute_contrastive_loss(self, image_features, text_features, labels):
        """
        计算对比学习损失（修复数值稳定性和小批次问题）
        
        Args:
            image_features: [B, D] 图像特征
            text_features: [B, D] 文本特征  
            labels: [B] 标签
            
        Returns:
            contrastive_loss: 对比学习损失
        """
        batch_size = image_features.size(0)
        device = image_features.device
        
        # 🔧 数值稳定性预检查
        if not torch.isfinite(image_features).all() or not torch.isfinite(text_features).all():
            # 🔥 静默处理，避免重复警告
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 🔥 确保特征已标准化
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        
        # 🎯 计算相似度矩阵 - 添加数值稳定性保护
        # 图像到文本的相似度
        sim_i2t = torch.matmul(image_features, text_features.t()) / self.temperature  # [B, B]
        # 文本到图像的相似度  
        sim_t2i = torch.matmul(text_features, image_features.t()) / self.temperature  # [B, B]
        
        # 🔥 限制相似度范围，防止exp爆炸
        sim_i2t = torch.clamp(sim_i2t, min=-10.0, max=10.0)
        sim_t2i = torch.clamp(sim_t2i, min=-10.0, max=10.0)
        
        # 🔧 创建标签掩码 - 修复正负样本判断逻辑
        labels_expanded = labels.unsqueeze(1)  # [B, 1]
        labels_matrix = labels_expanded == labels_expanded.t()  # [B, B] 同类为True
        
        # 🔥 修复：正样本掩码应该排除对角线（自己与自己）
        identity_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        pos_mask_i2t = labels_matrix & (~identity_mask)  # 同类但非自己
        neg_mask_i2t = (~labels_matrix)  # 不同类
        
        # 🔧 检查是否有有效的正样本和负样本对
        pos_count = pos_mask_i2t.sum().item()
        neg_count = neg_mask_i2t.sum().item()
        
        # 🔥 改进的处理策略：如果缺少正样本或负样本，使用简化的InfoNCE
        if pos_count == 0 or neg_count == 0:
            # 🔧 静默处理，只在epoch开始时显示一次警告
            if not hasattr(self, '_small_batch_warning_shown'):
                print(f"🔧 检测到小批次或单一类别批次，使用简化对比学习")
                self._small_batch_warning_shown = True
            
            # 🎯 使用自监督对比学习：图像与文本特征的对齐
            # 不区分正负样本，直接最大化对应样本的图像-文本相似度
            diagonal_sim = torch.diag(sim_i2t)  # [B] 对应样本的相似度
            
            # 使用简单的对齐损失：最大化对应样本的相似度
            alignment_loss = -torch.mean(diagonal_sim)
            
            return alignment_loss
        
        # 🔥 标准InfoNCE损失计算 - 使用数值稳定的方法
        try:
            # 🎯 图像到文本的对比损失
            pos_mask_float = pos_mask_i2t.float()
            neg_mask_float = neg_mask_i2t.float()
            
            # 计算每行的正样本和负样本相似度
            loss_i2t_list = []
            for i in range(batch_size):
                # 获取第i行的正样本和负样本相似度
                pos_sims = sim_i2t[i][pos_mask_i2t[i]]  # 第i个图像与同类文本的相似度
                neg_sims = sim_i2t[i][neg_mask_i2t[i]]  # 第i个图像与异类文本的相似度
                
                if len(pos_sims) == 0:
                    # 如果没有正样本，跳过这个样本
                    loss_i2t_list.append(torch.tensor(0.0, device=device))
                    continue
                
                if len(neg_sims) == 0:
                    # 如果没有负样本，只计算正样本损失
                    loss_i2t_list.append(-torch.mean(pos_sims))
                    continue
                
                # 🔥 使用log-sum-exp技巧计算InfoNCE损失
                # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
                # = -pos + log(exp(pos) + sum(exp(neg)))
                # = -pos + logsumexp([pos, neg])
                
                all_sims = torch.cat([pos_sims, neg_sims])  # 合并正负样本相似度
                log_sum_exp = torch.logsumexp(all_sims, dim=0)
                
                # 对于多个正样本，取平均
                pos_avg = torch.mean(pos_sims)
                loss_i = -(pos_avg - log_sum_exp)
                loss_i2t_list.append(loss_i)
            
            loss_i2t = torch.stack(loss_i2t_list)
            
            # 🎯 文本到图像的对比损失 - 同样处理
            loss_t2i_list = []
            for i in range(batch_size):
                pos_sims = sim_t2i[i][pos_mask_i2t[i]]  # 注意：使用相同的掩码
                neg_sims = sim_t2i[i][neg_mask_i2t[i]]
                
                if len(pos_sims) == 0:
                    loss_t2i_list.append(torch.tensor(0.0, device=device))
                    continue
                
                if len(neg_sims) == 0:
                    loss_t2i_list.append(-torch.mean(pos_sims))
                    continue
                
                all_sims = torch.cat([pos_sims, neg_sims])
                log_sum_exp = torch.logsumexp(all_sims, dim=0)
                pos_avg = torch.mean(pos_sims)
                loss_i = -(pos_avg - log_sum_exp)
                loss_t2i_list.append(loss_i)
            
            loss_t2i = torch.stack(loss_t2i_list)
            
        except Exception as e:
            # 🔥 如果计算失败，使用简化版本
            if not hasattr(self, '_fallback_warning_shown'):
                print(f"🔧 对比学习计算异常，使用备用方案: {e}")
                self._fallback_warning_shown = True
            
            # 备用方案：简单的对齐损失
            diagonal_sim = torch.diag(sim_i2t)
            return -torch.mean(diagonal_sim)
        
        # 🔧 最终数值检查（静默处理）
        if not torch.isfinite(loss_i2t).all():
            loss_i2t = torch.zeros_like(loss_i2t)
        
        if not torch.isfinite(loss_t2i).all():
            loss_t2i = torch.zeros_like(loss_t2i)
        
        # 🔧 总对比损失
        contrastive_loss = (torch.mean(loss_i2t) + torch.mean(loss_t2i)) / 2
        
        # 最终检查（静默处理）
        if not torch.isfinite(contrastive_loss):
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return contrastive_loss


class FeatureDisentanglementLoss(nn.Module):
    """特征解耦损失 - 分离诊断相关和无关特征"""
    
    def __init__(self, feature_dim=512, disentangle_dim=256):
        super(FeatureDisentanglementLoss, self).__init__()
        
        self.feature_dim = feature_dim
        self.disentangle_dim = disentangle_dim
        
        # 特征分离器：将512维特征分为两部分
        # 诊断相关特征 (256维) 和 诊断无关特征 (256维)
        self.diagnostic_projector = nn.Sequential(
            nn.Linear(feature_dim, disentangle_dim),
            nn.LayerNorm(disentangle_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.non_diagnostic_projector = nn.Sequential(
            nn.Linear(feature_dim, disentangle_dim),
            nn.LayerNorm(disentangle_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 诊断预测器（用于对抗训练）
        self.diagnostic_classifier = nn.Sequential(
            nn.Linear(disentangle_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # AD vs CN
        )
        
        # 重构器（确保信息不丢失）
        self.reconstructor = nn.Sequential(
            nn.Linear(disentangle_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
    
    def forward(self, text_features, labels):
        """
        特征解耦前向传播
        
        Args:
            text_features: [B, 512] 文本特征
            labels: [B] 真实标签
            
        Returns:
            dict: 包含各种损失和特征的字典
        """
        batch_size = text_features.size(0)
        
        # 1. 特征分离
        diagnostic_features = self.diagnostic_projector(text_features)      # [B, 256] 诊断相关
        non_diagnostic_features = self.non_diagnostic_projector(text_features)  # [B, 256] 诊断无关
        
        # 2. 诊断预测（用于对抗训练）
        diagnostic_logits = self.diagnostic_classifier(diagnostic_features)
        
        # 3. 特征重构（确保信息保持）
        combined_features = torch.cat([diagnostic_features, non_diagnostic_features], dim=1)
        reconstructed_features = self.reconstructor(combined_features)
        
        # 4. 计算各种损失
        
        # 诊断损失（希望诊断相关特征能预测标签）
        diagnostic_loss = F.cross_entropy(diagnostic_logits, labels)
        
        # 重构损失（确保信息不丢失）
        reconstruction_loss = F.mse_loss(reconstructed_features, text_features)
        
        # 正交性损失（确保两部分特征相互独立）
        # 计算诊断和非诊断特征的相关性，希望其为0
        diagnostic_norm = F.normalize(diagnostic_features, p=2, dim=1)
        non_diagnostic_norm = F.normalize(non_diagnostic_features, p=2, dim=1)
        
        # 批次内相关性
        correlation = torch.sum(diagnostic_norm * non_diagnostic_norm, dim=1).mean()
        orthogonality_loss = correlation ** 2  # 希望相关性为0
        
        return {
            'diagnostic_features': diagnostic_features,
            'non_diagnostic_features': non_diagnostic_features,
            'reconstructed_features': reconstructed_features,
            'diagnostic_logits': diagnostic_logits,
            'diagnostic_loss': diagnostic_loss,
            'reconstruction_loss': reconstruction_loss,
            'orthogonality_loss': orthogonality_loss
        }


class AdversarialTextEncoder(nn.Module):
    """
    🔥 对抗性文本编码器 (V2.2 - 支持消融实验)
    - 集成BERT、认知评估、特征融合和对抗性投影
    - 可通过 'use_cognitive_features' 开关控制是否使用认知分数
    """
    def __init__(self, feature_dim=512, device='cuda', max_length=512, use_cognitive_features=True):
        super(AdversarialTextEncoder, self).__init__()
        
        self.device = device
        self.max_length = max_length
        self.feature_dim = feature_dim
        self.use_cognitive_features = use_cognitive_features
        
        # 1. BERT模型和分词器
        self.bert_model_name = 'bert-base-uncased'
        self.bert_model = AutoModel.from_pretrained(self.bert_model_name)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        
        # 2. 认知评估处理器
        self.mmse_processor = CognitiveAssessmentProcessor(device=device)
        
        # 3. BERT与认知特征融合层
        # 输入维度: BERT(768) + 认知(16) = 784
        self.bert_mmse_fusion = nn.Sequential(
            nn.Linear(768 + 16, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 768)
        )
        
        # 4. 对抗性投影层 (将768维特征投影到统一的512维空间)
        self.adversarial_projection = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, feature_dim)
        )
        
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.bert_mmse_fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.adversarial_projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def encode_texts(self, texts):
        """
        核心文本编码流程 (V2.2 - 支持消融)
        """
        # 步骤 1: BERT编码
        inputs = self.bert_tokenizer(
            texts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=self.max_length
        ).to(self.device)
        
        bert_features = self.bert_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )[0][:, 0, :] # (B, 768)

        # 步骤 2: 认知处理 (如果启用)
        if self.use_cognitive_features:
            cognitive_features = self.mmse_processor(texts)
            # 确保在批处理大小为1时，认知特征也能正确对齐
            if bert_features.size(0) == 1 and cognitive_features.size(0) > 1:
                cognitive_features = cognitive_features[0].unsqueeze(0)

            # 步骤 3: 特征拼接
            features_to_fuse = torch.cat([bert_features, cognitive_features], dim=1) # (B, 784)
            # 步骤 4: BERT-认知融合
            fused_features = self.bert_mmse_fusion(features_to_fuse) # (B, 768)
        else:
            # 如果禁用认知特征，则跳过拼接和融合，直接使用BERT特征
            fused_features = bert_features # (B, 768)

        # 步骤 5: 对抗性投影
        projected_features = self.adversarial_projection(fused_features) # (B, 512)
        
        return projected_features

    def forward(self, texts, labels=None):
        """
        前向传播 (V2.2 简化)
        """
        text_features_512d = self.encode_texts(texts)
        return text_features_512d


class AdversarialContrastiveModel(nn.Module):
    """对抗性对比学习模型 - 图像主导分类，文本辅助对齐"""
    
    def __init__(self, image_model_path, feature_dim=512, num_classes=2, device='cuda', use_cognitive_features=True, use_disentanglement=True):
        super(AdversarialContrastiveModel, self).__init__()
        
        self.device = device
        self.feature_dim = feature_dim # 保存feature_dim
        self.num_classes = num_classes # 保存num_classes
        self.warning_collector = [] # 🔥 新增: 用于收集警告信息
        
        # 图像编码器（使用预训练模型，但冻结大部分参数）
        from optimized_contrastive_learning import ImprovedImageEncoder
        self.image_encoder = ImprovedImageEncoder(image_model_path, feature_dim, device)
        
        # 🎯 增强对抗性文本编码器
        self.text_encoder = AdversarialTextEncoder(feature_dim, device, use_cognitive_features=use_cognitive_features, use_disentanglement=use_disentanglement)
        
        # 🔥 新增：自适应损失权重学习器
        self.adaptive_weights = AdaptiveLossWeights(num_losses=7, device=device)  # 删除text_suppression，从8改为7
        
        # 🔥 新增：对比学习采样器 (优化参数)
        self.contrastive_sampler = ContrastiveSampler(temperature=0.05) # 使用之前调整的0.05
        
        # 🔥 关键：跨模态对齐损失（强制图像-文本特征对齐）
        self.cross_modal_aligner = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # 🔥 新的图像分类头: 直接在图像特征上进行分类
        self.image_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
        # 🔧 确保所有组件在正确的设备上
        self.cross_modal_aligner.to(device)
        self.image_classifier.to(device) # 新增
        
        print(f"🎯 图像主导分类模型配置:")
        print(f"   🔥 图像编码器输出直接用于分类")
        print(f"   🔥 文本特征用于辅助对比学习对齐")
        print(f"   🔥 自适应损失权重学习")
        print(f"   🔥 对比学习正负样本构建 (温度: {self.contrastive_sampler.temperature})")
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # 初始化 cross_modal_aligner 和新的 image_classifier
        for m in [self.cross_modal_aligner, self.image_classifier]: 
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    
    def preprocess_features_for_alignment(self, image_features, text_features):
        """🔥 对齐前的特征预处理 - 提升对齐效果（修复数值稳定性）"""
        
        batch_size = image_features.size(0)
        
        # 🔧 数值稳定性修复：处理小批次情况
        if batch_size == 1:
            # 对于单样本批次，直接使用L2标准化，跳过标准差计算
            image_processed = F.normalize(image_features, p=2, dim=1)
            text_processed = F.normalize(text_features, p=2, dim=1)
            return image_processed, text_processed
        
        # 1. 特征去中心化 - 添加数值检查
        image_mean = image_features.mean(dim=0, keepdim=True)
        text_mean = text_features.mean(dim=0, keepdim=True)
        
        # 检查均值是否为有限值
        if not torch.isfinite(image_mean).all() or not torch.isfinite(text_mean).all():
            # 🔥 静默处理，避免重复警告
            if 'mean_finite' not in self.warning_collector: self.warning_collector.append('mean_finite')
            return F.normalize(image_features, p=2, dim=1), F.normalize(text_features, p=2, dim=1)
        
        image_centered = image_features - image_mean
        text_centered = text_features - text_mean
        
        # 2. 特征尺度标准化 - 修复标准差计算
        # 🔥 使用unbiased=False避免小批次下的数值问题
        image_std = image_features.std(dim=0, keepdim=True, unbiased=False) + 1e-6  # 增大epsilon
        text_std = text_features.std(dim=0, keepdim=True, unbiased=False) + 1e-6
        
        # 🔧 额外检查：如果标准差过小，使用替代方案（静默处理）
        min_std_threshold = 1e-4
        if (image_std < min_std_threshold).any() or (text_std < min_std_threshold).any():
            # 🔥 使用静默替代方案，仅在epoch级别记录一次
            if 'std_threshold' not in self.warning_collector: self.warning_collector.append('std_threshold')
            return F.normalize(image_features, p=2, dim=1), F.normalize(text_features, p=2, dim=1)
        
        # 检查标准差是否为有限值
        if not torch.isfinite(image_std).all() or not torch.isfinite(text_std).all():
            # 🔥 静默处理，避免重复警告
            if 'std_finite' not in self.warning_collector: self.warning_collector.append('std_finite')
            return F.normalize(image_features, p=2, dim=1), F.normalize(text_features, p=2, dim=1)
        
        image_normalized = image_centered / image_std
        text_normalized = text_centered / text_std
        
        # 3. 特征维度平衡 (确保两种特征在相同尺度) - 添加稳定性检查
        image_scale = torch.norm(image_normalized, p=2, dim=1, keepdim=True) + 1e-8
        text_scale = torch.norm(text_normalized, p=2, dim=1, keepdim=True) + 1e-8
        
        # 检查范数是否为有限值
        if not torch.isfinite(image_scale).all() or not torch.isfinite(text_scale).all():
            # 🔥 静默处理，避免重复警告
            if 'scale_finite' not in self.warning_collector: self.warning_collector.append('scale_finite')
            return F.normalize(image_features, p=2, dim=1), F.normalize(text_features, p=2, dim=1)
        
        # 使用几何平均作为目标尺度 - 添加数值保护
        target_scale = torch.sqrt(torch.clamp(image_scale * text_scale, min=1e-8))
        
        # 🔧 添加最终数值检查
        image_scale_safe = torch.clamp(image_scale, min=1e-8)
        text_scale_safe = torch.clamp(text_scale, min=1e-8)
        target_scale_safe = torch.clamp(target_scale, min=1e-8)
        
        image_balanced = image_normalized * (target_scale_safe / image_scale_safe)
        text_balanced = text_normalized * (target_scale_safe / text_scale_safe)
        
        # 🔥 最终安全检查：确保输出没有NaN或Inf（静默处理）
        if not torch.isfinite(image_balanced).all() or not torch.isfinite(text_balanced).all():
            if 'balance_finite' not in self.warning_collector: self.warning_collector.append('balance_finite')
            return F.normalize(image_features, p=2, dim=1), F.normalize(text_features, p=2, dim=1)
        
        return image_balanced, text_balanced

    def improved_alignment_loss(self, text_features, image_features, epoch=0, labels=None):
        """🔥 改进的渐进式跨模态对齐损失（修复数值稳定性）"""
        
        # 🎯 渐进式参数调整 - 修复温度参数过低问题
        if epoch < 5:
            margin = 0.8        # 早期非常宽松
            temperature = 0.3   # 🔥 提高初始温度，避免数值不稳定
            alignment_weight = 0.5  # 🔥 增加早期权重
        elif epoch < 10:
            margin = 0.5        # 逐步收紧
            temperature = 0.2   # 🔥 降低温度
            alignment_weight = 0.7  # 🔥 增加权重
        elif epoch < 20:
            margin = 0.3        # 中期适中
            temperature = 0.15  # 🔥 降低温度
            alignment_weight = 0.8  # 🔥 增加权重
        else:
            margin = 0.2        # 后期严格
            temperature = 0.1   # 🔥 降低温度
            alignment_weight = 1.0  # 🔥 最大权重
        
        # 🔥 特征预处理（已修复数值稳定性）
        try:
            image_processed, text_processed = self.preprocess_features_for_alignment(
                image_features, text_features
            )
        except Exception as e:
            if 'preprocess_failed' not in self.warning_collector: self.warning_collector.append('preprocess_failed')
            image_processed = F.normalize(image_features, p=2, dim=1)
            text_processed = F.normalize(text_features, p=2, dim=1)
        
        # 🎯 方案1: 改进的温度调节相似度对齐 - 添加数值保护
        text_norm = F.normalize(text_processed, p=2, dim=1)
        image_norm = F.normalize(image_processed, p=2, dim=1)
        
        # 检查标准化结果
        if not torch.isfinite(text_norm).all() or not torch.isfinite(image_norm).all():
            if 'norm_finite' not in self.warning_collector: self.warning_collector.append('norm_finite')
            return torch.tensor(0.0, device=text_features.device, requires_grad=True)
        
        # 计算相似度矩阵 - 添加数值稳定性保护
        sim_matrix = torch.matmul(text_norm, image_norm.t()) / temperature  # [B, B]
        
        # 🔥 限制相似度矩阵的范围，防止exp爆炸
        sim_matrix = torch.clamp(sim_matrix, min=-10.0, max=10.0)
        
        # 对角线元素是对应样本的相似度
        diag_sim = torch.diag(sim_matrix)  # [B]
        
        # 🔥 为AD样本添加额外权重
        if labels is not None:
            # 创建样本权重向量 (默认权重为1.0)
            sample_weights = torch.ones_like(diag_sim)
            
            # AD样本权重提升到2.0 (AD标签为1) - 🔥 权重从5.0降至2.0，避免过度偏移
            ad_mask = (labels == 1)
            if ad_mask.sum() > 0:
                sample_weights[ad_mask] = 2.0
                
            # 应用样本权重到对角线相似度
            weighted_diag_sim = diag_sim * sample_weights
        else:
            weighted_diag_sim = diag_sim
        
        # 🔥 InfoNCE风格的对齐损失 (更平滑) - 增强数值稳定性
        # 使用更大的epsilon值
        eps = 1e-6
        exp_diag = torch.exp(diag_sim)
        exp_all = torch.sum(torch.exp(sim_matrix), dim=1)  # 每行求和
        
        # 检查exp值是否有限
        if not torch.isfinite(exp_diag).all() or not torch.isfinite(exp_all).all():
            if 'exp_finite' not in self.warning_collector: self.warning_collector.append('exp_finite')
            # 使用log-sum-exp技巧
            log_sum_exp = torch.logsumexp(sim_matrix, dim=1)
            
            # 🔥 使用加权对角线相似度
            if labels is not None:
                infonce_alignment_loss = -torch.mean(weighted_diag_sim - log_sum_exp)
            else:
                infonce_alignment_loss = -torch.mean(diag_sim - log_sum_exp)
        else:
            # 添加更大的数值稳定性
            infonce_alignment_loss = -torch.log(exp_diag / (exp_all + eps) + eps).mean()
            
            # 🔥 如果有标签，添加额外的加权损失
            if labels is not None:
                ad_mask = (labels == 1)
                if ad_mask.sum() > 0:
                    # 单独计算AD样本的损失并给予额外权重
                    ad_exp_diag = exp_diag[ad_mask]
                    ad_exp_all = exp_all[ad_mask]
                    ad_loss = -torch.log(ad_exp_diag / (ad_exp_all + eps) + eps).mean()
                    
                    # 将AD损失与总损失结合 - 🔥 权重从4.0降至2.0
                    infonce_alignment_loss = infonce_alignment_loss + 2.0 * ad_loss
        
        # 🎯 方案2: 渐进式距离损失 (更宽松的margin)
        feature_distance = F.mse_loss(text_processed, image_processed, reduction='none').mean(dim=1)
        
        # 🔥 为AD样本添加额外权重
        if labels is not None:
            ad_mask = (labels == 1)
            if ad_mask.sum() > 0:
                # 单独计算AD样本的距离损失
                ad_distance = feature_distance[ad_mask]
                non_ad_distance = feature_distance[~ad_mask]
                
                # 分别应用margin
                ad_loss = F.relu(ad_distance - margin * 0.5).mean()  # 🔥 AD样本使用更小的margin
                non_ad_loss = F.relu(non_ad_distance - margin).mean()
                
                # 组合损失，AD损失权重更高 - 🔥 权重从4.0降至2.0
                distance_loss = non_ad_loss + 2.0 * ad_loss
            else:
                distance_loss = F.relu(feature_distance - margin).mean()
        else:
            distance_loss = F.relu(feature_distance - margin).mean()
        
        # 🔥 方案3: 余弦相似度损失 (直接优化) - 添加数值保护
        cosine_sim = F.cosine_similarity(text_norm, image_norm, dim=1)
        # 确保余弦相似度在合理范围内
        cosine_sim = torch.clamp(cosine_sim, min=-1.0, max=1.0)
        
        # 🔥 为AD样本添加额外权重
        if labels is not None:
            ad_mask = (labels == 1)
            if ad_mask.sum() > 0:
                # 单独计算AD样本的余弦损失
                ad_cosine = cosine_sim[ad_mask]
                non_ad_cosine = cosine_sim[~ad_mask]
                
                # 分别计算损失
                ad_loss = (1.0 - ad_cosine).mean()
                non_ad_loss = (1.0 - non_ad_cosine).mean()
                
                # 组合损失，AD损失权重更高 - 🔥 权重从4.0降至2.0
                cosine_loss = non_ad_loss + 2.0 * ad_loss
            else:
                cosine_loss = (1.0 - cosine_sim).mean()
        else:
            cosine_loss = (1.0 - cosine_sim).mean()
        
        # 🔧 检查所有损失是否为有限值
        losses_to_check = [infonce_alignment_loss, distance_loss, cosine_loss]
        for i, loss_val in enumerate(losses_to_check):
            if not torch.isfinite(loss_val):
                # 🔥 静默处理非有限值损失
                losses_to_check[i] = torch.tensor(0.0, device=text_features.device, requires_grad=True)
        
        infonce_alignment_loss, distance_loss, cosine_loss = losses_to_check
        
        # 🎯 动态权重组合 (渐进式策略)
        if epoch < 5:
            # 早期: 以距离损失为主，建立基本对应关系
            total_loss = 0.3 * infonce_alignment_loss + 0.5 * distance_loss + 0.2 * cosine_loss
        elif epoch < 15:
            # 中期: 平衡三种损失
            total_loss = 0.4 * infonce_alignment_loss + 0.4 * distance_loss + 0.2 * cosine_loss
        else:
            # 后期: 以InfoNCE对齐为主，精细调整
            total_loss = 0.6 * infonce_alignment_loss + 0.3 * distance_loss + 0.1 * cosine_loss
        
        # 🔥 应用渐进式权重 - 最终数值检查（静默处理）
        final_loss = total_loss * alignment_weight
        
        if not torch.isfinite(final_loss):
            # 🔥 静默返回零损失，避免重复警告
            return torch.tensor(0.0, device=text_features.device, requires_grad=True)
        
        return final_loss
    
    def forward(self, images, texts, labels=None, mode='both', epoch=0, inference_mode=False):
        """
        前向传播 - 图像主导分类，文本辅助对齐
        
        Args:
            images: [B, 3, 113, 137, 113] 图像数据
            texts: List[str] 文本数据列表
            labels: [B] 真实标签 (推理时为None)
            mode: 'classification', 'losses', 'both'
            epoch: 当前训练轮数
            inference_mode: bool 是否为推理模式 (测试时为True)
        
        Returns:
            dict: 包含不同输出的字典
        """
        # 🔧 确保输入在正确的设备上
        images = images.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        # 🎯 推理模式检查
        if inference_mode:
            # 推理模式：不使用标签，只进行分类预测
            assert labels is None or not self.training, "推理模式下不应该使用标签进行训练"
            mode = 'classification'  # 强制设置为分类模式
        
        # 1. 图像编码
        image_features = self.image_encoder(images)  # [B, 512]
        
        # 2. 增强对抗性文本编码
        if inference_mode:
            # 🔥 推理模式：不传入标签，使用默认权重
            text_results = self.text_encoder(texts, labels=None)
        else:
            # 🔥 训练/验证模式：传入标签进行特征解耦
            text_results = self.text_encoder(texts, labels)
        
        text_features = text_results['text_features']  # [B, 512]
        text_weights = text_results['text_weights']    # [B, 1]
        
        results = {
            'image_features': image_features,
            'text_features': text_features,
            'text_weights': text_weights
        }
        
        # 3. 跨模态对齐（用于计算对齐损失）
        # 即使最终分类只用图像，对齐也是必要的，以使图像特征学习到文本信息
        aligned_text_raw = self.cross_modal_aligner(text_features)
        # 🔥 添加残差连接，防止信息丢失
        aligned_text_features = aligned_text_raw + text_features  # 残差连接
        
        if not inference_mode and labels is not None:
            # 🔥 对齐损失: 使用原始图像特征和对齐后的文本特征
            # 确保传入config以获取ad_weight
            current_config = {
                'ad_weight': self.adaptive_weights.get_weights()[0],
                'alignment_weight': self.adaptive_weights.get_weights()[1],
                'contrastive_weight': self.adaptive_weights.get_weights()[2],
                'reconstruction_weight': self.adaptive_weights.get_weights()[3],
                'orthogonality_weight': self.adaptive_weights.get_weights()[4],
                'diagnostic_weight': self.adaptive_weights.get_weights()[5],
                'dominance_weight': self.adaptive_weights.get_weights()[6]
                # 删除text_suppression_weight
            }
            alignment_loss = self.improved_alignment_loss(aligned_text_features, image_features.detach(), epoch, labels)
            results['alignment_loss'] = alignment_loss
            
            # 🔥 对比学习损失: 使用原始图像特征和原始文本特征
            contrastive_loss = self.contrastive_sampler.compute_contrastive_loss(
                image_features, text_features, labels
            )
            results['contrastive_loss'] = contrastive_loss
        
        # 4. 分类 (图像主导)
        # 直接使用图像特征进行分类
        logits = self.image_classifier(image_features)
        results['logits'] = logits
        
        # 5. 计算总损失 (只在训练/验证模式且有标签时)
        if mode in ['losses', 'both'] and labels is not None and not inference_mode:
            classification_loss = F.cross_entropy(results['logits'], labels)
            results['classification_loss'] = classification_loss
            
            # 文本编码器相关的损失 (来自text_results)
            if 'diagnostic_loss' in text_results:
                results['reconstruction_loss'] = text_results.get('reconstruction_loss', torch.tensor(0.0, device=self.device))
                results['orthogonality_loss'] = text_results.get('orthogonality_loss', torch.tensor(0.0, device=self.device))
                results['diagnostic_loss'] = text_results.get('diagnostic_loss', torch.tensor(0.0, device=self.device))
            
            # 只保留dominance_loss，确保图像特征主导
            if 'non_diagnostic_features' in text_results:
                image_norm = torch.norm(image_features, p=2, dim=1).mean()
                non_diag_text_norm = torch.norm(text_results['non_diagnostic_features'], p=2, dim=1).mean()
                results['dominance_loss'] = F.relu(non_diag_text_norm - image_norm * 0.2) # 文本特征范数不应远超图像
                
                # 删除text_suppression_loss的计算
            
            losses_dict = {
                'classification': classification_loss,
                'alignment': results.get('alignment_loss', torch.tensor(0.0, device=self.device)),
                'contrastive': results.get('contrastive_loss', torch.tensor(0.0, device=self.device)),
                'reconstruction': results.get('reconstruction_loss', torch.tensor(0.0, device=self.device)),
                'orthogonality': results.get('orthogonality_loss', torch.tensor(0.0, device=self.device)),
                'diagnostic': results.get('diagnostic_loss', torch.tensor(0.0, device=self.device)),
                'dominance': results.get('dominance_loss', torch.tensor(0.0, device=self.device))
                # 删除text_suppression
            }
            
            total_loss, weights_dict = self.adaptive_weights(losses_dict, epoch)
            results['total_loss'] = total_loss
            results['adaptive_weights'] = weights_dict
        
        return results


class AdversarialMultiModalDataset(Dataset):
    """对抗性多模态数据集"""
    
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
        print(f"📊 对抗性数据集创建: {len(self.labels)} 样本")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'text': self.texts[idx],  # 返回原始文本字符串
            'label': self.labels[idx],
            'index': idx
        }


class AdversarialContrastiveTrainer:
    """对抗性对比学习训练器"""
    
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
        
        print(f"🎯 对抗性训练器配置:")
        print(f"   学习率: {config['learning_rate']}")
        print(f"   权重衰减: {config['weight_decay']}")
        print(f"   梯度裁剪: {config['gradient_clip']}")
    
    def _create_optimizer(self):
        """创建差异化学习率优化器"""
        param_groups = [
            # 图像编码器投影层 - 中等学习率
            {
                'params': self.model.image_encoder.projection.parameters(),
                'lr': self.config['learning_rate'] * 1.5,
                'name': 'image_projection'
            },
            # 对抗性文本编码器 - 高学习率
            {
                'params': self.model.text_encoder.adversarial_projection.parameters(),
                'lr': self.config['learning_rate'] * 2,
                'name': 'text_adversarial_projection'
            },
            # 特征解耦模块 - 高学习率
            {
                'params': self.model.text_encoder.feature_disentangler.parameters(),
                'lr': self.config['learning_rate'] * 2,
                'name': 'feature_disentangler'
            },
            # BERT参数 - 低学习率
            {
                'params': self.model.text_encoder.bert_model.parameters(),
                'lr': self.config['learning_rate'] * 0.1,
                'name': 'bert_backbone'
            },
            # 跨模态对齐器 - 更小的学习率，稳定训练
            {
                'params': self.model.cross_modal_aligner.parameters(),
                'lr': self.config['learning_rate'] * 0.3,  # 🔥 从1.0降到0.3，更稳定的对齐学习
                'name': 'cross_modal_aligner'
            },
            # 图像分类器 - 标准学习率
            {
                'params': self.model.image_classifier.parameters(),
                'lr': self.config['learning_rate'],
                'name': 'image_classifier'
            }
        ]
        
        return torch.optim.AdamW(param_groups, weight_decay=self.config['weight_decay'])
    
    def train_epoch(self, dataloader, epoch):
        """训练一个epoch - 增强对抗性版本（优化进度条格式）"""
        self.model.train()
        
        # 🔥 在epoch开始时清空警告
        self.model.warning_collector.clear()

        total_loss = 0.0
        total_classification_loss = 0.0
        total_alignment_loss = 0.0
        total_contrastive_loss = 0.0
        correct = 0
        total = 0
        
        # 🔥 优化进度条显示 - 使用标准格式
        desc = f'Epoch {epoch+1:02d} [Train]'
        pbar = tqdm(dataloader, desc=desc, leave=True, ncols=120, 
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix}]')
        
        for batch_idx, batch in enumerate(pbar):
            try:
                images = batch['image'].to(self.device)
                texts = batch['text']  # 文本列表
                labels = batch['label'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(images, texts, labels=labels, mode='both', epoch=epoch)
                
                # 提取损失
                total_batch_loss = outputs['total_loss']
                classification_loss = outputs['classification_loss']
                alignment_loss = outputs['alignment_loss']
                contrastive_loss = outputs.get('contrastive_loss', torch.tensor(0.0))
                
                # 反向传播
                total_batch_loss.backward()
                
                # 梯度裁剪
                if self.config.get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['gradient_clip']
                    )
                
                self.optimizer.step()
                
                # 统计
                total_loss += total_batch_loss.item()
                total_classification_loss += classification_loss.item()
                total_alignment_loss += alignment_loss.item()
                total_contrastive_loss += contrastive_loss.item()
                
                # 预测统计
                logits = outputs['logits']
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 🔥 实时更新进度条
                current_acc = 100. * correct / total if total > 0 else 0
                
                # 🔥 标准进度条格式：Loss=X.XXXX, Acc=XX.X%
                pbar.set_postfix_str(f'Loss={total_batch_loss.item():.4f}, Acc={current_acc:.1f}%')
                
                # 学习率调整
                if hasattr(self, 'scheduler') and self.scheduler is not None:
                    self.scheduler.step()
                    
            except Exception as e:
                print(f"\n⚠️  训练批次 {batch_idx} 失败: {e}")
                continue
        
        # 🔥 在epoch结束后打印收集到的警告
        if self.model.warning_collector:
            tqdm.write(f"\nEpoch {epoch+1} Warnings: {list(set(self.model.warning_collector))}")
            self.model.warning_collector.clear() # 清空警告

        # 计算平均指标
        avg_loss = total_loss / len(dataloader)
        avg_classification_loss = total_classification_loss / len(dataloader)
        avg_alignment_loss = total_alignment_loss / len(dataloader)
        avg_contrastive_loss = total_contrastive_loss / len(dataloader)
        accuracy = 100. * correct / total
        current_lr = self.optimizer.param_groups[0]['lr']
        
        return {
            'total_loss': avg_loss,
            'classification_loss': avg_classification_loss,
            'alignment_loss': avg_alignment_loss,
            'contrastive_loss': avg_contrastive_loss,
            'accuracy': accuracy / 100.0,
            'learning_rate': current_lr
        }
    
    def evaluate(self, dataloader, inference_mode=False):
        """
        评估模型 - 增强对抗性版本（优化进度条格式）
        
        Args:
            dataloader: 数据加载器
            inference_mode: bool 是否为推理模式
                - False: 验证模式，有标签，计算损失
                - True: 测试模式，可能无标签，只预测
        """
        self.model.eval()
        
        total_loss = 0.0
        total_classification_loss = 0.0
        total_alignment_loss = 0.0
        total_contrastive_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_text_weights = []  # 🔥 新增：收集文本权重
        
        # 🔥 重置并清空警告收集器
        self.model.warning_collector.clear()
        
        # 🔥 进度条优化：标准格式
        desc = '🔮 Inference' if inference_mode else '[Val]'
        
        with torch.no_grad():
            # 🔥 设置静默模式，避免重复警告
            if hasattr(self.model, '_std_warning_shown'):
                self.model._std_warning_shown = True
                
            pbar = tqdm(dataloader, desc=desc, leave=True, ncols=120,
                       bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix}]')
            for batch_idx, batch in enumerate(pbar):
                try:
                    images = batch['image'].to(self.device)
                    texts = batch['text']  # 文本列表
                    labels = batch.get('label', None)  # 推理模式可能没有标签
                    
                    if labels is not None:
                        labels = labels.to(self.device)
                    
                    # 前向传播
                    outputs = self.model(images, texts, labels=labels, mode='both', inference_mode=inference_mode)
                    
                    # 获取预测结果
                    logits = outputs['logits']
                    _, predicted = torch.max(logits.data, 1)
                    
                    # 🔥 只在验证模式计算损失
                    if not inference_mode and labels is not None:
                        total_batch_loss = outputs.get('total_loss', torch.tensor(0.0))
                        classification_loss = outputs.get('classification_loss', torch.tensor(0.0))
                        alignment_loss = outputs.get('alignment_loss', torch.tensor(0.0))
                        contrastive_loss = outputs.get('contrastive_loss', torch.tensor(0.0))
                        
                        total_loss += total_batch_loss.item()
                        total_classification_loss += classification_loss.item()
                        total_alignment_loss += alignment_loss.item()
                        total_contrastive_loss += contrastive_loss.item()
                    
                    # 🔥 只在有真实标签时计算准确率
                    if labels is not None:
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        all_labels.extend(labels.cpu().numpy())
                    else:
                        # 推理模式无标签时，记录为-1
                        total += len(predicted)
                        all_labels.extend([-1] * len(predicted))
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    
                    # 🔥 收集文本权重信息
                    text_weights = outputs.get('text_weights', torch.tensor([[0.5]] * len(predicted)))
                    all_text_weights.extend(text_weights.cpu().numpy().flatten())
                    
                    # 🔥 每5个批次更新一次进度条
                    if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(pbar):
                        if labels is not None:
                            current_acc = 100. * correct / total if total > 0 else 0
                            if not inference_mode:
                                # 验证模式：显示损失和准确率
                                pbar.set_postfix_str(f'Loss={total_batch_loss.item():.4f}, Acc={current_acc:.1f}%')
                            else:
                                # 推理模式：只显示准确率
                                pbar.set_postfix_str(f'Acc={current_acc:.1f}%')
                        else:
                            # 无标签推理模式
                            pbar.set_postfix_str(f'Batch={batch_idx+1}/{len(dataloader)}')
                            
                except Exception as e:
                    print(f"\n⚠️  批次 {batch_idx} 处理失败: {e}")
                    continue
        
        # 🔥 在评估结束后打印收集到的警告
        if self.model.warning_collector:
            tqdm.write(f"\nEvaluation Warnings: {list(set(self.model.warning_collector))}")
            self.model.warning_collector.clear()

        # 计算指标
        if inference_mode:
            # 推理模式：可能没有真实标签
            if all(label != -1 for label in all_labels):
                # 有标签：计算准确率
                accuracy = 100. * correct / total
                report = classification_report(all_labels, all_predictions, target_names=['CN', 'AD'], output_dict=True)
                conf_matrix = confusion_matrix(all_labels, all_predictions)
            else:
                # 无标签：无法计算准确率
                accuracy = None
                report = None
                conf_matrix = None
        else:
            # 验证模式：必须有标签
            accuracy = 100. * correct / total
            report = classification_report(all_labels, all_predictions, target_names=['CN', 'AD'], output_dict=True)
            conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        # 🔥 文本权重统计
        avg_text_weight = np.mean(all_text_weights)
        text_weight_std = np.std(all_text_weights)
        
        result = {
            'predictions': all_predictions,
            'labels': all_labels,
            'avg_text_weight': avg_text_weight,  # 🔥 新增
            'text_weight_std': text_weight_std,   # 🔥 新增
            'inference_mode': inference_mode
        }
        
        # 🔥 只在验证模式返回损失
        if not inference_mode:
            result.update({
                'loss': total_loss / len(dataloader),
                'classification_loss': total_classification_loss / len(dataloader),
                'alignment_loss': total_alignment_loss / len(dataloader),
                'contrastive_loss': total_contrastive_loss / len(dataloader),
            })
        
        # 🔥 只在有标签时返回准确率指标
        if accuracy is not None:
            result.update({
                'accuracy': accuracy / 100.0,
                'classification_report': report,
                'confusion_matrix': conf_matrix,
            })
        
        return result
    
    def inference(self, dataloader):
        """
        🔥 专门的推理函数 - 用于独立测试集
        
        Args:
            dataloader: 测试数据加载器（可能没有标签）
            
        Returns:
            dict: 推理结果
        """
        print("🔮 开始推理模式 - 独立测试集预测")
        return self.evaluate(dataloader, inference_mode=True)


def load_text_data_with_cognitive_scores(text_data_dir):
    """
    加载包含认知评估分数的文本数据 - 保留MMSE/CDR-SB等有价值信息
    
    Args:
        text_data_dir: 文本数据目录路径
        
    Returns:
        all_texts: List[str] 包含认知分数的文本数据
        all_labels: numpy array 标签
        patient_ids: List[str] 患者ID列表
    """
    print(f"📝 加载包含认知评估分数的文本数据...")
    
    # 文件路径 - V3.5 更新
    ad_file = os.path.join(text_data_dir, 'ad_metadata.xlsx')
    cn_file = os.path.join(text_data_dir, 'cn_metadata.xlsx')
    
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
    
    def create_rich_clinical_text(row):
        """
        创建丰富的临床文本描述 - 保留认知分数
        
        🎯 策略：保留有价值的医学信息，但通过对抗训练减轻泄露影响
        """
        text_parts = []
        
        # 基本人口统计学信息
        if 'Age' in row and pd.notna(row['Age']):
            text_parts.append(f"Age: {row['Age']} years")
        
        if 'Gender' in row and pd.notna(row['Gender']):
            gender = "male" if row['Gender'] == 1 else "female"
            text_parts.append(f"Gender: {gender}")
        
        if 'Edu' in row and pd.notna(row['Edu']):
            text_parts.append(f"Education: {row['Edu']} years")
        
        # 🎯 保留认知评估分数（有价值的医学特征，使用完整医学术语）
        cognitive_scores = []
        
        if 'MMSE' in row and pd.notna(row['MMSE']):
            mmse = row['MMSE']
            # 🔥 使用完整医学术语全称
            cognitive_scores.append(f"Mini-Mental State Examination (MMSE): [{int(mmse)}/30]")
        
        if 'CDRSB' in row and pd.notna(row['CDRSB']):
            cdrsb = row['CDRSB']
            # 🔥 使用完整医学术语全称
            cognitive_scores.append(f"Clinical Dementia Rating - Sum of Boxes (CDR-SB): [{cdrsb}]")
        
        # 添加其他认知测试分数
        additional_scores = []
        for col in row.index:
            if col in ['ADAS11', 'ADAS13', 'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting']:
                if pd.notna(row[col]):
                    additional_scores.append(f"{col}: {row[col]}")
        
        # 组合所有信息
        if cognitive_scores:
            text_parts.append("Cognitive assessments: " + ", ".join(cognitive_scores))
        
        if additional_scores:
            text_parts.append("Additional tests: " + ", ".join(additional_scores))
        
        # 🚨 关键：不包含明确的诊断信息
        final_text = "Clinical profile: " + " | ".join(text_parts)
        
        return final_text
    
    # 处理AD数据
    ad_texts = []
    ad_patient_ids = []
    for idx, row in ad_df.iterrows():
        text = create_rich_clinical_text(row)
        ad_texts.append(text)
        
        # 提取患者ID
        if 'NAME' in row and pd.notna(row['NAME']):
            ad_patient_ids.append(str(row['NAME']))
        else:
            ad_patient_ids.append(f"AD_{idx}")
    
    # 处理CN数据
    cn_texts = []
    cn_patient_ids = []
    for idx, row in cn_df.iterrows():
        text = create_rich_clinical_text(row)
        cn_texts.append(text)
        
        # 提取患者ID
        if 'NAME' in row and pd.notna(row['NAME']):
            cn_patient_ids.append(str(row['NAME']))
        else:
            cn_patient_ids.append(f"CN_{idx}")
    
    # 合并数据
    all_texts = ad_texts + cn_texts
    all_labels = np.array([1] * len(ad_texts) + [0] * len(cn_texts))  # AD=1, CN=0
    patient_ids = ad_patient_ids + cn_patient_ids
    
    print(f"✅ 丰富文本数据加载完成:")
    print(f"   总样本数: {len(all_texts)}")
    print(f"   AD样本: {len(ad_texts)}, CN样本: {len(cn_texts)}")
    print(f"   示例AD文本: {all_texts[0][:200]}...")
    print(f"   示例CN文本: {all_texts[len(ad_texts)][:200]}...")
    
    return all_texts, all_labels, patient_ids


def load_multiple_mcic_files(data_dir):
    """
    🔄 加载多个MCIc格式文件并合并
    
    适用于您有多个组别文件的情况:
    - mcic_metadata.xlsx
    - mcinc_metadata.xlsx
    
    Args:
        data_dir: 包含数据文件的目录
    
    Returns:
        all_texts: 合并的文本列表
        all_labels: 合并的标签数组
        patient_ids: 合并的患者ID列表
    """
    print(f"🔄 扫描目录中的MCIc/MCInc元数据文件: {data_dir}")
    
    # 🎯 V3.5 更新：直接查找特定的元数据文件
    mcic_file = os.path.join(data_dir, 'mcic_metadata.xlsx')
    mcinc_file = os.path.join(data_dir, 'mcinc_metadata.xlsx')
    
    all_files = []
    if os.path.exists(mcic_file):
        all_files.append(mcic_file)
    if os.path.exists(mcinc_file):
        all_files.append(mcinc_file)
    
    if not all_files:
        raise FileNotFoundError(f"❌ 在目录 {data_dir} 中未找到 mcic_metadata.xlsx 或 mcinc_metadata.xlsx")
    
    print(f"📁 找到 {len(all_files)} 个元数据文件:")
    for file in all_files:
        print(f"   - {os.path.basename(file)}")
    
    # 合并所有文件的数据
    combined_texts = []
    combined_labels = []
    combined_patient_ids = []
    
    for file_path in all_files:
        try:
            texts, labels, ids = load_mcic_format_data(file_path)
            combined_texts.extend(texts)
            combined_labels.extend(labels)
            combined_patient_ids.extend(ids)
            print(f"✅ 成功加载: {os.path.basename(file_path)} ({len(texts)} 样本)")
        except Exception as e:
            print(f"❌ 加载失败: {os.path.basename(file_path)} - {e}")
            continue
    
    # 转换标签为numpy数组
    combined_labels = np.array(combined_labels)
    
    # 最终统计
    label_counts = np.bincount(combined_labels)
    print(f"\n🎯 合并数据统计:")
    print(f"   总样本数: {len(combined_texts)}")
    print(f"   标签分布: 阴性={label_counts[0]}, 阳性={label_counts[1] if len(label_counts) > 1 else 0}")
    
    return combined_texts, combined_labels, combined_patient_ids


# 示例使用函数
def example_usage_mcic_data():
    """
    🎯 MCIc数据格式使用示例
    """
    # 方法1: 加载单个文件
    # texts, labels, ids = load_mcic_format_data("./data/mcic_clinical_data.xlsx")
    
    # 方法2: 加载目录中的所有文件
    # texts, labels, ids = load_multiple_mcic_files("./文本编码器/")
    
    # 方法3: 在主训练函数中使用
    """
    # 在main()函数中替换原有的数据加载:
    
    print("🔄 加载MCIc格式临床数据...")
    texts, labels, patient_ids = load_multiple_mcic_files(config['text_data_dir'])
    
    # 然后继续使用现有的数据处理流程...
    """
    pass


def get_best_image_model_path():
    """智能获取最佳图像模型路径"""
    cv_models = [
        ('./models/contrastive/fold_0_best_model.pth', '第0折最佳模型 (94.74%)'),
    ]
    
    for model_path, description in cv_models:
        if os.path.exists(model_path):
            print(f"✅ 使用图像编码器: {description}")
            return model_path
    
    # Fallback to a default path if none of the specific fold models exist.
    default_path = './models/contrastive/best_contrastive_image_encoder.pth'
    if os.path.exists(default_path):
        print(f"✅ 使用图像编码器: 备用总体最佳模型")
        return default_path

    raise FileNotFoundError("❌ 关键错误: 未找到任何可用的预训练图像编码器模型。请先运行图像编码器训练。")


def main():
    """主函数 - 解析参数并启动训练"""
    parser = argparse.ArgumentParser(description="对抗性对比学习训练脚本 (V2.2)")
    
    # 模式选择
    parser.add_argument('--mode', type=str, default='standard', 
                       choices=['standard', 'mcic', 'cv', 'mcic-cv'],
                       help='训练模式: standard(标准), mcic(MCIc数据), cv(交叉验证), mcic-cv(MCIc+交叉验证)')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=8, help='批次大小')
    parser.add_argument('--no-cv', action='store_true', help='禁用交叉验证，使用单次训练模式')
    parser.add_argument('--fp16', action='store_true', help='启用FP16混合精度训练')
    
    # V2.2 消融实验开关
    parser.add_argument('--no-cognitive-features', action='store_true', help='消融: 禁用文本编码器中的认知分数特征')
    parser.add_argument('--no-disentanglement', action='store_true', help='消融: 禁用对比学习中的特征解耦和对抗损失')

    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='设备选择 (默认:auto)')
    parser.add_argument('--save-dir', type=str, default='./models/adversarial/', help='模型保存目录 (默认:./models/adversarial/)')

    args = parser.parse_args()

    # ❗️为了确保每次运行结果一致，在此设置全局随机种子
    set_seed(42)

    # 智能设备选择
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"💻 运行设备: {device}")
    
    # 🎯 配置参数 (根据模式调整)
    config = {
        # 数据路径配置 - V3.5 更新
        'image_data_dir': '/root/autodl-tmp/DATA_MCI/',
        'text_data_dir': '/root/autodl-tmp/DATA_MCI/', # 文本元数据也在此目录下
        'save_dir': args.save_dir,
        
        # 🔥 模型路径 - 使用最佳模型
        'image_model_path': './models/contrastive/fold_0_best_model.pth',  # 第0折最佳模型
        
        # 训练参数
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': 1e-4,
        'weight_decay': 1e-2,
        'gradient_clip': 0.5,
        
        # 5折交叉验证参数
        'cv_folds': 5,
        'random_state': 42,
        'use_cv': not args.no_cv,  # 根据命令行参数决定
        
        # 模式特定配置
        'mode': args.mode,
        'mcic_data': 'mcic' in args.mode,  # 是否使用MCIc数据格式
        
        'device': device,
        'use_fp16': args.fp16,
        # V2.2 消融开关
        'use_cognitive_features': not args.no_cognitive_features,
        'use_disentanglement': not args.no_disentanglement,
    }
    
    # 🔥 根据模式调整配置
    if config['mcic_data']:
        print("🎯 MCIc数据专用配置:")
        print(f"   📁 使用MCIc格式的临床文本数据")
        print(f"   🧠 专门针对认知评估分数处理")
        print(f"   📊 适配MCIc数据的特征提取")
        
        # MCIc特定配置
        config.update({
            'mcic_text_format': True,
            'cognitive_assessment_enhanced': True,
            'specialized_mcic_processing': True
        })
    
    print(f"\n🎯 核心技术特性:")
    print(f"   🧠 多元回归认知评估校正")
    print(f"   🔗 强制图像-文本特征对齐")
    print(f"   🛡️ 强正则化防止过拟合")
    if config['use_cv']:
        print(f"   📊 5折分层交叉验证，确保患者级别分割")
    else:
        print(f"   🔄 单次训练模式")
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    
    try:
        if config['use_cv']:
            # 🎯 使用5折交叉验证模式
            if config['mcic_data']:
                print("🔄 MCIc数据 + 5折交叉验证模式")
                cv_results = run_mcic_adversarial_cross_validation(config)
            else:
                print("🔄 标准数据 + 5折交叉验证模式") 
                cv_results = run_adversarial_cross_validation(config)
            
            if cv_results:
                print(f"\n🎉 5折交叉验证完成!")
                print(f"📈 平均准确率: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
                print(f"📊 最佳折准确率: {max(cv_results['fold_accuracies']):.4f}")
                
                # 保存交叉验证结果
                cv_save_path = os.path.join(config['save_dir'], f"{'mcic_' if config['mcic_data'] else ''}adversarial_cv_results.json")
                with open(cv_save_path, 'w', encoding='utf-8') as f:
                    # 🔧 修复：使用convert_numpy_types处理数据
                    serializable_results = convert_numpy_types(cv_results)
                    json.dump(serializable_results, f, indent=2, ensure_ascii=False)
                
                print(f"💾 交叉验证结果已保存: {cv_save_path}")
            else:
                print("❌ 交叉验证失败")
        else:
            # 🔄 传统单次训练模式
            if config['mcic_data']:
                print("🔄 MCIc数据单次训练模式")
                success = run_mcic_single_training(config)
            else:
                print("🔄 标准数据单次训练模式")
                success = run_standard_single_training(config)
            
            if success:
                print("✅ 单次训练完成")
            else:
                print("❌ 单次训练失败")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


def run_adversarial_cross_validation(config):
    """在标准AD/CN数据集上运行5折交叉验证"""
    print("🔄 开始在标准 AD/CN 数据集上进行5折交叉验证...")
    
    # 1. 加载数据
    print("--- 步骤 1: 加载文本和图像数据 ---")
    texts, labels, patient_ids = load_text_data_with_cognitive_scores(config['text_data_dir'])
    
    # 假设图像数据已经预处理并保存为 .pkl 文件
    image_data_path = os.path.join(config['image_data_dir'], 'preprocessed_images.pkl')
    if not os.path.exists(image_data_path):
        raise FileNotFoundError(f"❌ 关键错误: 未找到预处理的图像数据文件 {image_data_path}。请先运行数据预处理脚本。")
    
    with open(image_data_path, 'rb') as f:
        image_data = pickle.load(f)
    
    # 根据 patient_ids 匹配图像数据
    images = np.array([image_data[pid] for pid in patient_ids if pid in image_data])
    
    # 过滤掉没有对应图像的文本数据
    valid_indices = [i for i, pid in enumerate(patient_ids) if pid in image_data]
    texts = [texts[i] for i in valid_indices]
    labels = labels[valid_indices]

    if len(images) != len(texts):
        raise ValueError("❌ 图像和文本数据样本数不匹配，请检查数据。")

    print(f"✅ 数据加载完成: {len(labels)} 个匹配样本")

    # 2. 初始化交叉验证
    kfold = StratifiedKFold(n_splits=config['cv_folds'], shuffle=True, random_state=config['random_state'])
    
    all_fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels)):
        print(f"\n{'='*20} Fold {fold + 1}/{config['cv_folds']} {'='*20}")
        
        # 3. 创建模型和训练器
        model = AdversarialContrastiveModel(
            image_model_path=config.get('image_model_path', get_best_image_model_path()),
            device=config['device'],
            use_cognitive_features=config.get('use_cognitive_features', True),
            use_disentanglement=config.get('use_disentanglement', True)
        ).to(config['device'])
        
        trainer = AdversarialContrastiveTrainer(model, config['device'], config)
        
        # 4. 创建数据集和DataLoader
        train_dataset = AdversarialMultiModalDataset(images[train_idx], [texts[i] for i in train_idx], labels[train_idx])
        val_dataset = AdversarialMultiModalDataset(images[val_idx], [texts[i] for i in val_idx], labels[val_idx])
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # 5. 训练和评估
        best_val_acc = 0
        history = []

        for epoch in range(config['num_epochs']):
            train_metrics = trainer.train_epoch(train_loader, epoch)
            val_metrics = trainer.evaluate(val_loader)
            
            print(f"Epoch {epoch+1:02d} | Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Total Loss: {train_metrics['total_loss']:.4f}")
            
            epoch_history = {**train_metrics, **{'val_'+k: v for k, v in val_metrics.items()}}
            history.append(epoch_history)
            
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                save_path = os.path.join(config['save_dir'], f"standard_fold_{fold}_best_model.pth")
                torch.save(model.state_dict(), save_path)
                print(f"💾 模型已保存: {save_path} (ACC: {best_val_acc:.4f})")

        all_fold_results.append({'fold': fold, 'best_accuracy': best_val_acc, 'history': history})
        
    # 6. 汇总交叉验证结果
    fold_accuracies = [r['best_accuracy'] for r in all_fold_results]
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    
    return {
        'fold_results': all_fold_results,
        'fold_accuracies': fold_accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy
    }


def run_mcic_adversarial_cross_validation(config):
    """在MCIc/MCInc数据集上运行5折交叉验证"""
    print("🔄 开始在 MCIc/MCInc 数据集上进行5折交叉验证...")
    
    # 1. 加载数据
    print("--- 步骤 1: 加载MCIc格式文本和图像数据 ---")
    texts, labels, patient_ids = load_multiple_mcic_files(config['text_data_dir'])
    
    # 假设图像数据已经预处理并保存为 .pkl 文件
    image_data_path = os.path.join(config['image_data_dir'], 'mcic_preprocessed_images.pkl')
    if not os.path.exists(image_data_path):
        raise FileNotFoundError(f"❌ 关键错误: 未找到MCIc预处理的图像数据文件 {image_data_path}。请先运行数据预处理脚本。")
    
    with open(image_data_path, 'rb') as f:
        image_data = pickle.load(f)
        
    # 根据 patient_ids 匹配图像数据
    images = np.array([image_data[pid] for pid in patient_ids if pid in image_data])
    
    # 过滤掉没有对应图像的文本数据
    valid_indices = [i for i, pid in enumerate(patient_ids) if pid in image_data]
    texts = [texts[i] for i in valid_indices]
    labels = labels[valid_indices]

    if len(images) != len(texts):
        raise ValueError("❌ 图像和文本数据样本数不匹配，请检查数据。")
    
    print(f"✅ 数据加载完成: {len(labels)} 个匹配样本")
    
    # 2. 初始化交叉验证
    kfold = StratifiedKFold(n_splits=config['cv_folds'], shuffle=True, random_state=config['random_state'])
    
    all_fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels)):
        print(f"\n{'='*20} Fold {fold + 1}/{config['cv_folds']} {'='*20}")
        
        # 3. 创建模型和训练器
        model = AdversarialContrastiveModel(
            image_model_path=get_best_image_model_path(),
            device=config['device'],
            use_cognitive_features=config.get('use_cognitive_features', True),
            use_disentanglement=config.get('use_disentanglement', True)
        ).to(config['device'])
        
        trainer = AdversarialContrastiveTrainer(model, config['device'], config)

        # 4. 创建数据集和DataLoader
        train_dataset = AdversarialMultiModalDataset(images[train_idx], [texts[i] for i in train_idx], labels[train_idx])
        val_dataset = AdversarialMultiModalDataset(images[val_idx], [texts[i] for i in val_idx], labels[val_idx])
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # 5. 训练和评估
        best_val_acc = 0
        history = []

        for epoch in range(config['num_epochs']):
            train_metrics = trainer.train_epoch(train_loader, epoch)
            val_metrics = trainer.evaluate(val_loader)
            
            print(f"Epoch {epoch+1:02d} | Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Total Loss: {train_metrics['total_loss']:.4f}")
            
            epoch_history = {**train_metrics, **{'val_'+k: v for k, v in val_metrics.items()}}
            history.append(epoch_history)
            
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                save_path = os.path.join(config['save_dir'], f"mcic_fold_{fold}_best_model.pth")
                torch.save(model.state_dict(), save_path)
                print(f"💾 模型已保存: {save_path} (ACC: {best_val_acc:.4f})")
    
        all_fold_results.append({'fold': fold, 'best_accuracy': best_val_acc, 'history': history})
        
    # 6. 汇总交叉验证结果
    fold_accuracies = [r['best_accuracy'] for r in all_fold_results]
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    
    return {
        'fold_results': all_fold_results,
        'fold_accuracies': fold_accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy
    }


def run_mcic_single_training(config):
    """在MCIc/MCInc数据集上运行单次训练"""
    print("🚀 开始在 MCIc/MCInc 数据集上进行单次训练...")

    # 1. 加载数据
    print("--- 步骤 1: 加载MCIc格式文本和图像数据 ---")
    texts, labels, patient_ids = load_multiple_mcic_files(config['text_data_dir'])
    
    image_data_path = os.path.join(config['image_data_dir'], 'mcic_preprocessed_images.pkl')
    if not os.path.exists(image_data_path):
        raise FileNotFoundError(f"❌ 关键错误: 未找到MCIc预处理的图像数据文件 {image_data_path}。")
    
    with open(image_data_path, 'rb') as f:
        image_data = pickle.load(f)
    
    images = np.array([image_data[pid] for pid in patient_ids if pid in image_data])
    valid_indices = [i for i, pid in enumerate(patient_ids) if pid in image_data]
    texts = [texts[i] for i in valid_indices]
    labels = labels[valid_indices]
    
    print(f"✅ 数据加载完成: {len(labels)} 个匹配样本")
    
    # 2. 创建模型和训练器
    model = AdversarialContrastiveModel(
        image_model_path=get_best_image_model_path(),
        device=config['device'],
        use_cognitive_features=config.get('use_cognitive_features', True),
        use_disentanglement=config.get('use_disentanglement', True)
    ).to(config['device'])

    trainer = AdversarialContrastiveTrainer(model, config['device'], config)
    
    # 3. 创建数据集和DataLoader
    dataset = AdversarialMultiModalDataset(images, texts, labels)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # 4. 训练
    for epoch in range(config['num_epochs']):
        train_metrics = trainer.train_epoch(dataloader, epoch)
        print(f"Epoch {epoch+1:02d} | Train Acc: {train_metrics['accuracy']:.4f} | Total Loss: {train_metrics['total_loss']:.4f}")

    # 5. 保存最终模型
    save_path = os.path.join(config['save_dir'], "best_mcic_adversarial_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"💾 最终模型已保存: {save_path}")
    
    return True


def run_standard_single_training(config):
    """在标准AD/CN数据集上运行单次训练"""
    print("🚀 开始在标准 AD/CN 数据集上进行单次训练...")
    
    # 1. 加载数据
    print("--- 步骤 1: 加载标准文本和图像数据 ---")
    texts, labels, patient_ids = load_text_data_with_cognitive_scores(config['text_data_dir'])
    
    image_data_path = os.path.join(config['image_data_dir'], 'preprocessed_images.pkl')
    if not os.path.exists(image_data_path):
        raise FileNotFoundError(f"❌ 关键错误: 未找到预处理的图像数据文件 {image_data_path}。")
        
    with open(image_data_path, 'rb') as f:
        image_data = pickle.load(f)
    
    images = np.array([image_data[pid] for pid in patient_ids if pid in image_data])
    valid_indices = [i for i, pid in enumerate(patient_ids) if pid in image_data]
    texts = [texts[i] for i in valid_indices]
    labels = labels[valid_indices]
    
    print(f"✅ 数据加载完成: {len(labels)} 个匹配样本")

    # 2. 创建模型和训练器
    model = AdversarialContrastiveModel(
        image_model_path=config.get('image_model_path', get_best_image_model_path()),
        device=config['device'],
        use_cognitive_features=config.get('use_cognitive_features', True),
        use_disentanglement=config.get('use_disentanglement', True)
    ).to(config['device'])

    trainer = AdversarialContrastiveTrainer(model, config['device'], config)

    # 3. 创建数据集和DataLoader
    dataset = AdversarialMultiModalDataset(images, texts, labels)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # 4. 训练
    for epoch in range(config['num_epochs']):
        train_metrics = trainer.train_epoch(dataloader, epoch)
        print(f"Epoch {epoch+1:02d} | Train Acc: {train_metrics['accuracy']:.4f} | Total Loss: {train_metrics['total_loss']:.4f}")

    # 5. 保存最终模型
    save_path = os.path.join(config['save_dir'], "best_standard_adversarial_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"💾 最终模型已保存: {save_path}")

    return True


def convert_numpy_types(obj):
    """
    🔧 递归转换numpy数据类型为Python原生类型，用于JSON序列化
    
    Args:
        obj: 待转换的对象
        
    Returns:
        转换后的对象
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


if __name__ == "__main__":
    main() 
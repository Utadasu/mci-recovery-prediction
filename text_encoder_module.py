#!/usr/bin/env python3
"""
独立的文本编码器模块
======================
包含从对抗性学习脚本中移植过来的文本处理模块，
用于下游任务的特征提取。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from transformers import AutoModel, AutoTokenizer

# ==============================================================================
# 模块 1: 认知评估处理器
# ==============================================================================
class CognitiveAssessmentProcessor(nn.Module):
    """🔥 认知评估处理器 - 多元回归校正 + CDR-SB整合"""
    
    def __init__(self, device='cuda'):
        super(CognitiveAssessmentProcessor, self).__init__()
        
        self.device = device
        
        # 🎯 MMSE多元回归校正参数 (基于循证医学研究)
        self.mmse_regression_params = {
            'intercept': 29.1, 'age_coef': -0.045, 'age_squared_coef': -0.0003,
            'gender_coef': 0.1, 'education_coef': 0.35, 'education_squared_coef': -0.008
        }
        
        # 🎯 CDR-SB分箱策略
        self.cdrsb_bins = {
            'normal': [0, 0.5], 'questionable': [0.5, 2.5], 'mild': [2.5, 4.5],
            'moderate': [4.5, 9.0], 'severe': [9.0, 18.0]
        }
        
        # 定义网络层
        self.mmse_encoder = nn.Sequential(nn.Linear(2, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 64))
        self.cdrsb_bin_embedding = nn.Embedding(5, 32)
        self.cdrsb_encoder = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 32))
        self.cognitive_fusion = nn.Sequential(
            nn.Linear(64 + 32 + 32, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 16)
        )
        
    def extract_demographic_info(self, texts):
        demographics = []
        for text in texts:
            age_match = re.search(r'age (\d+\.?\d*)', text)
            gender_match = re.search(r'gender (\d)', text)
            edu_match = re.search(r'education (\d+)', text)
            
            age = float(age_match.group(1)) if age_match else 70.0
            gender = int(gender_match.group(1)) if gender_match else 0
            education = int(edu_match.group(1)) if edu_match else 12
            demographics.append({'age': age, 'gender': gender, 'education': education})
        return demographics

    def extract_mmse_scores(self, texts):
        scores = [float(re.search(r'mmse_score (\d+\.?\d*)', t).group(1)) if re.search(r'mmse_score (\d+\.?\d*)', t) else 25.0 for t in texts]
        return torch.tensor(scores, device=self.device, dtype=torch.float32)

    def extract_cdrsb_scores(self, texts):
        scores = [float(re.search(r'cdrsb_score (\d+\.?\d*)', t).group(1)) if re.search(r'cdrsb_score (\d+\.?\d*)', t) else 1.0 for t in texts]
        return torch.tensor(scores, device=self.device, dtype=torch.float32)

    def compute_mmse_correction(self, demographics):
        p = self.mmse_regression_params
        corrections = []
        for d in demographics:
            correction = (p['age_coef'] * (d['age'] - 70) + p['age_squared_coef'] * ((d['age'] - 70)**2) +
                          p['gender_coef'] * d['gender'] +
                          p['education_coef'] * (d['education'] - 12) + p['education_squared_coef'] * ((d['education'] - 12)**2))
            corrections.append(correction)
        return torch.tensor(corrections, device=self.device, dtype=torch.float32)

    def get_cdrsb_bins(self, cdrsb_scores):
        bins = torch.zeros_like(cdrsb_scores, dtype=torch.long)
        for i, score in enumerate(cdrsb_scores):
            if self.cdrsb_bins['normal'][0] <= score < self.cdrsb_bins['normal'][1]: bins[i] = 0
            elif score < self.cdrsb_bins['questionable'][1]: bins[i] = 1
            elif score < self.cdrsb_bins['mild'][1]: bins[i] = 2
            elif score < self.cdrsb_bins['moderate'][1]: bins[i] = 3
            else: bins[i] = 4
        return bins

    def forward(self, texts):
        demographics = self.extract_demographic_info(texts)
        mmse_scores = self.extract_mmse_scores(texts)
        cdrsb_scores = self.extract_cdrsb_scores(texts)
        
        mmse_corrections = self.compute_mmse_correction(demographics)
        corrected_mmse = mmse_scores - mmse_corrections
        
        mmse_input = torch.stack([(mmse_scores - 15.0) / 15.0, (corrected_mmse - 15.0) / 15.0], dim=1)
        mmse_features = self.mmse_encoder(mmse_input)
        
        cdrsb_bins = self.get_cdrsb_bins(cdrsb_scores)
        cdrsb_bin_features = self.cdrsb_bin_embedding(cdrsb_bins)
        cdrsb_continuous_features = self.cdrsb_encoder( ((cdrsb_scores - 4.5) / 4.5).unsqueeze(1) )
        
        combined_features = torch.cat([mmse_features, cdrsb_bin_features, cdrsb_continuous_features], dim=1)
        return self.cognitive_fusion(combined_features)


# ==============================================================================
# 模块 2: 对抗性文本编码器
# ==============================================================================
class AdversarialTextEncoder(nn.Module):
    """🔥 对抗性文本编码器 - 适配下游任务"""
    def __init__(self, feature_dim=512, device='cuda', max_length=512, use_cognitive_features=True, bert_model_path='../models/bert-base-uncased-local'):
        super(AdversarialTextEncoder, self).__init__()
        
        self.device = device
        self.max_length = max_length
        self.feature_dim = feature_dim
        self.use_cognitive_features = use_cognitive_features
        
        print("🔧 初始化对抗性文本编码器...")
        # 1. BERT模型和分词器
        try:
            print(f"   尝试从本地路径加载BERT: {bert_model_path}")
            self.bert_model = AutoModel.from_pretrained(bert_model_path)
            self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
            print("   ✅ 本地BERT加载成功。")
        except Exception as e:
            print(f"   ⚠️ 本地加载失败: {e}。回退到在线下载'bert-base-uncased'...")
            self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
            self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            print("   ✅ 在线BERT加载成功。")

        # 2. 认知评估处理器
        self.mmse_processor = CognitiveAssessmentProcessor(device=device)
        
        # 3. BERT与认知特征融合层
        fusion_input_dim = 768 + 16 if self.use_cognitive_features else 768
        self.bert_mmse_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024), nn.LayerNorm(1024), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(1024, 768)
        )
        
        # 4. 最终投影层
        self.final_projection = nn.Sequential(
            nn.Linear(768, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(1024, feature_dim)
        )
        
    def forward(self, texts):
        # BERT编码
        inputs = self.bert_tokenizer(
            texts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length
        ).to(self.device)
        bert_features = self.bert_model(**inputs)[0][:, 0, :]

        if self.use_cognitive_features:
            # 确保在批处理大小为1时，也能正确处理
            if bert_features.size(0) == 1 and len(texts) > 0:
                 cognitive_features = self.mmse_processor(texts)
                 if cognitive_features.size(0) > 1:
                     cognitive_features = cognitive_features[0].unsqueeze(0)
            else:
                 cognitive_features = self.mmse_processor(texts)
                 
            features_to_fuse = torch.cat([bert_features, cognitive_features], dim=1)
            fused_features = self.bert_mmse_fusion(features_to_fuse)
        else:
            fused_features = bert_features
        
        projected_features = self.final_projection(fused_features)
        return F.normalize(projected_features, p=2, dim=1) 
#!/usr/bin/env python3
"""
🎯 MCI恢复预测系统
==================

基于已训练的对抗性对比学习模型进行MCI恢复预测
- 使用预训练模型提取图像和文本特征
- 训练下游分类器预测MCI患者是否恢复为认知正常(CN)
- 支持交叉验证和独立测试
- 计算详细的性能指标和ROC曲线
- 支持模型保存和加载

版本: 1.0.0
日期: 2025-12-23
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from collections import defaultdict
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import re
import time
from datetime import datetime
import random

# 导入已有的模型
from adversarial_contrastive_learning import AdversarialContrastiveModel
from text_encoder_module import AdversarialTextEncoder

warnings.filterwarnings('ignore')

# 设置中文字体和避免编码问题
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# 全局随机种子，确保可重复性
RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    """设置随机种子，确保所有库的结果可重复"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 全局随机种子已设置为: {seed}")

# 推荐配置
RECOMMENDED_CONFIG = {
    'fusion_strategy': 'image_only',  # 使用图像特征进行分类
    'classifier_type': 'xgb',  # 使用XGBoost模型
    'regularization_strength': 10.0,
    'cross_validation': 'kfold',  # 使用K折交叉验证
    'kfold_splits': 5,
    'feature_standardization': True,
    'max_iter': 6000,
    'feature_noise_std': 0.0,
    'expected_accuracy_range': (0.75, 0.90),
    'ensemble_voting': True,
    'use_region_ensemble': False,
    'probability_calibration': True,
    'temperature_scaling': True,
    'focal_loss_gamma': 2.0,
    'label_smoothing': 0.1,
    'adaptive_regularization': True,
    'data_augmentation': True,
    'max_acceptable_random_acc': 0.60,
    'min_accuracy_diff': 0.15,
}

class MCIDataLoader:
    """MCI数据加载器，用于加载和处理MCI患者数据"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.mci_recovered_dir = os.path.join(data_dir, 'totalMCI_Recovered')  # 恢复为CN的MCI患者
        self.mci_not_recovered_dir = os.path.join(data_dir, 'totalMCI_NotRecovered')  # 未恢复的MCI患者
        self.metadata_file = os.path.join(data_dir, 'mci_recovery_metadata.xlsx')
        
        # 检查路径是否存在
        self._check_paths()
        
        print(f"🔧 初始化MCI数据加载器...")
        print(f"   MCI恢复患者目录: {self.mci_recovered_dir}")
        print(f"   MCI未恢复患者目录: {self.mci_not_recovered_dir}")
        print(f"   元数据文件: {self.metadata_file}")
    
    def _check_paths(self):
        """检查数据路径是否存在"""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        # 检查MCI恢复和未恢复目录
        for dir_path, dir_name in [(self.mci_recovered_dir, 'MCI恢复患者'), (self.mci_not_recovered_dir, 'MCI未恢复患者')]:
            if not os.path.exists(dir_path):
                # 尝试其他可能的目录名
                alt_dirs = [
                    dir_path.replace('total', ''),
                    dir_path.replace('total', 'MCI'),
                    dir_path.replace('total', 'mci'),
                    dir_path.lower(),
                    dir_path.replace('MCI', 'mci')
                ]
                for alt_path in alt_dirs:
                    if os.path.exists(alt_path):
                        dir_path = alt_path
                        break
                else:
                    raise FileNotFoundError(f"{dir_name}目录不存在: {dir_path}")
    
    def load_mci_images(self):
        """加载MCI患者图像数据"""
        print(f"🔄 加载MCI图像数据...")
        
        # 用于存储各模态图像的字典
        id_to_modalities = defaultdict(dict)
        patient_labels = {}
        
        # 加载恢复和未恢复的MCI患者图像
        for patient_group, label, folder_name in [
            ('Recovered', 1, self.mci_recovered_dir),
            ('NotRecovered', 0, self.mci_not_recovered_dir)
        ]:
            print(f"   扫描{patient_group}患者目录: {folder_name}")
            for modality, folder_suffix in [('CSF', 'CSF'), ('GRAY', 'GRAY'), ('WHITE', 'WHITE')]:
                modality_dir = os.path.join(folder_name, f'total{modality}')
                if not os.path.exists(modality_dir):
                    print(f"⚠️ 警告: {patient_group} {modality}模态目录不存在: {modality_dir}")
                    continue
                    
                print(f"   扫描{patient_group} {modality}模态: {modality_dir}")
                for file in os.listdir(modality_dir):
                    if file.endswith('.nii') or file.endswith('.nii.gz'):
                        file_path = os.path.join(modality_dir, file)
                        try:
                            # 提取患者ID
                            patient_id = self._extract_patient_id_from_filename(file)
                            if patient_id:
                                id_to_modalities[patient_id][modality] = file_path
                                patient_labels[patient_id] = label
                        except Exception as e:
                            print(f"⚠️ 警告: 处理文件 {file_path} 时出错: {str(e)}")
        
        # 处理收集到的图像数据
        images = []
        labels = []
        patient_ids = []
        
        # 检查每个患者是否有完整的三模态数据
        for patient_id, modalities in id_to_modalities.items():
            # 检查是否有三个模态
            required_keys = ['CSF', 'GRAY', 'WHITE']
            if all(key in modalities for key in required_keys):
                try:
                    # 加载三个模态
                    csf_img = self._load_and_normalize_image(modalities['CSF'])
                    gray_img = self._load_and_normalize_image(modalities['GRAY'])
                    white_img = self._load_and_normalize_image(modalities['WHITE'])
                    
                    # 确保三个模态形状一致
                    if csf_img.shape == gray_img.shape == white_img.shape:
                        # 合并三个模态为一个多通道图像 [3, D, H, W]
                        multi_modal_img = np.stack([csf_img, gray_img, white_img], axis=0)
                        
                        images.append(multi_modal_img)
                        labels.append(patient_labels[patient_id])
                        patient_ids.append(patient_id)
                    else:
                        print(f"⚠️ 警告: 患者 {patient_id} 的三个模态形状不一致，跳过")
                except Exception as e:
                    print(f"⚠️ 警告: 处理患者 {patient_id} 的图像时出错: {str(e)}")
            else:
                missing = [key for key in required_keys if key not in modalities]
                print(f"⚠️ 警告: 患者 {patient_id} 缺少模态: {missing}，跳过")
        
        # 转换为numpy数组
        images = np.array(images) if images else np.array([])
        labels = np.array(labels)
        
        print(f"✅ 加载完成: {np.sum(labels==1)} 恢复患者 + {np.sum(labels==0)} 未恢复患者 = {len(images)} 图像")
        if len(images) > 0:
            print(f"   图像形状: {images.shape}")
        
        return images, labels, patient_ids
    
    def _load_and_normalize_image(self, file_path):
        """加载并标准化单个图像"""
        import nibabel as nib
        from scipy.ndimage import zoom
        
        # 加载图像
        img = nib.load(file_path).get_fdata()
        
        # 确保图像是浮点类型
        img = img.astype(np.float32)
        
        # 标准化到[0, 1]范围
        if np.max(img) > np.min(img):
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
        
        return img
    
    def _extract_patient_id_from_filename(self, filename):
        """从文件名中提取患者ID"""
        # 优先匹配类似002_S_4447格式的ID
        match = re.search(r'(\d{3}_S_\d{4})', filename)
        if match:
            return match.group(1)
            
        # 尝试其他常见模式
        patterns = [
            r'(\d+)_.*\.nii',  # 123_date.nii
            r'.*_(\d+)_.*\.nii',  # prefix_123_date.nii
            r'.*_(\d+)\.nii',  # prefix_123.nii
            r'(\d+)\.nii'  # 123.nii
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1)
        
        # 如果没有匹配，使用文件名（不包括扩展名）作为ID
        return os.path.splitext(filename)[0]
    
    def load_mci_text_data(self):
        """加载MCI患者文本元数据"""
        print(f"   加载结构化MCI元数据...")
        all_texts, all_labels, all_patient_ids = [], [], []
        
        # 检查元数据文件是否存在
        if not os.path.exists(self.metadata_file):
            print(f"⚠️ 警告: MCI元数据文件未找到: {self.metadata_file}")
            return [], [], []
        
        try:
            # 加载元数据
            df = pd.read_excel(self.metadata_file)
            
            # 标准化列名
            df.columns = [col.lower() for col in df.columns]
            
            # 检查必要的列是否存在
            required_cols = ['subject', 'age', 'gender', 'education', 'mmse', 'cdrsb', 'recovery_status']
            if not all(col in df.columns for col in required_cols):
                print(f"❌ MCI元数据表缺少必要列. 需要: {required_cols}, 实际: {list(df.columns)}")
                return [], [], []
            
            for _, row in df.iterrows():
                patient_id = str(row['subject'])
                # 恢复状态: 1=恢复, 0=未恢复
                label = 1 if row['recovery_status'] == 'Recovered' else 0
                
                # 统一性别编码：'Male' -> 1, 'Female' -> 0
                gender_code = 1 if isinstance(row['gender'], str) and row['gender'].lower() == 'male' else 0
                        
                # 将结构化数据拼接成一个描述性字符串
                text = (f"age {row['age']}, gender {gender_code}, education {row['education']}, "
                        f"mmse_score {row['mmse']}, cdrsb_score {row['cdrsb']}")
                
                all_texts.append(text)
                all_labels.append(label)
                all_patient_ids.append(patient_id)
        except Exception as e:
            print(f"❌ 读取MCI元数据文件失败: {e}")
            return [], [], []
        
        print(f"   ✅ 成功为 {len(all_texts)} 名MCI受试者加载了元数据。")
        return all_texts, all_labels, all_patient_ids
    
    def align_image_text_data(self, images, texts, image_labels, text_labels, image_patient_ids, text_patient_ids):
        """对齐图像和文本数据 - 基于患者ID"""
        print("🔄 对齐图像和文本数据...")
        print(f"   图像数据: {len(images)} 个样本")
        print(f"   文本数据: {len(texts)} 个样本")
        
        # 创建ID到索引的映射
        image_id_to_idx = {pid: i for i, pid in enumerate(image_patient_ids)}
        text_id_to_idx = {pid: i for i, pid in enumerate(text_patient_ids)}
        
        # 找到共同的患者ID
        common_ids = set(image_patient_ids) & set(text_patient_ids)
        
        if not common_ids:
            print("⚠️ 警告: 图像和文本数据没有共同的患者ID")
            print("⚠️ 检查患者ID格式是否一致")
            return np.array([]), [], np.array([]), []
        else:
            print(f"✅ 找到 {len(common_ids)} 个共同患者ID")
            
            # 按共同ID重新排列数据
            aligned_images = []
            aligned_texts = []
            aligned_labels = []
            aligned_patient_ids = []
            
            for pid in common_ids:
                img_idx = image_id_to_idx[pid]
                txt_idx = text_id_to_idx[pid]
                
                aligned_images.append(images[img_idx])
                aligned_texts.append(texts[txt_idx])
                aligned_labels.append(image_labels[img_idx])  # 使用图像标签
                aligned_patient_ids.append(pid)
            
            aligned_images = np.array(aligned_images)
            aligned_labels = np.array(aligned_labels)
        
        print(f"✅ 数据对齐完成: {len(aligned_images)} 个样本")
        print(f"   恢复患者: {np.sum(aligned_labels==1)} 个样本")
        print(f"   未恢复患者: {np.sum(aligned_labels==0)} 个样本")
        
        return aligned_images, aligned_texts, aligned_labels, aligned_patient_ids

class FeatureExtractor:
    """特征提取器 - 基于预训练的对抗性对比学习模型"""
    
    def __init__(self, model_path, device='cuda', batch_size=16):
        self.device = device
        self.model_path = model_path
        self.batch_size = batch_size
        
        print(f"🔧 初始化特征提取器...")
        print(f"   模型路径: {model_path}")
        print(f"   设备: {device}")
        
        # 加载图像编码器
        self.image_encoder = self._load_pretrained_image_model()
        self.image_encoder.to(self.device)
        self.image_encoder.eval()

        # 动态检测图像特征维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 113, 137, 113).to(self.device)
            dummy_output = self.image_encoder(dummy_input, return_features=True)
            image_output_dim = dummy_output.shape[1]
            print(f"🔧 检测到图像编码器输出维度: {image_output_dim}")

        # 创建特征调整层，确保输出维度为512
        if image_output_dim != 512:
            print(f"   ⚠️ 输出维度不是512，添加图像特征调整层 {image_output_dim} -> 512")
            self.image_feature_adjust = nn.Sequential(
                nn.Linear(image_output_dim, 512),
                nn.ReLU(),
            ).to(self.device)
        else:
            self.image_feature_adjust = nn.Identity()

        # 初始化文本编码器
        self.text_encoder = AdversarialTextEncoder(
            feature_dim=512, 
            device=self.device
        )
        self.text_encoder.to(self.device)
        self.text_encoder.eval()

        print(f"特征提取器初始化完成")
    
    def _load_pretrained_image_model(self):
        """加载预训练的图像模型"""
        print(f"🔄 智能加载预训练图像模型...")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在 {self.model_path}")
        
        try:
            from optimized_models import ImprovedResNetCBAM3D
        except ImportError:
            raise ImportError("❌ 无法导入ImprovedResNetCBAM3D模型类，请确保optimized_models.py在路径中")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # 检测基础通道数
        base_channels = 12
        if 'stem.0.weight' in state_dict and state_dict['stem.0.weight'].shape[0] == 4:
            base_channels = 8
        elif 'image_encoder.backbone.stem.0.weight' in state_dict and state_dict['image_encoder.backbone.stem.0.weight'].shape[0] == 4:
            base_channels = 8
            
        # 创建模型
        model = ImprovedResNetCBAM3D(
            in_channels=3, num_classes=2, base_channels=base_channels, 
            dropout_rate=0.3, use_global_pool=False
        )
        
        keys = list(state_dict.keys())
        
        # 处理不同类型的检查点
        if any(k.startswith('image_encoder.backbone.') for k in keys):
            print("   🔧 检测到完整的对抗性模型检查点")
            temp_state_dict = {}
            prefix = 'image_encoder.backbone.'
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    new_key = key[len(prefix):]
                    temp_state_dict[new_key] = value
            backbone_state_dict = {
                k: v for k, v in temp_state_dict.items() 
                if not k.startswith('fusion') and not k.startswith('classifier')
            }
            model.load_state_dict(backbone_state_dict, strict=False)
        else:
            print("   🔧 检测到独立的图像编码器检查点")
            backbone_state_dict = {
                k: v for k, v in state_dict.items() 
                if not k.startswith('fusion') and not k.startswith('classifier')
            }
            model.load_state_dict(backbone_state_dict, strict=False)

        return model
    
    def extract_image_features(self, images):
        """提取图像特征"""
        print(f"🖼 提取图像特征: {images.shape}")
        
        if len(images) == 0:
            return np.empty((0, 512))
            
        features_list = []
        with torch.no_grad():
            for i in tqdm(range(0, len(images), self.batch_size), desc="提取图像特征"):
                batch_images = images[i:i+self.batch_size]
                batch_tensor = torch.FloatTensor(batch_images).to(self.device)
                image_features = self.image_encoder(batch_tensor, return_features=True)
                
                # 应用特征维度适配层
                adjusted_features = self.image_feature_adjust(image_features)
                
                features_list.append(adjusted_features.cpu().numpy())
            
        features = np.concatenate(features_list, axis=0)
        return features
            
    def extract_text_features(self, texts):
        """提取文本特征"""
        print(f"📝 提取文本特征: {len(texts)} 个文本")
        
        if not texts:
            return np.empty((0, 512))
        
        features_list = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size), desc="提取文本特征"):
                batch_texts = texts[i:i+self.batch_size]
                text_features = self.text_encoder(batch_texts)
                features_list.append(text_features.cpu().numpy())

        features = np.concatenate(features_list, axis=0)
        return features
    
    def extract_multimodal_features(self, images, texts, fusion_strategy='image_only', feature_noise_std=0.0):
        """提取多模态特征"""
        print(f"🔄 提取多模态特征 (策略: {fusion_strategy})...")
            
        # 提取图像特征
        image_features = self.extract_image_features(images)
        
        # 提取文本特征
        text_features = self.extract_text_features(texts)
        
        # 添加特征噪声（如果指定）
        if feature_noise_std > 0:
            np.random.seed(RANDOM_SEED)
            image_noise = np.random.normal(0, feature_noise_std, image_features.shape)
            image_features = image_features + image_noise
            print(f"   添加特征噪声: std={feature_noise_std}")

        print(f"🔧 特征提取完成:")
        print(f"   图像特征形状: {image_features.shape}")
        
        # 根据融合策略返回特征
        if fusion_strategy == 'image_only':
            return image_features
        elif fusion_strategy == 'text_only':
            return text_features
        elif fusion_strategy == 'concatenate':
            return np.concatenate((image_features, text_features), axis=1)
        elif fusion_strategy == 'weighted_average':
            # 默认权重：图像0.7，文本0.3
            return 0.7 * image_features + 0.3 * text_features
        else:
            raise ValueError(f"未知的融合策略: {fusion_strategy}")

class MCIRecoveryClassifier:
    """MCI恢复分类器，用于预测MCI患者是否会恢复为认知正常"""
    
    def __init__(self, config=None):
        self.config = config or RECOMMENDED_CONFIG
        self.classifier_type = self.config.get('classifier_type', 'xgb')
        self.feature_standardization = self.config.get('feature_standardization', True)
        self.scaler = StandardScaler() if self.feature_standardization else None
        
        print(f"🔧 初始化MCI恢复分类器...")
        print(f"   分类器类型: {self.classifier_type}")
        print(f"   特征标准化: {self.feature_standardization}")
    
    def _create_classifier(self):
        """创建分类器实例"""
        if self.classifier_type == 'logistic':
            return LogisticRegression(
                C=self.config.get('regularization_strength', 1.0),
                max_iter=self.config.get('max_iter', 6000),
                random_state=RANDOM_SEED
            )
        elif self.classifier_type == 'svm':
            return SVC(
                C=self.config.get('regularization_strength', 1.0),
                kernel='rbf',
                probability=True,
                random_state=RANDOM_SEED
            )
        elif self.classifier_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=RANDOM_SEED
            )
        elif self.classifier_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=RANDOM_SEED
            )
        elif self.classifier_type == 'xgb':
            return xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                reg_lambda=self.config.get('regularization_strength', 1.0),
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=RANDOM_SEED
            )
        else:
            raise ValueError(f"未知的分类器类型: {self.classifier_type}")
    
    def train_and_evaluate(self, features, labels, patient_ids=None):
        """训练并评估MCI恢复分类器"""
        print(f"🔄 开始训练和评估MCI恢复分类器...")
        start_time = time.time()
        
        # 标准化特征
        if self.feature_standardization:
            features = self.scaler.fit_transform(features)
        
        # 交叉验证类型
        cross_validation = self.config.get('cross_validation', 'kfold')
        
        if cross_validation == 'kfold':
            # K折交叉验证
            kfold_splits = self.config.get('kfold_splits', 5)
            kf = KFold(n_splits=kfold_splits, shuffle=True, random_state=RANDOM_SEED)
            
            all_preds = np.zeros_like(labels, dtype=float)
            all_probs = np.zeros_like(labels, dtype=float)
            fold_results = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(features, labels)):
                print(f"\n   📊 第 {fold_idx + 1}/{kfold_splits} 折交叉验证")
                
                # 划分训练集和验证集
                X_train, X_val = features[train_idx], features[val_idx]
                y_train, y_val = labels[train_idx], labels[val_idx]
                
                # 创建并训练分类器
                classifier = self._create_classifier()
                classifier.fit(X_train, y_train)
                
                # 预测
                y_pred = classifier.predict(X_val)
                y_prob = classifier.predict_proba(X_val)[:, 1] if hasattr(classifier, 'predict_proba') else classifier.decision_function(X_val)
                
                # 保存结果
                all_preds[val_idx] = y_pred
                all_probs[val_idx] = y_prob
                
                # 计算性能指标
                fold_result = self._calculate_metrics(y_val, y_pred, y_prob)
                fold_results.append(fold_result)
                
                print(f"   准确率: {fold_result['accuracy']:.4f}")
                print(f"   精确率: {fold_result['precision']:.4f}")
                print(f"   召回率: {fold_result['recall']:.4f}")
                print(f"   F1分数: {fold_result['f1']:.4f}")
                print(f"   AUC: {fold_result['auc']:.4f}")
            
            # 计算平均结果
            avg_results = self._calculate_average_results(fold_results)
            print(f"\n📋 平均交叉验证结果:")
            for metric, value in avg_results.items():
                print(f"   {metric}: {value:.4f}")
            
            # 计算总体结果
            overall_results = self._calculate_metrics(labels, all_preds, all_probs)
            print(f"\n📊 总体结果:")
            for metric, value in overall_results.items():
                print(f"   {metric}: {value:.4f}")
            
            # 生成分类报告
            print(f"\n📋 分类报告:")
            print(classification_report(labels, all_preds, target_names=['未恢复', '恢复']))
            
            # 生成混淆矩阵
            self._plot_confusion_matrix(labels, all_preds)
            
            # 生成ROC曲线
            self._plot_roc_curve(labels, all_probs)
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ 训练和评估完成，耗时: {elapsed_time:.2f} 秒")
        
        return {
            'fold_results': fold_results,
            'avg_results': avg_results,
            'overall_results': overall_results,
            'predictions': all_preds,
            'probabilities': all_probs
        }
    
    def _calculate_metrics(self, y_true, y_pred, y_prob):
        """计算分类性能指标"""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        
        # 计算AUC
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def _calculate_average_results(self, fold_results):
        """计算平均交叉验证结果"""
        avg_results = {}
        metrics = fold_results[0].keys()
        
        for metric in metrics:
            avg_results[metric] = np.mean([fold[metric] for fold in fold_results])
            avg_results[f'{metric}_std'] = np.std([fold[metric] for fold in fold_results])
        
        return avg_results
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                  xticklabels=['未恢复', '恢复'], 
                  yticklabels=['未恢复', '恢复'])
        plt.title('MCI恢复预测混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        
        # 保存混淆矩阵
        os.makedirs('./results', exist_ok=True)
        plt.savefig('./results/confusion_matrix.png')
        print(f"✅ 混淆矩阵已保存到 ./results/confusion_matrix.png")
        plt.close()
    
    def _plot_roc_curve(self, y_true, y_prob):
        """绘制ROC曲线"""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('MCI恢复预测ROC曲线')
        plt.legend(loc="lower right")
        
        # 保存ROC曲线
        os.makedirs('./results', exist_ok=True)
        plt.savefig('./results/roc_curve.png')
        print(f"✅ ROC曲线已保存到 ./results/roc_curve.png")
        plt.close()
    
    def predict(self, features):
        """使用训练好的分类器进行预测"""
        # 标准化特征
        if self.feature_standardization:
            features = self.scaler.transform(features)
        
        # 预测
        y_pred = self.classifier.predict(features)
        y_prob = self.classifier.predict_proba(features)[:, 1] if hasattr(self.classifier, 'predict_proba') else self.classifier.decision_function(features)
        
        return y_pred, y_prob
    
    def save_model(self, save_path):
        """保存训练好的模型"""
        import pickle
        
        model_data = {
            'classifier': self.classifier,
            'config': self.config,
            'scaler': self.scaler
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ 模型已保存到: {save_path}")
    
    def load_model(self, load_path):
        """加载训练好的模型"""
        import pickle
        
        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.config = model_data.get('config', self.config)
        self.scaler = model_data.get('scaler', None)
        
        print(f"✅ 模型已从: {load_path} 加载")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCI恢复预测系统")
    parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp/DATA_MCI/', help='MCI数据目录')
    parser.add_argument('--model_path', type=str, default='./models/adversarial/best_mcic_adversarial_cv_model.pth', help='预训练模型路径')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--save_results', type=bool, default=True, help='是否保存结果')
    parser.add_argument('--fusion_strategy', type=str, default='image_only', help='特征融合策略')
    parser.add_argument('--classifier_type', type=str, default='xgb', help='分类器类型')
    parser.add_argument('--cross_validation', type=str, default='kfold', help='交叉验证类型')
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed()
    
    # 加载配置
    config = RECOMMENDED_CONFIG
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # 更新配置
    config['fusion_strategy'] = args.fusion_strategy
    config['classifier_type'] = args.classifier_type
    config['cross_validation'] = args.cross_validation
    
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    data_loader = MCIDataLoader(args.data_dir)
    images, labels, patient_ids = data_loader.load_mci_images()
    
    if len(images) == 0:
        print("❌ 没有加载到任何图像数据，退出程序")
        return
    
    # 加载预训练模型
    feature_extractor = FeatureExtractor(args.model_path, device=device)
    
    # 提取特征
    features = feature_extractor.extract_multimodal_features(images, [], fusion_strategy=args.fusion_strategy)
    
    # 创建并训练分类器
    classifier = MCIRecoveryClassifier(config)
    results = classifier.train_and_evaluate(features, labels, patient_ids)
    
    # 保存结果
    if args.save_results:
        os.makedirs('./results', exist_ok=True)
        results_file = f'./results/mci_recovery_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4, default=lambda x: float(x) if isinstance(x, np.float32) else x.tolist() if isinstance(x, np.ndarray) else str(x))
        print(f"✅ 结果已保存到: {results_file}")
    
    print("\n🎉 MCI恢复预测完成！")

if __name__ == '__main__':
    main()

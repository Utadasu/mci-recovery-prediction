"""
🚀 阿尔茨海默病多模态对比学习训练脚本
========================================

训练流程:
1. 🔄 加载预训练的图像和文本编码器
2. 📊 创建图像-文本配对数据集
3. 🎯 对比学习预训练（图像-文本特征对齐）
4. 📈 微调分类（端到端训练）
5. 📋 评估多模态融合性能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from contrastive_learning import (
    MultiModalContrastiveModel, 
    PreExtractedFeaturesLoader,
    create_contrastive_model
)
from data_utils import load_early_fusion_data
from optimized_models import ImprovedResNetCBAM3D


class MultiModalDataset(Dataset):
    """
    多模态数据集 - 图像和文本配对
    """
    def __init__(self, 
                 image_data: np.ndarray,
                 text_features: np.ndarray,
                 labels: np.ndarray,
                 text_encoder,
                 mode: str = 'contrastive'):
        """
        Args:
            image_data: 图像数据 [N, 3, D, H, W]
            text_features: 文本特征 [N, 512] 或文本列表
            labels: 标签 [N]
            text_encoder: 文本编码器（用于编码原始文本）
            mode: 'contrastive' 或 'classification'
        """
        self.image_data = image_data
        self.text_features = text_features
        self.labels = labels
        self.text_encoder = text_encoder
        self.mode = mode
        
        print(f"📊 多模态数据集初始化: {len(self.image_data)} 样本")
        print(f"   图像形状: {self.image_data.shape}")
        print(f"   文本特征形状: {self.text_features.shape}")
        print(f"   标签分布: AD={np.sum(labels==1)}, CN={np.sum(labels==0)}")
    
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, idx):
        # 图像数据
        image = torch.tensor(self.image_data[idx], dtype=torch.float32)
        
        # 文本特征（预提取的512维特征）
        text_feature = torch.tensor(self.text_features[idx], dtype=torch.float32)
        
        # 标签
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # 为对比学习生成虚拟的input_ids和attention_mask
        # 实际使用中，可以从预训练文本编码器获取
        fake_input_ids = torch.zeros(128, dtype=torch.long)
        fake_attention_mask = torch.ones(128, dtype=torch.long)
        
        return {
            'image': image,
            'text_feature': text_feature,
            'input_ids': fake_input_ids,
            'attention_mask': fake_attention_mask,
            'label': label
        }


class ContrastiveTrainer:
    """
    对比学习训练器
    """
    def __init__(self, 
                 model: MultiModalContrastiveModel,
                 device: str = 'cuda',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5):
        self.model = model
        self.device = device
        
        # 为不同部分设置不同的学习率
        self.optimizer = self._setup_optimizer(learning_rate, weight_decay)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        
        # 损失权重
        self.contrastive_weight = 1.0
        self.classification_weight = 1.0
        
        print(f"🎯 训练器初始化完成，设备: {device}")
    
    def _setup_optimizer(self, lr: float, weight_decay: float):
        """设置优化器 - 为不同模块使用不同学习率"""
        params = []
        
        # 图像编码器投影层 - 较高学习率
        params.append({
            'params': self.model.image_encoder.image_projector.parameters(),
            'lr': lr * 2,
            'weight_decay': weight_decay
        })
        
        # 文本编码器投影层 - 较高学习率
        params.append({
            'params': self.model.text_encoder.text_projector.parameters(),
            'lr': lr * 2,
            'weight_decay': weight_decay
        })
        
        # 融合分类器 - 标准学习率
        params.append({
            'params': self.model.fusion_classifier.parameters(),
            'lr': lr,
            'weight_decay': weight_decay
        })
        
        # 如果骨干网络未冻结，使用更小的学习率
        if any(p.requires_grad for p in self.model.image_encoder.backbone.parameters()):
            params.append({
                'params': self.model.image_encoder.backbone.parameters(),
                'lr': lr * 0.1,
                'weight_decay': weight_decay
            })
        
        if any(p.requires_grad for p in self.model.text_encoder.bert.parameters()):
            params.append({
                'params': self.model.text_encoder.bert.parameters(),
                'lr': lr * 0.1,
                'weight_decay': weight_decay
            })
        
        return optim.AdamW(params)
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        contrastive_loss_sum = 0
        classification_loss_sum = 0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # 数据移动到设备
            images = batch['image'].to(self.device)
            text_features = batch['text_feature'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播 - 对比学习模式
            contrastive_output = self.model(
                images, input_ids, attention_mask, mode='contrastive'
            )
            contrastive_loss = contrastive_output['contrastive_loss']
            
            # 前向传播 - 分类模式
            classification_output = self.model(
                images, input_ids, attention_mask, mode='classification'
            )
            logits = classification_output['logits']
            
            # 分类损失
            classification_loss = nn.CrossEntropyLoss()(logits, labels)
            
            # 总损失
            total_batch_loss = (
                self.contrastive_weight * contrastive_loss + 
                self.classification_weight * classification_loss
            )
            
            # 反向传播
            total_batch_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 统计
            total_loss += total_batch_loss.item()
            contrastive_loss_sum += contrastive_loss.item()
            classification_loss_sum += classification_loss.item()
            
            # 准确率计算
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{total_batch_loss.item():.4f}',
                'Acc': f'{correct_predictions/total_samples:.4f}',
                'CL': f'{contrastive_loss.item():.4f}',
                'CE': f'{classification_loss.item():.4f}'
            })
        
        # 学习率调度
        self.scheduler.step()
        
        return {
            'total_loss': total_loss / len(dataloader),
            'contrastive_loss': contrastive_loss_sum / len(dataloader),
            'classification_loss': classification_loss_sum / len(dataloader),
            'accuracy': correct_predictions / total_samples,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """评估模型"""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='评估中'):
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 分类模式
                output = self.model(
                    images, input_ids, attention_mask, mode='classification'
                )
                logits = output['logits']
                
                # 损失
                loss = nn.CrossEntropyLoss()(logits, labels)
                total_loss += loss.item()
                
                # 预测
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_predictions)
        report = classification_report(
            all_labels, all_predictions, 
            target_names=['CN', 'AD'], 
            output_dict=True
        )
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        return {
            'accuracy': accuracy,
            'loss': total_loss / len(dataloader),
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'predictions': all_predictions,
            'labels': all_labels
        }


def load_multimodal_data(image_data_dir: str, 
                        text_features_path: str,
                        test_size: float = 0.2,
                        random_state: int = 42) -> Tuple:
    """
    加载多模态数据
    
    Args:
        image_data_dir: 图像数据目录
        text_features_path: 文本特征文件路径
        test_size: 测试集比例
        random_state: 随机种子
    
    Returns:
        (train_images, train_text, train_labels, val_images, val_text, val_labels)
    """
    print("🔄 加载多模态数据...")
    
    # 加载图像数据
    print("📸 加载图像数据...")
    image_data, labels = load_early_fusion_data(image_data_dir)
    print(f"✅ 图像数据加载完成: {image_data.shape}")
    
    # 加载文本特征
    print("📝 加载文本特征...")
    text_loader = PreExtractedFeaturesLoader(text_features_path)
    text_features = text_loader.features  # [N, 512]
    
    # 确保数据对应
    assert len(image_data) == len(text_features), \
        f"图像和文本数据数量不匹配: {len(image_data)} vs {len(text_features)}"
    
    # 数据分割
    train_indices, val_indices = train_test_split(
        range(len(image_data)), 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels
    )
    
    train_images = image_data[train_indices]
    train_text = text_features[train_indices]
    train_labels = labels[train_indices]
    
    val_images = image_data[val_indices]
    val_text = text_features[val_indices]
    val_labels = labels[val_indices]
    
    print(f"📊 数据分割完成:")
    print(f"   训练集: {len(train_images)} 样本")
    print(f"   验证集: {len(val_images)} 样本")
    
    return train_images, train_text, train_labels, val_images, val_text, val_labels


def main():
    """主训练函数"""
    print("🚀 开始多模态对比学习训练...")
    
    # 配置参数
    config = {
        'image_model_path': './models/smart_downsample_spatial_ch12.pth',  # 🔥 智能下采样模型
        'text_model_path': '/tmp/pycharm_project_194/备份5.27/文本编码器/alzheimer_bert_complete_model.pth',
        'text_features_path': '/tmp/pycharm_project_194/备份5.27/文本编码器/alzheimer_features_512d.pkl',
        'image_data_dir': '/root/autodl-tmp/DATA_MCI/test_data/',  # 修正为实际服务器路径
        'batch_size': 16,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': './models/contrastive',
        'freeze_backbones': True,
        'contrastive_weight': 1.0,
        'classification_weight': 2.0  # 给分类任务更高权重
    }
    
    # 验证并修正数据路径
    if not os.path.exists(config['image_data_dir']):
        print(f"⚠️  默认数据路径不存在: {config['image_data_dir']}")
        
        # 尝试自动检测数据路径
        possible_paths = [
            "/root/autodl-tmp/DATA_MCI/test_data/",
            "/data/alzheimer/mri/", 
            "/tmp/data/alzheimer/",
            "./data/mri/"
        ]
        
        found_path = None
        for path in possible_paths:
            if os.path.exists(path):
                # 检查是否包含预期的子目录结构
                expected_subdirs = ["123-AD-MRI", "123-CN-MRI"]
                if all(os.path.exists(os.path.join(path, subdir)) for subdir in expected_subdirs):
                    found_path = path
                    break
        
        if found_path:
            config['image_data_dir'] = found_path
            print(f"✅ 自动检测到数据路径: {found_path}")
        else:
            print("❌ 无法自动检测数据路径，请手动指定!")
            print("💡 请确保数据目录包含以下结构:")
            print("   123-AD-MRI/ 和 123-CN-MRI/")
            return
    
    print(f"📋 训练配置:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # 创建模型
    print("\n🎯 创建多模态对比学习模型...")
    model = create_contrastive_model(
        image_model_path=config['image_model_path'],
        text_model_path=config['text_model_path'],
        device=config['device'],
        freeze_backbones=config['freeze_backbones']
    )
    
    # 加载数据
    print("\n📊 加载训练数据...")
    try:
        train_images, train_text, train_labels, val_images, val_text, val_labels = load_multimodal_data(
            image_data_dir=config['image_data_dir'],
            text_features_path=config['text_features_path']
        )
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        print("💡 请确保以下文件存在:")
        print(f"   - 图像数据目录: {config['image_data_dir']}")
        print(f"   - 文本特征文件: {config['text_features_path']}")
        return
    
    # 创建数据集
    train_dataset = MultiModalDataset(
        train_images, train_text, train_labels, 
        model.text_encoder, mode='contrastive'
    )
    val_dataset = MultiModalDataset(
        val_images, val_text, val_labels,
        model.text_encoder, mode='contrastive'
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建训练器
    trainer = ContrastiveTrainer(
        model=model,
        device=config['device'],
        learning_rate=config['learning_rate']
    )
    
    # 设置损失权重
    trainer.contrastive_weight = config['contrastive_weight']
    trainer.classification_weight = config['classification_weight']
    
    # 训练循环
    print(f"\n🎯 开始训练，共 {config['num_epochs']} 轮...")
    best_val_accuracy = 0.0
    training_history = []
    
    for epoch in range(config['num_epochs']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"{'='*50}")
        
        # 训练
        train_metrics = trainer.train_epoch(train_loader, epoch)
        
        # 验证
        val_metrics = trainer.evaluate(val_loader)
        
        # 记录历史
        epoch_history = {
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': val_metrics
        }
        training_history.append(epoch_history)
        
        # 打印结果
        print(f"\n📊 Epoch {epoch+1} 结果:")
        print(f"   训练 - 损失: {train_metrics['total_loss']:.4f}, 准确率: {train_metrics['accuracy']:.4f}")
        print(f"   验证 - 损失: {val_metrics['loss']:.4f}, 准确率: {val_metrics['accuracy']:.4f}")
        print(f"   学习率: {train_metrics['learning_rate']:.6f}")
        
        # 保存最佳模型
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            
            # 保存模型
            save_path = os.path.join(config['save_dir'], 'best_contrastive_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_accuracy': best_val_accuracy,
                'config': config
            }, save_path)
            
            print(f"🏆 新的最佳模型已保存: {save_path} (准确率: {best_val_accuracy:.4f})")
    
    # 保存训练历史
    history_path = os.path.join(config['save_dir'], 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        # 转换numpy数组为列表以便JSON序列化
        history_to_save = []
        for epoch_data in training_history:
            epoch_copy = epoch_data.copy()
            # 移除无法JSON序列化的项目
            if 'confusion_matrix' in epoch_copy['val']:
                epoch_copy['val']['confusion_matrix'] = epoch_copy['val']['confusion_matrix'].tolist()
            history_to_save.append(epoch_copy)
        
        json.dump(history_to_save, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 训练完成！")
    print(f"📈 最佳验证准确率: {best_val_accuracy:.4f}")
    print(f"💾 模型保存路径: {config['save_dir']}")
    print(f"📊 训练历史保存: {history_path}")


if __name__ == "__main__":
    main() 
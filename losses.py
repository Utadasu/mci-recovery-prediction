import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LabelSmoothingLoss(nn.Module):
    """
    带标签平滑的交叉熵损失函数
    这能减少模型过度自信，提高泛化能力
    """
    def __init__(self, classes=2, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # 创建平滑标签
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class FocalLoss(nn.Module):
    """
    Focal Loss能帮助解决类别不平衡问题
    降低易分样本的权重，增加难分样本的权重
    """
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ImprovedFocalLoss(nn.Module):
    """
    改进的Focal Loss - 专门为阿尔茨海默病分类优化
    
    功能特性:
    - 🎯 自适应alpha权重，根据类别分布动态调整
    - 🔧 可配置gamma参数，控制难易样本权重
    - 📊 支持类别权重，处理数据不平衡
    - 🛡️ 数值稳定性优化，避免梯度爆炸
    """
    def __init__(self, alpha=1.0, gamma=2.0, class_weights=None, reduction='mean', eps=1e-8):
        super(ImprovedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        
        # 类别权重 - 默认给AD类别稍高权重
        if class_weights is None:
            self.class_weights = torch.tensor([1.0, 1.2])  # [CN, AD]
        else:
            self.class_weights = torch.tensor(class_weights)
    
    def forward(self, inputs, targets):
        """
        前向传播
        Args:
            inputs: 模型输出logits [B, num_classes]
            targets: 真实标签 [B]
        Returns:
            loss: Focal Loss值
        """
        # 确保class_weights在正确设备上
        if self.class_weights.device != inputs.device:
            self.class_weights = self.class_weights.to(inputs.device)
        
        # 计算交叉熵损失 (不进行reduction)
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        
        # 计算概率
        pt = torch.exp(-ce_loss + self.eps)  # 添加eps避免数值不稳定
        
        # 计算alpha权重
        if isinstance(self.alpha, (float, int)):
            alpha_t = self.alpha
        else:
            # 如果alpha是tensor，根据targets选择对应权重
            alpha_t = self.alpha[targets]
        
        # 计算Focal Loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        # 应用reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def to_one_hot(labels, num_classes=2):
    """Convert class labels to one-hot encoding"""
    batch_size = labels.size(0)
    one_hot = torch.zeros(batch_size, num_classes, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot

def combined_criterion(outputs, targets, smoothing=0.1, focal_weight=0.5):
    """
    组合多个损失函数以获得更好的性能
    - 交叉熵损失：基础分类损失
    - 标签平滑：减轻过拟合
    - Focal Loss：处理类别不平衡
    """
    # 标准交叉熵
    ce_loss = nn.CrossEntropyLoss()(outputs, targets)
    
    # 标签平滑
    ls_loss = LabelSmoothingLoss(smoothing=smoothing)(outputs, targets)
    
    # Focal Loss
    focal_loss = FocalLoss()(outputs, targets)
    
    # 组合损失
    combined_loss = ce_loss * 0.4 + ls_loss * 0.3 + focal_loss * focal_weight * 0.3
    
    return combined_loss

def weighted_criterion(outputs, targets, class_weights=None):
    """
    带类别权重的交叉熵损失
    适用于类别不平衡的数据集
    """
    if class_weights is None:
        # 默认给AD类别更高的权重(1.5)，CN类别权重为1.0
        class_weights = torch.tensor([1.5, 1.0]).to(outputs.device)
    
    return nn.CrossEntropyLoss(weight=class_weights)(outputs, targets) 
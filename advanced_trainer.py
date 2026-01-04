import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import os
import numpy as np
from tqdm import tqdm
import io
import torch.nn.functional as F
from advanced_models import AdaptiveFusionModel, MixUp, ProgressiveDropout
from optimized_models import create_improved_resnet3d, EMAModel

class AdvancedTrainer:
    def __init__(self, models, device, num_classes=2, fusion_type='adaptive'):
        """
        高级训练器，支持多种正则化技术和融合方法
        
        参数:
        - models: 字典，键为模型名称，值为模型实例
        - device: 训练设备 (cuda/cpu)
        - num_classes: 分类数量
        - fusion_type: 融合类型 ('adaptive'/'weighted'/'voting')
        """
        self.models = models
        self.device = device
        self.num_classes = num_classes
        self.fusion_type = fusion_type
        
        # 创建融合模型（如果采用自适应融合）
        if fusion_type == 'adaptive':
            self.fusion_model = AdaptiveFusionModel(num_models=len(models), num_classes=num_classes).to(device)
        
        # MixUp正则化
        self.mixup = MixUp(alpha=0.2)
        
        # 存储历史最佳模型
        self.best_models = {}
        self.best_val_acc = 0.0
        
    def train_individual_models(self, train_loaders, val_loaders, 
                               num_epochs=50, learning_rate=0.001, 
                               weight_decay=0.01, patience=10, 
                               use_mixup=True, mixup_prob=0.5):
        """训练单个模型"""
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n===== 训练 {model_name} 模型 =====")
            
            # 为每个模型设置优化器和调度器
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6
            )
            criterion = nn.CrossEntropyLoss()
            
            # 初始化训练变量
            best_val_acc = 0.0
            no_improve_epochs = 0
            best_model_state = None
            
            # 初始化混合精度训练
            scaler = amp.GradScaler()
            
            # 确定当前组织类型
            tissue_type = list(train_loaders.keys())[0]  # 获取第一个键
            train_loader = train_loaders[tissue_type]
            val_loader = val_loaders[tissue_type]
            
            for epoch in range(num_epochs):
                # 训练阶段
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train {model_name}]')
                for data in train_pbar:
                    # 处理返回的数据 - 可能有3个或4个值
                    if len(data) == 4:
                        inputs, labels, patient_ids, modality = data
                    else:
                        inputs, labels, patient_ids = data
                    
                    inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                    
                    optimizer.zero_grad()
                    
                    # 应用MixUp正则化（如果启用）
                    apply_mixup = use_mixup and np.random.random() < mixup_prob
                    
                    # 使用混合精度
                    with amp.autocast():
                        if apply_mixup:
                            mixed_inputs, mixed_targets = self.mixup(inputs, labels, training=True)
                            outputs = model(mixed_inputs)
                            # 使用混合标签计算损失
                            loss = torch.sum(-mixed_targets * F.log_softmax(outputs, dim=1), dim=1).mean()
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                    
                    # 使用scaler进行反向传播
                    scaler.scale(loss).backward()
                    # 梯度裁剪防止爆炸
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    train_loss += loss.item()
                    
                    # 如果使用了mixup，则使用原始标签计算准确率
                    if not apply_mixup:
                        _, predicted = outputs.max(1)
                        train_total += labels.size(0)
                        train_correct += predicted.eq(labels).sum().item()
                    else:
                        # 对于mixup数据，我们不计算训练准确率（或使用原始标签）
                        _, predicted = outputs.max(1)
                        train_total += labels.size(0)
                        train_correct += predicted.eq(labels).sum().item()
                    
                    train_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100.*train_correct/train_total:.2f}%' if train_total > 0 else 'N/A'
                    })
                
                # 验证阶段
                val_loss, val_acc = self.validate_model(model, val_loader, criterion)
                
                # 更新学习率
                scheduler.step(val_acc)
                current_lr = optimizer.param_groups[0]['lr']
                
                print(f'\nEpoch [{epoch+1}/{num_epochs}] - {model_name}:')
                print(f'Train Loss: {train_loss / len(train_loader):.4f}, '
                      f'Train Acc: {100.*train_correct/train_total:.2f}%' if train_total > 0 else 'N/A')
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                print(f'Learning Rate: {current_lr:.6f}')
                
                # 保存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                    }
                    
                    # 安全保存模型
                    try:
                        # 首先尝试保存到临时缓冲区
                        buffer = io.BytesIO()
                        torch.save(best_model_state, buffer)
                        buffer.seek(0)
                        
                        # 创建模型目录（如果不存在）
                        os.makedirs('./models', exist_ok=True)
                        
                        # 如果成功，写入文件
                        model_path = f'./models/best_{model_name}.pth'
                        with open(model_path, 'wb') as f:
                            f.write(buffer.read())
                            
                        print(f'保存最佳{model_name}模型，验证准确率: {val_acc:.2f}%')
                    except Exception as e:
                        print(f"保存模型时出错: {str(e)}")
                        print("继续训练而不保存模型...")
                    
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    print(f'验证准确率连续{no_improve_epochs}个epoch没有提高')
                
                # 早停检查
                if no_improve_epochs >= patience:
                    print(f'早停在epoch {epoch+1}')
                    break
            
            # 训练后，加载最佳模型
            if best_model_state is not None:
                try:
                    model_path = f'./models/best_{model_name}.pth'
                    if os.path.exists(model_path):
                        # 尝试加载保存的模型文件
                        checkpoint = torch.load(model_path)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        print(f'加载最佳{model_name}模型，验证准确率: {best_val_acc:.2f}%')
                    else:
                        # 直接从内存加载
                        model.load_state_dict(best_model_state['model_state_dict'])
                        print(f'从内存加载最佳{model_name}模型，验证准确率: {best_val_acc:.2f}%')
                except Exception as e:
                    print(f"加载模型时出错: {str(e)}")
                    print("使用当前模型参数...")
            
            # 更新结果
            results[model_name] = {
                'model': model,
                'best_val_acc': best_val_acc
            }
            
            # 更新最佳模型字典
            self.best_models[model_name] = model.state_dict()
        
        return results
    
    def validate_model(self, model, val_loader, criterion):
        """验证模型性能"""
        model.eval()
        val_loss = 0
        
        # 用于存储每个患者的预测结果
        patient_predictions = {}
        patient_labels = {}
        
        with torch.no_grad():
            for data in val_loader:
                # 处理返回的数据 - 可能有3个或4个值
                if len(data) == 4:
                    inputs, labels, patient_ids, modalities = data
                else:
                    inputs, labels, patient_ids = data
                    modalities = ['unknown'] * len(patient_ids)
                
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # 收集每个患者的预测结果
                for i, patient_id in enumerate(patient_ids):
                    modality = modalities[i] if isinstance(modalities, list) else modalities
                    
                    if patient_id not in patient_predictions:
                        patient_predictions[patient_id] = []
                        patient_labels[patient_id] = labels[i].item()
                    
                    # 保存预测结果和模态信息
                    pred = outputs[i].argmax().item()
                    patient_predictions[patient_id].append((pred, modality))
        
        # 计算每个患者的最终预测（多数投票）
        correct_patients = 0
        for patient_id in patient_predictions:
            # 提取预测结果
            preds = [p[0] for p in patient_predictions[patient_id]]
            
            # 对每个患者的所有模态预测进行多数投票
            if preds:
                final_prediction = max(set(preds), key=preds.count)
                if final_prediction == patient_labels[patient_id]:
                    correct_patients += 1
        
        # 计算患者级别的准确率
        val_acc = 100.0 * correct_patients / len(patient_predictions) if patient_predictions else 0
        val_loss = val_loss / len(val_loader) if val_loader else 0
        
        return val_loss, val_acc
    
    def train_fusion_model(self, test_loaders, num_epochs=20, learning_rate=0.0005):
        """训练自适应融合模型"""
        if self.fusion_type != 'adaptive':
            print("只有在fusion_type='adaptive'模式下才能训练融合模型")
            return None
        
        # 选择一个测试加载器
        test_loader = test_loaders[next(iter(test_loaders))]
        
        # 设置优化器和损失函数
        optimizer = optim.Adam(self.fusion_model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # 训练融合模型
        print("\n===== 训练自适应融合模型 =====")
        self.fusion_model.train()
        
        for epoch in range(num_epochs):
            fusion_loss = 0.0
            correct = 0
            total = 0
            
            epoch_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Fusion]')
            for data in epoch_pbar:
                # 处理数据
                if len(data) == 4:
                    inputs, labels, patient_ids, modalities = data
                else:
                    inputs, labels, patient_ids = data
                
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 获取每个模型的预测
                model_outputs = []
                for model_name, model in self.models.items():
                    model.eval()  # 确保模型处于评估模式
                    with torch.no_grad():
                        output = model(inputs)
                    model_outputs.append(output)
                
                # 融合预测
                optimizer.zero_grad()
                weighted_sum, mlp_fusion, weights = self.fusion_model(model_outputs)
                
                # 计算损失 (使用MLP融合结果)
                loss = criterion(mlp_fusion, labels)
                loss.backward()
                optimizer.step()
                
                fusion_loss += loss.item()
                
                # 计算准确率 (使用MLP融合结果)
                _, predicted = mlp_fusion.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 显示权重和准确率
                weight_str = ", ".join([f"{w:.2f}" for w in weights.detach().cpu().numpy()])
                epoch_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%',
                    'weights': weight_str
                })
            
            # 打印当前epoch结果
            print(f'\nEpoch [{epoch+1}/{num_epochs}]:')
            print(f'Fusion Loss: {fusion_loss / len(test_loader):.4f}, '
                  f'Fusion Acc: {100.*correct/total:.2f}%')
            print(f'模型权重: {weight_str}')
        
        # 保存训练好的融合模型
        try:
            os.makedirs('./models', exist_ok=True)
            torch.save(self.fusion_model.state_dict(), './models/fusion_model.pth')
            print('融合模型已保存到 "./models/fusion_model.pth"')
        except Exception as e:
            print(f"保存融合模型时出错: {str(e)}")
        
        return self.fusion_model
    
    def adaptive_fusion_test(self, test_loader):
        """使用自适应融合模型进行测试"""
        self.fusion_model.eval()
        correct = 0
        total = 0
        
        # 用于存储每个患者的预测结果
        patient_predictions = {}
        patient_labels = {}
        
        with torch.no_grad():
            for data in test_loader:
                # 处理数据
                if len(data) == 4:
                    inputs, labels, patient_ids, modalities = data
                else:
                    inputs, labels, patient_ids = data
                
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 获取每个模型的预测
                model_outputs = []
                for model_name, model in self.models.items():
                    model.eval()
                    output = model(inputs)
                    model_outputs.append(output)
                
                # 使用融合模型进行预测
                weighted_sum, mlp_fusion, weights = self.fusion_model(model_outputs)
                
                # 获取预测结果 (使用MLP融合)
                _, predicted = mlp_fusion.max(1)
                
                # 收集患者级别预测
                for i, patient_id in enumerate(patient_ids):
                    if patient_id not in patient_predictions:
                        patient_predictions[patient_id] = []
                        patient_labels[patient_id] = labels[i].item()
                    
                    patient_predictions[patient_id].append(predicted[i].item())
        
        # 计算患者级别准确率
        correct_patients = 0
        for patient_id in patient_predictions:
            preds = patient_predictions[patient_id]
            if preds:
                final_prediction = max(set(preds), key=preds.count)
                if final_prediction == patient_labels[patient_id]:
                    correct_patients += 1
        
        # 计算总体准确率
        accuracy = 100.0 * correct_patients / len(patient_predictions) if patient_predictions else 0
        
        # 获取最终模型权重
        final_weights = F.softmax(self.fusion_model.fusion_weights, dim=0).cpu().numpy()
        weights_dict = {name: weight.item() for name, weight in zip(self.models.keys(), final_weights)}
        
        return accuracy, weights_dict
    
    def load_models(self, model_paths):
        """从文件加载模型"""
        for model_name, path in model_paths.items():
            if model_name in self.models and os.path.exists(path):
                try:
                    checkpoint = torch.load(path)
                    if 'model_state_dict' in checkpoint:
                        self.models[model_name].load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.models[model_name].load_state_dict(checkpoint)
                    print(f"成功加载{model_name}模型：{path}")
                except Exception as e:
                    print(f"加载{model_name}模型时出错: {str(e)}")
            else:
                print(f"无法加载{model_name}模型，模型不存在或路径无效: {path}")
        
        # 如果有融合模型，也加载它
        fusion_path = './models/fusion_model.pth'
        if self.fusion_type == 'adaptive' and os.path.exists(fusion_path):
            try:
                self.fusion_model.load_state_dict(torch.load(fusion_path))
                print(f"成功加载融合模型：{fusion_path}")
            except Exception as e:
                print(f"加载融合模型时出错: {str(e)}")

def train_advanced_models(data_loaders, device, fusion_type='adaptive'):
    """
    训练多个高级模型并进行融合
    
    参数:
    - data_loaders: 包含训练/验证/测试加载器的字典
    - device: 训练设备
    - fusion_type: 融合方法类型
    
    返回:
    - 训练结果和融合模型
    """
    from advanced_models import DenseNet3D, ResNetCBAM3D
    
    # 创建模型
    models = {
        'DenseNet3D': DenseNet3D(in_channels=1, growth_rate=12, 
                               block_config=(4, 8, 12, 8), 
                               num_init_features=32, num_classes=2).to(device),
        'ResNetCBAM3D': ResNetCBAM3D(in_channels=1, num_classes=2, base_channels=16).to(device)
    }
    
    # 创建训练器
    trainer = AdvancedTrainer(models, device, num_classes=2, fusion_type=fusion_type)
    
    # 训练各个模型
    results = trainer.train_individual_models(
        train_loaders=data_loaders['train'],
        val_loaders=data_loaders['val'],
        num_epochs=40,
        learning_rate=0.0005,
        weight_decay=0.01,
        patience=10,
        use_mixup=True
    )
    
    # 如果使用自适应融合，训练融合模型
    if fusion_type == 'adaptive':
        fusion_model = trainer.train_fusion_model(
            test_loaders=data_loaders['test'],
            num_epochs=15,
            learning_rate=0.0002
        )
    else:
        fusion_model = None
    
    # 在测试集上进行评估
    test_loader = data_loaders['test'][next(iter(data_loaders['test']))]
    if fusion_type == 'adaptive':
        accuracy, weights = trainer.adaptive_fusion_test(test_loader)
        print(f"\n自适应融合测试准确率: {accuracy:.2f}%")
        print(f"最终模型权重: {weights}")
    
    return results, fusion_model

# 修改train_enhanced_ensemble函数中的优化器和正则化部分
def train_enhanced_ensemble(data_loaders, device, model_type='densenet'):
    """
    Train the enhanced ensemble model with cross-tissue knowledge transfer
    
    Args:
        data_loaders: Dictionary containing train/val/test loaders for each tissue type
        device: Training device (cuda/cpu)
        model_type: Base model type ('densenet'/'resnet')
    
    Returns:
        Trained ensemble model and performance metrics
    """
    from advanced_models import create_ensemble_model, DenseNet3D, ResNetCBAM3D
    import time
    
    print("\n====== Starting Enhanced Ensemble Training ======")
    
    # Select base model class
    base_model_class = DenseNet3D if model_type == 'densenet' else ResNetCBAM3D
    
    # Create ensemble model
    ensemble_model = create_ensemble_model(device, base_model_class=base_model_class)
    print(f"Created ensemble model with {model_type} base architecture")
    
    # 改进的优化器配置 - 增加权重衰减以减少过拟合
    optimizer = optim.AdamW(ensemble_model.parameters(), lr=0.0003, weight_decay=0.02)
    
    # 使用更适合医学图像的学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # 初始周期长度
        T_mult=2,  # 每次重启后周期长度乘数
        eta_min=1e-6  # 最小学习率
    )
    
    # Initialize mixed precision training
    scaler = amp.GradScaler()
    
    # 添加指数平均移动模型 (EMA)，提高测试时性能
    ema_decay = 0.999
    ema_model = None
    
    # Training parameters
    num_epochs = 50
    patience = 15  # 增加耐心值，避免过早停止
    best_val_acc = 0.0
    no_improve_epochs = 0
    
    # Metrics tracking
    metrics_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Update progressive dropouts
        for tissue, model in ensemble_model.tissue_models.items():
            for module in model.modules():
                if isinstance(module, ProgressiveDropout):
                    module.update_epoch(epoch)
        
        # Training phase
        ensemble_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Create batch iterator for each tissue type
        tissue_types = list(data_loaders['train'].keys())
        batch_iterators = {tissue: iter(data_loaders['train'][tissue]) for tissue in tissue_types}
        max_batches = max(len(data_loaders['train'][tissue]) for tissue in tissue_types)
        
        train_pbar = tqdm(range(max_batches), desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx in train_pbar:
            # Prepare input dictionary for each tissue type
            inputs_dict = {}
            labels_dict = {}
            any_data = False
            
            for tissue in tissue_types:
                try:
                    # Get next batch for this tissue
                    batch_data = next(batch_iterators[tissue])
                    
                    # Handle different data formats
                    if len(batch_data) >= 3:  # Has at least inputs, labels, patient_ids
                        inputs, labels = batch_data[0], batch_data[1]
                        inputs_dict[tissue] = inputs.to(device, non_blocking=True)
                        labels_dict[tissue] = labels.to(device, non_blocking=True)
                        any_data = True
                except StopIteration:
                    # This tissue's dataloader is exhausted
                    pass
            
            if not any_data:
                continue
            
            # Choose a reference tissue for labels
            reference_tissue = next(iter(labels_dict.keys()))
            labels = labels_dict[reference_tissue]
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with amp.autocast():
                # Get model outputs
                outputs = ensemble_model(inputs_dict)
                
                # Use fused output for loss if available
                if 'fused' in outputs:
                    logits = outputs['fused']
                    # 使用标签平滑提高泛化能力
                    loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
                    
                    # Add knowledge distillation loss between tissue models
                    if len(outputs) > 2:  # More than one tissue + fused
                        kd_temp = 3.0  # 降低知识蒸馏温度
                        tissue_outputs = [outputs[t] for t in tissue_types if t in outputs and t != 'fused']
                        
                        # Calculate pairwise KL divergence
                        kd_loss = 0.0
                        for i in range(len(tissue_outputs)):
                            for j in range(i+1, len(tissue_outputs)):
                                # Soften probabilities
                                p_i = F.softmax(tissue_outputs[i] / kd_temp, dim=1)
                                p_j = F.softmax(tissue_outputs[j] / kd_temp, dim=1)
                                
                                # Bidirectional KL divergence
                                kl_ij = F.kl_div(p_i.log(), p_j, reduction='batchmean')
                                kl_ji = F.kl_div(p_j.log(), p_i, reduction='batchmean')
                                kd_loss += (kl_ij + kl_ji) / 2
                        
                        # 增加知识蒸馏权重
                        loss += 0.2 * kd_loss
                        
                        # 添加一致性正则化
                        if np.random.random() < 0.5:  # 随机应用
                            consistency_loss = 0.0
                            for t1 in tissue_outputs:
                                for t2 in tissue_outputs:
                                    if t1 is not t2:
                                        consistency_loss += F.mse_loss(
                                            F.softmax(t1, dim=1), 
                                            F.softmax(t2, dim=1)
                                        )
                            loss += 0.05 * consistency_loss  # 较小权重的一致性损失
                else:
                    # If fusion not possible, use the reference tissue
                    logits = outputs[reference_tissue]
                    loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # 增加梯度裁剪强度，防止权重更新过大
            torch.nn.utils.clip_grad_norm_(ensemble_model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            
            # 更新EMA模型
            if ema_model is None:
                ema_model = {name: param.clone().detach() 
                           for name, param in ensemble_model.named_parameters()}
            else:
                for name, param in ensemble_model.named_parameters():
                    ema_model[name] = ema_model[name] * ema_decay + param.clone().detach() * (1 - ema_decay)
            
            # 根据批次索引更新学习率，实现更平滑的调整
            if batch_idx % 50 == 0:
                scheduler.step()
            
            # Update metrics
            train_loss += loss.item()
            
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Calculate average training metrics
        avg_train_loss = train_loss / max_batches
        train_accuracy = 100. * train_correct / train_total if train_total > 0 else 0
        
        # 暂时应用EMA权重进行验证
        orig_params = {}
        if ema_model is not None:
            for name, param in ensemble_model.named_parameters():
                if param.requires_grad:
                    orig_params[name] = param.data.clone()
                    param.data.copy_(ema_model[name])
        
        # Validation phase
        ensemble_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Track patient-level predictions
        patient_predictions = {}
        patient_labels = {}
        
        with torch.no_grad():
            # Create combined validation dataloader
            val_pbar = tqdm(range(max(len(data_loaders['val'][tissue]) for tissue in tissue_types)), 
                        desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            
            val_iterators = {tissue: iter(data_loaders['val'][tissue]) for tissue in tissue_types}
            
            for _ in val_pbar:
                # Prepare input dictionary
                inputs_dict = {}
                labels = None
                patient_ids = None
                any_data = False
                
                for tissue in tissue_types:
                    try:
                        batch_data = next(val_iterators[tissue])
                        
                        # Process batch data
                        if len(batch_data) >= 3:  # Has at least inputs, labels, patient_ids
                            inputs, batch_labels, batch_patient_ids = batch_data[0], batch_data[1], batch_data[2]
                            inputs_dict[tissue] = inputs.to(device)
                            
                            # Use the first tissue's labels and patient IDs
                            if labels is None:
                                labels = batch_labels.to(device)
                                patient_ids = batch_patient_ids
                            
                            any_data = True
                    except StopIteration:
                        pass
                
                if not any_data or labels is None:
                    continue
                
                # Forward pass
                outputs = ensemble_model(inputs_dict)
                
                # Use fused output if available
                if 'fused' in outputs:
                    logits = outputs['fused']
                else:
                    # Use first available tissue
                    logits = outputs[next(iter(outputs.keys()))]
                
                # Calculate loss
                loss = F.cross_entropy(logits, labels)
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Store patient-level predictions
                for i, patient_id in enumerate(patient_ids):
                    if patient_id not in patient_predictions:
                        patient_predictions[patient_id] = []
                        patient_labels[patient_id] = labels[i].item()
                    
                    patient_predictions[patient_id].append(predicted[i].item())
        
        # 恢复原始模型参数
        if ema_model is not None and orig_params:
            for name, param in ensemble_model.named_parameters():
                if name in orig_params:
                    param.data.copy_(orig_params[name])
        
        # Calculate patient-level accuracy
        patient_correct = 0
        for patient_id, preds in patient_predictions.items():
            # Majority voting
            pred_counts = np.bincount(preds)
            final_pred = np.argmax(pred_counts)
            
            if final_pred == patient_labels[patient_id]:
                patient_correct += 1
        
        patient_accuracy = 100. * patient_correct / len(patient_predictions) if patient_predictions else 0
        val_accuracy = 100. * val_correct / val_total if val_total > 0 else 0
        
        # Calculate average validation loss
        avg_val_loss = val_loss / max(1, sum(1 for _ in range(max(len(data_loaders['val'][tissue]) for tissue in tissue_types))))
        
        # Update metrics history
        metrics_history['train_loss'].append(avg_train_loss)
        metrics_history['train_acc'].append(train_accuracy)
        metrics_history['val_loss'].append(avg_val_loss)
        metrics_history['val_acc'].append(val_accuracy)
        metrics_history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        print(f"Patient-level Val Acc: {patient_accuracy:.2f}%")
        
        # Check for improvement
        if patient_accuracy > best_val_acc:
            best_val_acc = patient_accuracy
            
            # Save best model
            try:
                os.makedirs('./models', exist_ok=True)
                # 保存EMA模型参数
                if ema_model is not None:
                    # 临时应用EMA权重
                    orig_state = {}
                    for name, param in ensemble_model.named_parameters():
                        if param.requires_grad:
                            orig_state[name] = param.data.clone()
                            param.data.copy_(ema_model[name])
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': ensemble_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': patient_accuracy,
                }, './models/best_ensemble_model.pth')
                
                # 恢复原始参数
                if ema_model is not None and orig_state:
                    for name, param in ensemble_model.named_parameters():
                        if name in orig_state:
                            param.data.copy_(orig_state[name])
                
                print(f"Saved best model with val accuracy: {patient_accuracy:.2f}%")
                no_improve_epochs = 0
            except Exception as e:
                print(f"Error saving model: {str(e)}")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs")
        
        # Early stopping check
        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model for evaluation
    try:
        checkpoint = torch.load('./models/best_ensemble_model.pth')
        ensemble_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model with val accuracy: {checkpoint['val_acc']:.2f}%")
    except Exception as e:
        print(f"Error loading best model: {str(e)}")
    
    # Get tissue types from test data loaders
    tissue_types = list(data_loaders['test'].keys())
    
    # Evaluate on test set
    eval_results = evaluate_ensemble(
        ensemble_model, data_loaders['test'], device, tissue_types
    )
    
    print("\n====== Final Test Results ======")
    print(f"Slice-level Test Accuracy: {eval_results['test_accuracy']:.2f}%")
    print(f"Patient-level Test Accuracy: {eval_results['patient_test_accuracy']:.2f}%")
    print("Tissue-specific Accuracies:")
    for tissue, acc in eval_results['tissue_accuracies'].items():
        print(f"  {tissue}: {acc:.2f}%")
    
    return ensemble_model, {
        'metrics_history': metrics_history,
        'test_accuracy': eval_results['test_accuracy'],
        'patient_test_accuracy': eval_results['patient_test_accuracy'],
        'tissue_accuracies': eval_results['tissue_accuracies'],
        'modality_contributions': eval_results['modality_contributions'],
        'relative_contributions': eval_results['relative_contributions'],
        'ensemble_contribution': eval_results['ensemble_contribution'],
        'best_val_accuracy': best_val_acc
    }

def evaluate_ensemble(model, test_loaders, device, tissue_types):
    """
    Evaluate the ensemble model on test data with detailed analysis per modality
    
    Args:
        model: Trained ensemble model
        test_loaders: Dictionary of test data loaders for each tissue type
        device: Device to run evaluation on
        tissue_types: List of tissue types
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # 准备各个指标容器
    all_predictions = []
    all_labels = []
    tissue_correct = {tissue: 0 for tissue in tissue_types}
    tissue_total = {tissue: 0 for tissue in tissue_types}
    
    # 患者级别预测
    patient_predictions = {}
    patient_labels = {}
    
    # 记录每个模态的贡献度
    modality_contributions = {tissue: [] for tissue in tissue_types}
    ensemble_contributions = []
    
    with torch.no_grad():
        # Create iterators for each tissue type
        iterators = {tissue: iter(test_loaders[tissue]) for tissue in tissue_types}
        max_batches = max(len(loader) for loader in test_loaders.values())
        
        for _ in range(max_batches):
            inputs_dict = {}
            labels = None
            patient_ids = None
            
            for tissue in tissue_types:
                try:
                    batch_data = next(iterators[tissue])
                    
                    # Process batch data
                    if len(batch_data) >= 3:  # Has inputs, labels, patient_ids
                        inputs, batch_labels, batch_patient_ids = batch_data[0], batch_data[1], batch_data[2]
                        inputs_dict[tissue] = inputs.to(device)
                        
                        # Use the first tissue's labels and patient IDs
                        if labels is None:
                            labels = batch_labels.to(device)
                            patient_ids = batch_patient_ids
                except StopIteration:
                    pass
            
            if not inputs_dict or labels is None:
                continue
            
            # Forward pass
            outputs = model(inputs_dict)
            
            # Individual tissue predictions
            for tissue in tissue_types:
                if tissue in outputs:
                    # Calculate accuracy for each tissue type
                    _, tissue_preds = outputs[tissue].max(1)
                    tissue_correct[tissue] += tissue_preds.eq(labels).sum().item()
                    tissue_total[tissue] += labels.size(0)
                    
                    # 记录每个模态的贡献 - 计算预测概率与正确标签的相关性
                    probs = F.softmax(outputs[tissue], dim=1)
                    label_probs = probs[torch.arange(probs.size(0)), labels]
                    modality_contributions[tissue].extend(label_probs.cpu().numpy().tolist())  # 转换为列表
            
            # Use fused output if available
            if 'fused' in outputs:
                logits = outputs['fused']
            else:
                # Use first available tissue
                logits = outputs[next(iter(outputs.keys()))]
            
            # Calculate predictions
            _, predicted = logits.max(1)
            
            # Store predictions and labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Store patient-level predictions
            for i, patient_id in enumerate(patient_ids):
                if patient_id not in patient_predictions:
                    patient_predictions[patient_id] = []
                    patient_labels[patient_id] = labels[i].item()
                
                patient_predictions[patient_id].append(predicted[i].item())
                
            # 记录集成模型的贡献
            if 'fused' in outputs:
                fused_probs = F.softmax(outputs['fused'], dim=1)
                fused_label_probs = fused_probs[torch.arange(fused_probs.size(0)), labels]
                ensemble_contributions.extend(fused_label_probs.cpu().numpy().tolist())  # 转换为列表
    
    # Calculate slice-level accuracy
    slice_accuracy = 100. * np.mean(np.array(all_predictions) == np.array(all_labels))
    
    # Calculate tissue-specific accuracies
    tissue_accuracies = {}
    for tissue in tissue_types:
        if tissue_total[tissue] > 0:
            tissue_accuracies[tissue] = 100. * tissue_correct[tissue] / tissue_total[tissue]
        else:
            tissue_accuracies[tissue] = 0.0
    
    # Calculate patient-level accuracy
    patient_correct = 0
    for patient_id, preds in patient_predictions.items():
        # Majority voting
        pred_counts = np.bincount(preds)
        final_pred = np.argmax(pred_counts)
        
        if final_pred == patient_labels[patient_id]:
            patient_correct += 1
    
    patient_accuracy = 100. * patient_correct / len(patient_predictions) if patient_predictions else 0.0
    
    # 计算每个模态的平均贡献
    modality_avg_contributions = {}
    for tissue in tissue_types:
        if modality_contributions[tissue]:
            modality_avg_contributions[tissue] = float(np.mean(modality_contributions[tissue]))  # 确保是标量
        else:
            modality_avg_contributions[tissue] = 0.0
    
    # 计算集成模型的平均贡献
    ensemble_avg_contribution = float(np.mean(ensemble_contributions)) if ensemble_contributions else 0.0
    
    # 分析计算每个模态的相对贡献度
    if modality_avg_contributions:
        total_contribution = sum(modality_avg_contributions.values())
        relative_contributions = {t: float((c/total_contribution)*100) if total_contribution > 0 else 0.0 
                                for t, c in modality_avg_contributions.items()}
    else:
        relative_contributions = {t: 0.0 for t in tissue_types}
    
    # 返回结果指标
    return {
        'test_accuracy': float(slice_accuracy),
        'patient_test_accuracy': float(patient_accuracy),
        'tissue_accuracies': tissue_accuracies,
        'modality_contributions': modality_avg_contributions,
        'relative_contributions': relative_contributions,
        'ensemble_contribution': ensemble_avg_contribution
    } 

def train_improved_resnet(data_loaders, device, save_dir='./models', tissue_types=None):
    """
    使用改进的ResNetCBAM3D模型进行训练，应用多种先进技术减少过拟合
    支持早期融合 - 将CSF、GREY、WHITE三种组织类型作为三通道输入
    
    参数:
    - data_loaders: 包含训练和验证数据加载器的字典
    - device: 训练设备
    - save_dir: 模型保存目录
    - tissue_types: 组织类型列表，默认为['CSF', 'GREY', 'WHITE']
    """
    import torch.optim as optim
    from torch.cuda import amp
    from tqdm import tqdm
    import os
    import numpy as np
    
    if tissue_types is None:
        tissue_types = ['CSF', 'GREY', 'WHITE']
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 记录每个组织类型的结果
    results = {}
    
    for tissue_type in tissue_types:
        print(f"\n===== 训练 {tissue_type} 组织类型的改进ResNetCBAM3D模型 =====")
        
        # 获取数据加载器
        train_loader = data_loaders[f'train_{tissue_type}']
        val_loader = data_loaders[f'val_{tissue_type}']
        
        # 创建改进的ResNetCBAM3D模型
        # 对于早期融合，我们仍使用单通道输入，稍后再修改主训练函数进行多通道融合
        model = create_improved_resnet3d(in_channels=1, num_classes=2, device=device)
        
        # 创建EMA模型 - 减少测试时的波动
        ema_model = EMAModel(model, decay=0.998)
        
        # 使用AdamW优化器，带权重衰减
        optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)  # 降低学习率，增加权重衰减
        
        # 使用余弦退火学习率调度器，预热阶段和更好的冷却
        total_epochs = 60  # 增加总轮次
        warmup_epochs = 5
        
        # 余弦退火学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,  # 第一个退火周期
            T_mult=2,  # 每个周期的倍增因子
            eta_min=1e-6  # 最小学习率
        )
        
        # 使用带标签平滑的交叉熵损失
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 混合精度训练
        scaler = amp.GradScaler()
        
        # 初始化训练变量
        best_val_acc = 0.0
        best_val_loss = float('inf')
        best_model_state = None
        no_improve_epochs = 0
        patience = 12  # 增加耐心值
        
        # 主训练循环
        for epoch in range(total_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # 自定义学习率预热
            if epoch < warmup_epochs:
                # 线性预热，低起点
                warmup_factor = (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.0003 * warmup_factor  # 降低学习率
            
            # 训练进度条
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs} [Train {tissue_type}]')
            
            for batch in train_pbar:
                # 确保我们正确处理数据加载器返回的内容
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 2:
                        inputs, labels = batch
                    elif len(batch) == 3:
                        inputs, labels, _ = batch  # 忽略第三个返回值
                    else:
                        inputs, labels = batch[0], batch[1]  # 只取前两个值
                else:
                    # 如果dataloader返回字典
                    inputs = batch['image'] if 'image' in batch else batch['input']
                    labels = batch['label'] if 'label' in batch else batch['target']
                
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # 随机深度缩放 - 更适合小数据集
                depth_factor = np.random.uniform(0.85, 1.0) if epoch > warmup_epochs else 1.0
                
                # 清零梯度
                optimizer.zero_grad()
                
                # 混合精度训练
                with amp.autocast():
                    outputs = model(inputs)
                    
                    # 应用CutMix数据增强（概率0.3）
                    if np.random.random() < 0.3 and epoch > warmup_epochs:
                        # 创建混合样本的索引和权重
                        indices = torch.randperm(inputs.size(0)).to(device)
                        lam = np.random.beta(1.0, 1.0)
                        
                        # 创建混合尺寸
                        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                        
                        # 执行CutMix
                        inputs[:, :, bbx1:bbx2, bby1:bby2, :] = inputs[indices, :, bbx1:bbx2, bby1:bby2, :]
                        
                        # 混合输出和损失
                        mixed_outputs = model(inputs)
                        mixed_loss = lam * criterion(mixed_outputs, labels) + (1 - lam) * criterion(mixed_outputs, labels[indices])
                        loss = mixed_loss
                    else:
                        # 正常损失
                        loss = criterion(outputs, labels)
                    
                    # 添加L1正则化 - 稀疏模型权重
                    if epoch > warmup_epochs//2:
                        l1_regularization = 0.0
                        for param in model.parameters():
                            l1_regularization += torch.norm(param, 1)
                        loss += 0.0001 * l1_regularization
                
                # 反向传播与优化
                scaler.scale(loss).backward()
                
                # 梯度裁剪 - 增加裁剪阈值以提高稳定性
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)  # 增加裁剪阈值
                
                scaler.step(optimizer)
                scaler.update()
                
                # 更新EMA模型
                ema_model.update(model)
                
                # 计算训练指标
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                train_loss += loss.item()
                
                # 更新进度条
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            # 应用学习率调度器
            if epoch >= warmup_epochs:
                scheduler.step()
            
            # 验证阶段 - 使用EMA模型
            ema_model.apply_shadow()
            val_loss, val_acc = validate_model_with_ema(ema_model.module, val_loader, criterion, device)
            ema_model.restore()
            
            # 打印训练信息
            current_lr = optimizer.param_groups[0]['lr']
            print(f'\nEpoch [{epoch+1}/{total_epochs}] - {tissue_type}:')
            print(f'Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {100.*train_correct/train_total:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Learning Rate: {current_lr:.6f}')
            
            # 保存最佳模型
            improved = False
            
            # 判断是否有改进
            if val_acc > best_val_acc:
                improved = True
                best_val_acc = val_acc
            elif val_acc == best_val_acc and val_loss < best_val_loss:
                improved = True
                best_val_loss = val_loss
            
            if improved:
                # 保存EMA模型状态
                ema_model.apply_shadow()
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': ema_model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }
                ema_model.restore()
                
                # 保存模型
                model_filename = f'{save_dir}/best_improved_resnet_{tissue_type}.pth'
                torch.save(best_model_state, model_filename)
                print(f'保存最佳模型，验证准确率: {val_acc:.2f}%')
                
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                print(f'验证准确率连续{no_improve_epochs}个epoch没有提高')
            
            # 早停检查
            if no_improve_epochs >= patience:
                print(f'早停在epoch {epoch+1}')
                break
        
        # 训练结束后加载最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state['model_state_dict'])
            print(f'已加载{tissue_type}的最佳模型，验证准确率: {best_val_acc:.2f}%')
        
        # 记录结果
        results[tissue_type] = {
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'best_epoch': best_model_state['epoch'] if best_model_state else -1,
            'model_path': f'{save_dir}/best_improved_resnet_{tissue_type}.pth'
        }
    
    # 返回训练结果
    return results

def rand_bbox(size, lam):
    """用于CutMix的随机边界框生成函数"""
    W = size[2]
    H = size[3]
    D = size[4]
    
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cut_d = int(D * cut_rat)
    
    # 随机中心点
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    cz = np.random.randint(D)
    
    # 边界框坐标
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2

def validate_model_with_ema(model, val_loader, criterion, device):
    """使用验证集评估模型（配合EMA使用）"""
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # 确保我们正确处理数据加载器返回的内容
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    inputs, labels = batch
                elif len(batch) == 3:
                    inputs, labels, _ = batch  # 忽略第三个返回值
                else:
                    inputs, labels = batch[0], batch[1]  # 只取前两个值
            else:
                # 如果dataloader返回字典
                inputs = batch['image'] if 'image' in batch else batch['input']
                labels = batch['label'] if 'label' in batch else batch['target']
                
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 统计
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    # 计算最终指标
    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100. * val_correct / val_total
    
    return avg_val_loss, val_acc

def train_fusion_with_improved_models(data_loaders, device, model_paths, tissue_types=None, save_dir='./models'):
    """
    使用预训练的改进模型进行融合训练
    
    参数:
    - data_loaders: 包含训练和验证数据加载器的字典
    - device: 训练设备
    - model_paths: 预训练模型的路径字典
    - tissue_types: 组织类型列表，默认为['CSF', 'GREY', 'WHITE']
    - save_dir: 保存目录
    """
    if tissue_types is None:
        tissue_types = ['CSF', 'GREY', 'WHITE']
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建各组织类型的基础模型
    base_models = {}
    
    for tissue_type in tissue_types:
        # 创建模型
        model = create_improved_resnet3d(in_channels=1, num_classes=2, device=device)
        
        # 加载预训练权重
        if tissue_type in model_paths:
            try:
                checkpoint = torch.load(model_paths[tissue_type])
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f'已加载{tissue_type}预训练模型，验证准确率: {checkpoint["val_acc"]:.2f}%')
            except Exception as e:
                print(f'加载{tissue_type}模型时出错: {str(e)}')
        
        # 冻结基础模型的参数
        for param in model.parameters():
            param.requires_grad = False
        
        # 只允许分类层微调
        for param in model.classifier1.parameters():
            param.requires_grad = True
        for param in model.classifier2.parameters():
            param.requires_grad = True
        
        # 将模型设为评估模式
        model.eval()
        base_models[tissue_type] = model
    
    print("已加载所有基础模型，创建融合模型...")
    
    # 创建自适应融合模型
    class ImprovedFusionModel(nn.Module):
        def __init__(self, base_models, feature_dim=256, num_classes=2):
            super(ImprovedFusionModel, self).__init__()
            self.base_models = nn.ModuleDict(base_models)
            self.tissue_types = list(base_models.keys())
            
            # 注意力融合模块
            self.attention = nn.Sequential(
                nn.Linear(feature_dim * len(self.tissue_types), 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, len(self.tissue_types))
            )
            
            # 特征投影层
            self.projection = nn.ModuleDict({
                tissue: nn.Sequential(
                    nn.Linear(feature_dim, feature_dim),
                    nn.LayerNorm(feature_dim),
                    nn.ReLU(inplace=True)
                ) for tissue in self.tissue_types
            })
            
            # 最终分类器
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, num_classes)
            )
        
        def forward(self, inputs_dict):
            # 从各模型提取特征
            features = {}
            raw_logits = {}
            
            for tissue, model in self.base_models.items():
                if tissue in inputs_dict:
                    # 提取特征
                    with torch.no_grad():
                        features[tissue] = model(inputs_dict[tissue], return_features=True)
                    
                    # 投影特征
                    features[tissue] = self.projection[tissue](features[tissue])
            
            # 检查是否所有组织类型都有输入
            if len(features) < len(self.tissue_types):
                missing = set(self.tissue_types) - set(features.keys())
                raise ValueError(f"缺少以下组织类型的输入: {missing}")
            
            # 连接特征
            concat_features = torch.cat([features[t] for t in self.tissue_types], dim=1)
            
            # 计算注意力权重
            attention_weights = F.softmax(self.attention(concat_features), dim=1)
            
            # 加权融合特征
            fused_feature = torch.zeros_like(features[self.tissue_types[0]])
            for i, tissue in enumerate(self.tissue_types):
                fused_feature += features[tissue] * attention_weights[:, i].unsqueeze(1)
            
            # 最终分类
            outputs = self.classifier(fused_feature)
            
            return outputs, attention_weights
    
    # 实例化融合模型
    fusion_model = ImprovedFusionModel(base_models, feature_dim=256, num_classes=2).to(device)
    
    # 设置优化器 - 只训练融合模型的部分
    optimizer = optim.AdamW(fusion_model.parameters(), lr=0.0003, weight_decay=0.01)
    
    # 使用余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    
    # 使用交叉熵损失
    criterion = nn.CrossEntropyLoss()
    
    # 训练参数
    num_epochs = 40
    best_val_acc = 0.0
    best_model_state = None
    patience = 10
    no_improve_epochs = 0
    
    # 获取训练和验证数据加载器
    train_loaders = {t: data_loaders[f'train_{t}'] for t in tissue_types}
    val_loaders = {t: data_loaders[f'val_{t}'] for t in tissue_types}
    
    # 确保所有数据加载器的长度一致
    min_train_len = min(len(loader) for loader in train_loaders.values())
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        fusion_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 创建训练数据迭代器
        train_iterators = {t: iter(loader) for t, loader in train_loaders.items()}
        
        # 创建进度条
        pbar = tqdm(range(min_train_len), desc=f'Epoch {epoch+1}/{num_epochs} [Train Fusion]')
        
        for _ in pbar:
            # 收集各组织类型的数据
            inputs_dict = {}
            labels = None
            
            try:
                for tissue_type in tissue_types:
                    # 获取当前组织类型的数据
                    data = next(train_iterators[tissue_type])
                    inputs, batch_labels = data
                    
                    # 转移到设备
                    inputs_dict[tissue_type] = inputs.to(device)
                    
                    # 保存标签（所有组织类型应具有相同的标签）
                    if labels is None:
                        labels = batch_labels.to(device)
                
                # 清零梯度
                optimizer.zero_grad()
                
                # 前向传播
                outputs, attention_weights = fusion_model(inputs_dict)
                
                # 计算损失
                loss = criterion(outputs, labels)
                
                # 反向传播和优化
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # 计算统计数据
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%',
                    'attn': f'{attention_weights.mean(0).detach().cpu().numpy()}'
                })
                
            except StopIteration:
                break
        
        # 更新学习率
        scheduler.step()
        
        # 验证阶段
        fusion_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # 确保所有验证数据加载器的长度一致
        min_val_len = min(len(loader) for loader in val_loaders.values())
        
        # 创建验证数据迭代器
        val_iterators = {t: iter(loader) for t, loader in val_loaders.items()}
        
        with torch.no_grad():
            for _ in range(min_val_len):
                # 收集各组织类型的数据
                inputs_dict = {}
                labels = None
                
                try:
                    for tissue_type in tissue_types:
                        # 获取当前组织类型的数据
                        data = next(val_iterators[tissue_type])
                        inputs, batch_labels = data
                        
                        # 转移到设备
                        inputs_dict[tissue_type] = inputs.to(device)
                        
                        # 保存标签
                        if labels is None:
                            labels = batch_labels.to(device)
                    
                    # 前向传播
                    outputs, _ = fusion_model(inputs_dict)
                    
                    # 计算损失
                    loss = criterion(outputs, labels)
                    
                    # 统计
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
                except StopIteration:
                    break
        
        # 计算验证指标
        avg_val_loss = val_loss / min_val_len if min_val_len > 0 else 0
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0
        
        # 打印训练信息
        print(f'\nEpoch [{epoch+1}/{num_epochs}] - Fusion:')
        print(f'Train Loss: {train_loss / min_train_len:.4f}, Train Acc: {100.*train_correct/train_total:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': fusion_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }
            
            # 保存模型
            torch.save(best_model_state, f'{save_dir}/best_improved_fusion.pth')
            print(f'保存最佳融合模型，验证准确率: {val_acc:.2f}%')
            
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f'验证准确率连续{no_improve_epochs}个epoch没有提高')
        
        # 早停检查
        if no_improve_epochs >= patience:
            print(f'早停在epoch {epoch+1}')
            break
    
    # 加载最佳模型
    if best_model_state is not None:
        fusion_model.load_state_dict(best_model_state['model_state_dict'])
        print(f'已加载最佳融合模型，验证准确率: {best_val_acc:.2f}%')
    
    return fusion_model, best_val_acc 
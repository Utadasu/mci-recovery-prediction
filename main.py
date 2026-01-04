import os
import torch
import torch.nn as nn
import json
import numpy as np
import random
from torch.utils.data import Subset, DataLoader

from data_utils import create_tissue_specific_dataset, create_data_loaders
from advanced_trainer import AdvancedTrainer, train_advanced_models, train_improved_resnet, train_fusion_with_improved_models
from optimized_models import create_improved_resnet3d
from early_fusion import train_early_fusion_model, train_hierarchical_swin_model
from early_fusion_fixed import train_memory_optimized_early_fusion
from quick_finetune import quick_finetune_model
from deep_architecture_finetune import deep_architecture_finetune

# 设置CUDA内存分配器并优化性能
torch.cuda.set_per_process_memory_fraction(0.95)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# 设置环境变量以优化CUDA性能
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"



def train_improved_tissue_models(data_path, device, export_path='./models'):
    """使用改进的ResNetCBAM3D模型训练各个组织类型的模型"""
    tissue_types = ['CSF', 'GRAY', 'WHITE']
    
    # 创建导出目录
    os.makedirs(export_path, exist_ok=True)
    
    # 准备数据加载器字典
    data_loaders = {}
    
    for tissue_type in tissue_types:
        print(f"\n准备 {tissue_type} 组织的数据...")
        
        # 创建数据集
        dataset = create_tissue_specific_dataset(data_path, tissue_type)
        
        # 创建训练和验证数据加载器
        train_loader, val_loader = create_data_loaders(dataset, batch_size=16)
        
        # 保存数据加载器
        data_loaders[f'train_{tissue_type}'] = train_loader
        data_loaders[f'val_{tissue_type}'] = val_loader
        
        print(f"{tissue_type} 数据集统计:")
        print(f"训练样本数: {len(train_loader.dataset)}")
        print(f"验证样本数: {len(val_loader.dataset)}")
        
    # 使用改进的训练函数训练模型
    results = train_improved_resnet(
        data_loaders=data_loaders,
        device=device,
        save_dir=export_path,
        tissue_types=tissue_types
    )
    
    # 打印最终结果
    print("\n===== 改进模型训练结果 =====")
    for tissue_type, result in results.items():
        print(f"{tissue_type}: 最佳验证准确率 = {result['best_val_acc']:.2f}%, "
              f"最佳轮次 = {result['best_epoch']}")
    
    return results

def train_improved_fusion(data_loaders, device, export_path='./models'):
    """训练改进的融合模型"""
    tissue_types = ['CSF', 'GRAY', 'WHITE']
    
    # 准备模型路径
    model_paths = {}
    for tissue_type in tissue_types:
        model_path = f"{export_path}/best_improved_resnet_{tissue_type}.pth"
        if os.path.exists(model_path):
            model_paths[tissue_type] = model_path
            print(f"找到 {tissue_type} 预训练模型: {model_path}")
        else:
            print(f"警告: 未找到 {tissue_type} 预训练模型")
    
    if not model_paths:
        print("错误: 未找到任何预训练模型。请先训练单个改进模型。")
        return None, 0
    
    # 训练融合模型
    fusion_model, best_val_acc = train_fusion_with_improved_models(
        data_loaders=data_loaders,
        device=device,
        model_paths=model_paths,
        tissue_types=tissue_types,
        save_dir=export_path
    )
    
    print(f"\n融合模型训练完成，最佳验证准确率: {best_val_acc:.2f}%")
    
    return fusion_model, best_val_acc

def create_unified_dataset(data_path):
    """创建统一的数据集，确保每个患者的三个模态都被加载"""
    # 创建各个组织类型的数据集
    print("\n====== 创建数据集 ======")
    modality_datasets = {}
    
    for tissue_type in ['CSF', 'GRAY', 'WHITE']:
        print(f"\n加载 {tissue_type} 数据集...")
        modality_datasets[tissue_type] = create_tissue_specific_dataset(data_path, tissue_type)
    
    # 创建患者ID到各模态样本索引的映射
    patient_modalities = {}
    
    # 处理每个模态的数据集
    for tissue_type, dataset in modality_datasets.items():
        for i, patient_id in enumerate(dataset.patient_ids):
            if patient_id not in patient_modalities:
                patient_modalities[patient_id] = {'label': dataset.labels[i], 'modalities': {}}
            
            # 保存该患者该模态的索引
            patient_modalities[patient_id]['modalities'][tissue_type] = i
    
    # 筛选出拥有所有三个模态的患者
    complete_patients = []
    for patient_id, info in patient_modalities.items():
        if len(info['modalities']) == 3:  # 患者有全部三个模态
            complete_patients.append(patient_id)
    
    print(f"\n拥有全部三个模态的患者数: {len(complete_patients)}")
    print(f"拥有部分模态的患者数: {len(patient_modalities) - len(complete_patients)}")
    
    if len(complete_patients) == 0:
        raise ValueError("没有找到拥有全部三个模态的患者！")
    
    return modality_datasets, complete_patients, patient_modalities

def create_patient_aware_splits(modality_datasets, complete_patients, patient_modalities):
    """创建考虑患者整体的数据集划分，保证AD/CN分布平衡"""
    # 按标签(AD/CN)分组
    ad_patients = []
    cn_patients = []
    
    for patient_id in complete_patients:
        if patient_modalities[patient_id]['label'] == 0:  # AD
            ad_patients.append(patient_id)
        else:  # CN
            cn_patients.append(patient_id)
    
    print(f"\n按疾病分组的患者统计:")
    print(f"AD患者总数: {len(ad_patients)}")
    print(f"CN患者总数: {len(cn_patients)}")
    
    # 随机打乱，但保持疾病类别独立
    random.shuffle(ad_patients)
    random.shuffle(cn_patients)
    
    # 为每个类别分别划分
    train_ad = ad_patients[:int(len(ad_patients)*0.7)]
    val_ad = ad_patients[int(len(ad_patients)*0.7):int(len(ad_patients)*0.85)]
    test_ad = ad_patients[int(len(ad_patients)*0.85):]
    
    train_cn = cn_patients[:int(len(cn_patients)*0.7)]
    val_cn = cn_patients[int(len(cn_patients)*0.7):int(len(cn_patients)*0.85)]
    test_cn = cn_patients[int(len(cn_patients)*0.85):]
    
    # 合并保持平衡的划分
    train_patients = train_ad + train_cn
    val_patients = val_ad + val_cn
    test_patients = test_ad + test_cn
    
    # 打乱合并后的列表，保持AD/CN比例但随机排序
    random.shuffle(train_patients)
    random.shuffle(val_patients)
    random.shuffle(test_patients)
    
    print(f"\n按疾病平衡的数据集划分信息:")
    print(f"训练集: AD={len(train_ad)}, CN={len(train_cn)}, 总计={len(train_patients)}")
    print(f"验证集: AD={len(val_ad)}, CN={len(val_cn)}, 总计={len(val_patients)}")
    print(f"测试集: AD={len(test_ad)}, CN={len(test_cn)}, 总计={len(test_patients)}")
    
    # 为每个组织类型创建数据加载器
    train_loaders = {}
    val_loaders = {}
    test_loaders = {}
    
    for tissue_type, dataset in modality_datasets.items():
        # 创建索引
        train_indices = [patient_modalities[pid]['modalities'][tissue_type] 
                         for pid in train_patients if tissue_type in patient_modalities[pid]['modalities']]
        val_indices = [patient_modalities[pid]['modalities'][tissue_type] 
                         for pid in val_patients if tissue_type in patient_modalities[pid]['modalities']]
        test_indices = [patient_modalities[pid]['modalities'][tissue_type] 
                         for pid in test_patients if tissue_type in patient_modalities[pid]['modalities']]
        
        # 创建子集
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)
        
        print(f"\n{tissue_type} 数据集:")
        print(f"训练集样本数: {len(train_indices)}")
        print(f"验证集样本数: {len(val_indices)}")
        print(f"测试集样本数: {len(test_indices)}")
        
        # 创建数据加载器
        train_loaders[tissue_type] = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        
        val_loaders[tissue_type] = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
        
        test_loaders[tissue_type] = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
    
    return train_loaders, val_loaders, test_loaders

def train_quick_validation_model(data_loaders, device, save_dir='./models'):
    """
    快速验证模型 - 用于调试和验证训练流程
    使用轻量级架构，快速收敛，便于发现问题
    """
    import torch.optim as optim
    from torch.cuda import amp
    from tqdm import tqdm
    import os
    import numpy as np
    
    print("\n===== 快速验证模型训练 =====")
    
    # 创建轻量级验证模型
    class QuickValidationModel(nn.Module):
        def __init__(self, in_channels=3, num_classes=2):
            super(QuickValidationModel, self).__init__()
            
            # 极简的特征提取器
            self.features = nn.Sequential(
                # 第一层 - 大幅降采样
                nn.Conv3d(in_channels, 8, kernel_size=7, stride=4, padding=3),
                nn.BatchNorm3d(8),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                
                # 第二层
                nn.Conv3d(8, 16, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                
                # 第三层
                nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool3d((1, 1, 1))
            )
            
            # 简单分类器
            self.classifier = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(16, num_classes)
            )
            
            # 权重初始化
            self._initialize_weights()
        
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
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    # 创建早期融合数据加载器
    from early_fusion import create_early_fusion_loaders
    
    # 准备数据加载器字典
    train_data_loaders = {f'train_{k}': v for k, v in data_loaders['train'].items()}
    val_data_loaders = {f'val_{k}': v for k, v in data_loaders['val'].items()}
    
    fusion_loaders = create_early_fusion_loaders(
        {**train_data_loaders, **val_data_loaders}, 
        batch_size=32,  # 从8改为32，充分利用32GB GPU显存
        debug=True
    )
    
    train_loader = fusion_loaders['train']
    val_loader = fusion_loaders['val']
    
    # 创建模型
    model = QuickValidationModel(in_channels=3, num_classes=2).to(device)
    
    # 测试模型前向传播
    sample_batch, sample_labels = next(iter(train_loader))
    sample_batch = sample_batch.to(device)
    with torch.no_grad():
        sample_output = model(sample_batch)
        print(f"模型输出形状: {sample_output.shape}")
        print(f"输出范围: [{sample_output.min().item():.3f}, {sample_output.max().item():.3f}]")
    
    # 使用AdamW优化器，较高学习率快速收敛
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    
    # 使用余弦退火调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
    
    # 检查数据集标签分布
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.numpy())
    
    label_counts = np.bincount(train_labels)
    print(f"训练集标签分布: {label_counts}")
    
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
    
    # 交叉熵损失
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 混合精度训练
    scaler = amp.GradScaler()
    
    # 训练参数
    num_epochs = 25  # 快速训练
    best_val_acc = 0.0
    best_model_state = None
    patience = 8  # 较短的耐心值
    no_improve_epochs = 0
    
    # 训练统计
    stats = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"开始快速验证训练，总轮次: {num_epochs}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Quick Val]')
        
        for inputs, labels in train_pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 混合精度训练
            with amp.autocast():
                outputs = model(inputs)
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
        
        # 更新学习率
        scheduler.step()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # 混淆矩阵
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
        
        # 打印信息
        print(f'\nEpoch [{epoch+1}/{num_epochs}] - Quick Validation:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Val Acc per class: {val_acc_per_class}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'混淆矩阵:\n{conf_matrix}')
        
        # 检查是否两个类别都有预测
        both_classes_predicted = conf_matrix[0, 0] > 0 and conf_matrix[1, 1] > 0
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': avg_val_loss,
                'stats': stats,
                'conf_matrix': conf_matrix.tolist()
            }
            
            torch.save(best_model_state, f'{save_dir}/best_quick_validation_model.pth')
            print(f'保存最佳模型，验证准确率: {val_acc:.2f}%')
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        
        # 早停检查
        if no_improve_epochs >= patience:
            print(f'早停在epoch {epoch+1}')
            break
        
        # 如果模型开始预测两个类别，说明训练正常
        if both_classes_predicted:
            print("✓ 模型正常：能够预测两个类别")
        else:
            print("⚠ 警告：模型只预测单一类别")
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state_dict'])
        print(f'已加载最佳快速验证模型，验证准确率: {best_val_acc:.2f}%')
    
    return {
        'model': model,
        'best_val_acc': best_val_acc,
        'best_epoch': best_model_state['epoch'] if best_model_state else -1,
        'model_path': f'{save_dir}/best_quick_validation_model.pth',
        'stats': stats,
        'final_conf_matrix': conf_matrix.tolist()
    }

def main():
    # 设置随机种子
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 请求用户输入数据目录
    print("\n请输入数据目录路径:")
    data_root = input("数据根目录路径: ").strip()
    if not data_root:
        # 默认路径 - 遵循MCI_DATA规范
        data_root = "/root/autodl-tmp/MCI_DATA"

    # 设置数据路径 - 更新为用户输入或默认路径，遵循MCI_DATA规范
    data_path = {
        'ad_dir': os.path.join(data_root, "totalAD"),
        'cn_dir': os.path.join(data_root, "totalCN")
    }
    
    # 验证基础目录是否存在
    print("\n验证基础目录:")
    print(f"AD基础目录: {data_path['ad_dir']}")
    print(f"AD基础目录存在: {os.path.exists(data_path['ad_dir'])}")
    print(f"CN基础目录: {data_path['cn_dir']}")
    print(f"CN基础目录存在: {os.path.exists(data_path['cn_dir'])}")
    
    if not os.path.exists(data_path['ad_dir']):
        raise ValueError(f"AD基础目录不存在: {data_path['ad_dir']}")
    if not os.path.exists(data_path['cn_dir']):
        raise ValueError(f"CN基础目录不存在: {data_path['cn_dir']}")
    
    # 列出基础目录的内容
    print("\nAD基础目录内容:")
    for item in os.listdir(data_path['ad_dir']):
        print(f"  - {item}")
    
    print("\nCN基础目录内容:")
    for item in os.listdir(data_path['cn_dir']):
        print(f"  - {item}")
    
    # 创建数据加载器
    print("\n===== 创建数据加载器 =====")
    try:
        # 创建统一数据集
        modality_datasets, complete_patients, patient_modalities = create_unified_dataset(data_path)
        
        # 创建患者感知的数据划分
        train_loaders, val_loaders, test_loaders = create_patient_aware_splits(
            modality_datasets, complete_patients, patient_modalities
        )
        
        # 组织数据加载器为期望的格式
        data_loaders = {
            'train': train_loaders,
            'val': val_loaders,
            'test': test_loaders
        }
        
        print("✅ 数据加载器创建成功")
        print(f"✅ 训练加载器: {list(train_loaders.keys())}")
        print(f"✅ 验证加载器: {list(val_loaders.keys())}")
        print(f"✅ 测试加载器: {list(test_loaders.keys())}")
        
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}")
        return
    
    # 设置模型保存目录
    model_save_dir = './models'
    os.makedirs(model_save_dir, exist_ok=True)
    
    # 初始化结果列表
    all_results = []
    
    # 询问用户选择模型架构和训练方法
    print("\n请选择要训练的模型:")
    print("1. 训练所有组织类型的模型 (CSF, GRAY, WHITE)")
    print("3. 训练早期融合模型")
    print("6. 层次化Swin-Transformer - 队友提出的新架构 (最新)")
    print("7. 内存优化版早期融合模型训练 - 修复版 (推荐)")
    print("8. 深度架构微调 - 更深层次的信息保留优化")
    print("0. 退出")
    model_choice = input("请输入选项 (0,1,3,6-8): ").strip()
    
    if model_choice == "1":
        # 使用原始的高级训练函数
        print("\n====== 训练所有组织类型的模型 ======")
        results, fusion_model = train_advanced_models(data_loaders, device, fusion_type='adaptive')
        
        # 打印各个模型的性能
        print("\n====== 单一模型性能 ======")
        model_accuracies = {}
        for model_name, result in results.items():
            accuracy = result['best_val_acc']
            model_accuracies[model_name] = accuracy
            print(f"{model_name} 模型验证准确率: {accuracy:.2f}%")
        
        # 保存结果
        try:
            results_file = 'model_results.json'
            with open(results_file, 'w') as f:
                json.dump(model_accuracies, f, indent=4)
            print(f"\n结果已保存到 {results_file}")
        except Exception as e:
            print(f"保存结果时出错: {str(e)}")
        
        all_results.append({
            'type': '所有组织类型模型',
            'results': results
        })
    
    elif model_choice == "3":
        print("=== 训练早期融合模型 ===")
        from early_fusion import train_early_fusion_model, create_early_fusion_loaders
        
        # 创建早期融合数据加载器
        fusion_loaders = create_early_fusion_loaders(data_loaders, batch_size=4, debug=True)
        
        # 训练早期融合模型
        fusion_results = train_early_fusion_model(
            fusion_loaders,  # 传递正确格式的数据加载器
            device,
            save_dir=model_save_dir
        )
        
        all_results.append({
            'type': '早期融合',
            'results': fusion_results
        })
    
    elif model_choice == "6":
        print("=== 训练层次化Swin-Transformer模型 ===")
        from early_fusion import train_hierarchical_swin_model
        model_info = train_hierarchical_swin_model(data_loaders, device, save_dir=model_save_dir)
        all_results.append({
            'type': '层次化Swin-Transformer',
            'results': model_info
        })
        
    elif model_choice == "7":
        print("=== 内存优化版早期融合模型训练 ===")
        # 使用修复版训练函数
        model_info = train_memory_optimized_early_fusion(data_loaders, device, save_dir=model_save_dir)
        all_results.append({
            'type': '内存优化早期融合',
            'results': model_info
        })
    
    elif model_choice == "8":
        print("=== 深度架构微调 - 更深层次的信息保留优化 ===")
        # 检查是否存在预训练模型
        model_path = "./models/best_memory_optimized_early_fusion.pth"
        if not os.path.exists(model_path):
            print("❌ 未找到预训练模型，请先训练模型")
            print("   建议选择选项7进行训练")
        else:
            print(f"✅ 找到预训练模型: {model_path}")
            print("🔧 将基于现有模型创建增强版架构...")
            
            # 询问训练轮次
            try:
                epochs = int(input("请输入训练轮次 (推荐10-15轮): ").strip() or "10")
                if epochs < 5 or epochs > 30:
                    print("轮次应在5-30之间，使用默认值10")
                    epochs = 10
            except ValueError:
                print("输入无效，使用默认值10")
                epochs = 10
            
            # 执行深度架构微调
            best_acc = deep_architecture_finetune(model_path, data_loaders, device, epochs=epochs)
            
            if best_acc:
                all_results.append({
                    'type': f'深度架构微调({epochs}轮)',
                    'results': {'best_val_acc': best_acc}
                })
    
    elif model_choice == "0":
        print("退出程序")
        return
    
    else:
        print("无效选项，请重新选择")
        return
    
    # 训练完成后的结果汇总
    print("\n" + "="*50)
    print("           训练完成 - 结果汇总")
    print("="*50)
    
    if all_results:
        for result in all_results:
            result_type = result['type']
            result_data = result['results']
            
            print(f"\n🔹 {result_type}:")
            if isinstance(result_data, dict):
                if 'best_val_acc' in result_data:
                    print(f"   最佳验证准确率: {result_data['best_val_acc']:.2f}%")
                if 'best_epoch' in result_data:
                    print(f"   最佳轮次: {result_data['best_epoch']}")
                if 'model_path' in result_data and result_data['model_path']:
                    print(f"   模型保存路径: {result_data['model_path']}")
            else:
                print(f"   结果: {result_data}")
        
        print(f"\n✅ 所有训练任务已完成！")
        print(f"📁 模型保存目录: {model_save_dir}")
    else:
        print("⚠️ 没有训练任务完成")

if __name__ == "__main__":
    main() 
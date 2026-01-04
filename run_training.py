#!/usr/bin/env python3
"""
🚀 一键启动脚本 - 智能下采样 + 对比学习训练
==============================================

功能特性:
- 🔥 智能下采样层ImprovedResNetCBAM3D模型训练
- 🎯 多模态对比学习训练
- 📊 自动模型路径管理
- 🔧 GPU内存自适应配置
"""

import os
import sys
import torch
import argparse
from datetime import datetime

def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
        print(f"✅ GPU: {gpu_name} ({gpu_memory}GB)")
    else:
        print("❌ CUDA不可用，将使用CPU")
        return False
    
    # 检查数据路径
    data_paths = [
        "/root/autodl-tmp/DATA_MCI/test_data/",
        "./test_data/",
        "../test_data/"
    ]
    
    data_path = None
    for path in data_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path:
        print(f"✅ 数据路径: {data_path}")
    else:
        print("❌ 未找到数据路径")
        return False
    
    # 检查模型目录
    os.makedirs('./models', exist_ok=True)
    print("✅ 模型目录已准备")
    
    return True, data_path

def train_smart_downsample_model(data_path, device='cuda'):
    """训练智能下采样模型"""
    print("\n🔥 步骤1: 训练智能下采样模型")
    print("=" * 50)
    
    from train_smart_downsample import SmartDownsampleTrainer
    
    # 创建训练器
    trainer = SmartDownsampleTrainer(device=device, save_dir='./models')
    
    # 训练智能下采样模型
    model, best_acc, model_path = trainer.train(
        data_path=data_path,
        use_global_pool=False,  # 使用智能下采样
        base_channels=12,
        num_epochs=30,  # 适中的训练轮数
        batch_size=4,
        learning_rate=1e-4,
        max_samples=None,
        patience=10
    )
    
    print(f"\n✅ 智能下采样模型训练完成!")
    print(f"   最佳准确率: {best_acc:.2f}%")
    print(f"   模型路径: {model_path}")
    
    return model_path, best_acc

def train_contrastive_model(image_model_path, data_path, device='cuda'):
    """训练对比学习模型"""
    print("\n🎯 步骤2: 训练多模态对比学习模型")
    print("=" * 50)
    
    from contrastive_learning import create_contrastive_model
    from data_utils import create_multimodal_dataset_from_excel
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    
    # 🔥 优先使用对比学习专用图像编码器
    contrastive_image_paths = [
        './models/contrastive/contrastive_image_encoder_ch12.pth',
        './models/contrastive/contrastive_image_encoder_ch8.pth',
        image_model_path  # 回退到传入的路径
    ]
    
    final_image_model_path = None
    for path in contrastive_image_paths:
        if path and os.path.exists(path):
            final_image_model_path = path
            print(f"✅ 找到对比学习图像编码器: {path}")
            break
    
    if not final_image_model_path:
        print(f"⚠️  未找到对比学习图像编码器，将使用随机初始化")
        final_image_model_path = None
    
    # 创建对比学习模型
    model = create_contrastive_model(
        image_model_path=final_image_model_path,
        text_model_path=None,  # 暂不使用文本预训练模型
        device=device,
        freeze_backbones=False  # 不冻结，允许端到端训练
    )
    
    # 加载真实的多模态数据（从Excel文件）
    print("📊 加载真实多模态数据（从Excel文件）...")
    try:
        image_data, texts, labels = create_multimodal_dataset_from_excel(
            image_data_dir=data_path,
            text_data_dir="./文本编码器",
            max_samples=100  # 限制样本数用于测试
        )
    except Exception as e:
        print(f"❌ 真实文本数据加载失败: {e}")
        print("🔧 回退到虚拟文本数据...")
        
        # 回退方案：使用虚拟文本数据
        from data_utils import load_early_fusion_data
        image_data, labels = load_early_fusion_data(data_path, max_samples=100)
        
        # 创建虚拟文本数据
        print("📝 创建虚拟文本数据...")
        batch_size = 4
        seq_length = 128
        vocab_size = 30522  # BERT词汇表大小
        
        num_samples = len(labels)
        input_ids = torch.randint(0, vocab_size, (num_samples, seq_length))
        attention_mask = torch.ones(num_samples, seq_length)
        
        # 创建数据集
        image_tensor = torch.FloatTensor(image_data)
        label_tensor = torch.LongTensor(labels)
        
        dataset = TensorDataset(image_tensor, input_ids, attention_mask, label_tensor)
        texts = None  # 标记为虚拟数据
    
    # 如果使用真实文本数据，需要进行文本编码
    if texts is not None:
        print("📝 编码真实文本数据...")
        
        # 获取文本编码器进行编码
        text_encoder = model.text_encoder
        
        # 批量编码文本
        batch_size = 8  # 文本编码批次大小
        all_input_ids = []
        all_attention_masks = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            input_ids, attention_mask = text_encoder.encode_text(batch_texts, max_length=512)
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
        
        # 合并所有批次
        input_ids = torch.cat(all_input_ids, dim=0)
        attention_mask = torch.cat(all_attention_masks, dim=0)
        
        print(f"✅ 文本编码完成: {input_ids.shape}")
        
        # 创建数据集
        image_tensor = torch.FloatTensor(image_data)
        label_tensor = torch.LongTensor(labels)
        
        dataset = TensorDataset(image_tensor, input_ids, attention_mask, label_tensor)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    batch_size = 4  # 训练批次大小
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"📈 训练集: {len(train_dataset)} 样本")
    print(f"📈 验证集: {len(val_dataset)} 样本")
    if texts is not None:
        print(f"📝 使用真实文本数据训练")
    else:
        print(f"📝 使用虚拟文本数据训练")
    
    # 简单训练循环
    import torch.optim as optim
    from tqdm import tqdm
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    num_epochs = 5  # 简短训练用于测试
    best_val_acc = 0.0
    
    print(f"🚀 开始对比学习训练 ({num_epochs} 轮)...")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, input_ids, attention_mask, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # 对比学习 + 分类
            contrastive_output = model(images, input_ids, attention_mask, mode='contrastive')
            classification_output = model(images, input_ids, attention_mask, mode='classification')
            
            # 组合损失
            contrastive_loss = contrastive_output['contrastive_loss']
            classification_loss = criterion(classification_output['logits'], labels)
            
            total_loss = 0.5 * contrastive_loss + 1.0 * classification_loss
            
            total_loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += total_loss.item()
            _, predicted = classification_output['logits'].max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, input_ids, attention_mask, labels in val_loader:
                images = images.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                output = model(images, input_ids, attention_mask, mode='classification')
                _, predicted = output['logits'].max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  训练准确率: {train_acc:.2f}%")
        print(f"  验证准确率: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'using_real_text': texts is not None,
            }, './models/best_contrastive_model.pth')
            print(f"  ✅ 保存最佳模型: {val_acc:.2f}%")
    
    print(f"\n✅ 对比学习训练完成!")
    print(f"   最佳验证准确率: {best_val_acc:.2f}%")
    if texts is not None:
        print(f"   ✅ 使用了真实文本数据")
    else:
        print(f"   ⚠️  使用了虚拟文本数据")
    
    return best_val_acc

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='智能下采样 + 对比学习训练')
    parser.add_argument('--skip-downsample', action='store_true', 
                       help='跳过智能下采样训练，直接进行对比学习')
    parser.add_argument('--train-contrastive-encoder', action='store_true',
                       help='训练对比学习专用图像编码器')
    parser.add_argument('--data-path', type=str, default=None,
                       help='数据路径')
    
    args = parser.parse_args()
    
    print("🚀 智能下采样 + 对比学习训练启动")
    print("=" * 60)
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查环境
    env_check = check_environment()
    if isinstance(env_check, tuple):
        success, data_path = env_check
        if not success:
            print("❌ 环境检查失败")
            return
    else:
        print("❌ 环境检查失败")
        return
    
    # 使用用户指定的数据路径
    if args.data_path:
        data_path = args.data_path
        print(f"📁 使用指定数据路径: {data_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 🔥 新增选项：训练对比学习专用图像编码器
    if args.train_contrastive_encoder:
        print("\n🔥 训练对比学习专用图像编码器")
        print("=" * 50)
        
        try:
            from train_image_encoder_for_contrastive import ContrastiveImageEncoderTrainer
            
            trainer = ContrastiveImageEncoderTrainer(device=device)
            
            # 使用标准配置训练
            model, best_acc, model_path = trainer.train(
                data_path=data_path,
                base_channels=12,
                num_epochs=50,
                batch_size=4,
                learning_rate=1e-4,
                max_samples=None,
                patience=15
            )
            
            print(f"\n✅ 对比学习图像编码器训练完成!")
            print(f"   最佳准确率: {best_acc:.2f}%")
            print(f"   模型路径: {model_path}")
            
            # 保存训练曲线
            trainer.save_training_plots(12)
            
            # 继续进行对比学习训练
            image_model_path = model_path
            
        except Exception as e:
            print(f"❌ 对比学习图像编码器训练失败: {e}")
            print("🔧 将使用随机初始化进行对比学习训练")
            image_model_path = None
    else:
        # 步骤1: 训练智能下采样模型
        if not args.skip_downsample:
            try:
                image_model_path, downsample_acc = train_smart_downsample_model(data_path, device)
            except Exception as e:
                print(f"❌ 智能下采样训练失败: {e}")
                print("🔧 将使用随机初始化进行对比学习训练")
                image_model_path = None
                downsample_acc = 0.0
        else:
            print("⏭️  跳过智能下采样训练")
            image_model_path = './models/smart_downsample_spatial_ch12.pth'
            downsample_acc = 0.0
    
    # 步骤2: 训练对比学习模型
    try:
        contrastive_acc = train_contrastive_model(image_model_path, data_path, device)
    except Exception as e:
        print(f"❌ 对比学习训练失败: {e}")
        contrastive_acc = 0.0
    
    # 总结
    print("\n" + "=" * 60)
    print("🎉 训练完成总结")
    print("=" * 60)
    
    if args.train_contrastive_encoder:
        print(f"🔥 对比学习图像编码器: 已训练并保存到 ./models/contrastive/")
    elif not args.skip_downsample:
        print(f"🔥 智能下采样模型: {downsample_acc:.2f}%")
    
    print(f"🎯 对比学习模型: {contrastive_acc:.2f}%")
    
    print(f"\n📁 模型保存目录:")
    if args.train_contrastive_encoder:
        print(f"   对比学习图像编码器: ./models/contrastive/")
    print(f"   对比学习模型: ./models/")
    print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📝 后续步骤:")
    if args.train_contrastive_encoder:
        print("1. ✅ 对比学习图像编码器已训练完成")
        print("2. 🔗 可直接用于多模态对比学习")
    else:
        print("1. 🔥 可选择训练专用对比学习图像编码器:")
        print("   python run_training.py --train-contrastive-encoder")
    print("2. 等待队友提供文本预训练模型")
    print("3. 使用真实文本数据重新训练对比学习模型")
    print("4. 进行完整的多模态融合评估")

if __name__ == "__main__":
    main() 
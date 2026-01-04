import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import numpy as np
from torch.utils.data import DataLoader
from dataset import SimpleDataset
from data_utils import create_tissue_specific_dataset, create_data_loaders
from advanced_trainer import train_improved_resnet, train_fusion_with_improved_models
from optimized_models import create_improved_resnet3d, ImprovedResNetCBAM3D, EMAModel

# 设置随机种子以提高可重复性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='3D医学图像分类训练脚本')
    parser.add_argument('--data_path', type=str, default='/path/to/data',
                        help='数据集根目录')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'ensemble'],
                        help='运行模式: train(训练单个模型), test(测试模型), ensemble(训练融合模型)')
    parser.add_argument('--tissue_type', type=str, default='ALL', 
                        choices=['CSF', 'GREY', 'WHITE', 'ALL'],
                        help='组织类型，或ALL表示全部')
    parser.add_argument('--model_dir', type=str, default='./models',
                        help='模型保存路径')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批处理大小')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='训练设备')
    return parser.parse_args()

def prepare_data_loaders(args):
    """准备数据加载器"""
    print(f"\n===== 准备数据加载器 =====")
    
    tissue_types = ['CSF', 'GREY', 'WHITE'] if args.tissue_type == 'ALL' else [args.tissue_type]
    data_loaders = {}
    
    for tissue_type in tissue_types:
        print(f"\n加载 {tissue_type} 组织的数据...")
        
        # 创建数据集
        dataset = create_tissue_specific_dataset(args.data_path, tissue_type)
        
        # 创建训练和验证数据加载器
        train_loader, val_loader = create_data_loaders(
            dataset, 
            batch_size=args.batch_size,
            num_workers=4
        )
        
        # 保存数据加载器
        data_loaders[f'train_{tissue_type}'] = train_loader
        data_loaders[f'val_{tissue_type}'] = val_loader
        
        print(f"{tissue_type} 数据集统计:")
        print(f"训练样本数: {len(train_loader.dataset)}")
        print(f"验证样本数: {len(val_loader.dataset)}")
    
    return data_loaders, tissue_types

def train(args):
    """训练改进的ResNetCBAM3D模型"""
    print(f"\n===== 开始训练改进的ResNetCBAM3D模型 =====")
    
    # 准备数据
    data_loaders, tissue_types = prepare_data_loaders(args)
    
    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 训练模型
    if args.tissue_type == 'ALL':
        # 训练所有组织类型
        results = train_improved_resnet(
            data_loaders=data_loaders,
            device=device,
            save_dir=args.model_dir,
            tissue_types=tissue_types
        )
        
        # 打印结果
        print("\n===== 训练结果 =====")
        for tissue_type, result in results.items():
            print(f"{tissue_type}: 最佳验证准确率 = {result['best_val_acc']:.2f}%, "
                  f"最佳轮次 = {result['best_epoch']}")
    else:
        # 训练单个组织类型的模型
        data_loaders_single = {
            f'train_{args.tissue_type}': data_loaders[f'train_{args.tissue_type}'],
            f'val_{args.tissue_type}': data_loaders[f'val_{args.tissue_type}']
        }
        
        results = train_improved_resnet(
            data_loaders=data_loaders_single,
            device=device,
            save_dir=args.model_dir,
            tissue_types=[args.tissue_type]
        )
        
        # 打印结果
        print(f"\n训练结果 - {args.tissue_type}: 最佳验证准确率 = {results[args.tissue_type]['best_val_acc']:.2f}%, "
              f"最佳轮次 = {results[args.tissue_type]['best_epoch']}")

def ensemble(args):
    """训练融合模型"""
    print(f"\n===== 开始训练融合模型 =====")
    
    # 准备数据
    data_loaders, tissue_types = prepare_data_loaders(args)
    
    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 查找预训练模型路径
    model_paths = {}
    for tissue_type in tissue_types:
        model_path = f"{args.model_dir}/best_improved_resnet_{tissue_type}.pth"
        if os.path.exists(model_path):
            model_paths[tissue_type] = model_path
            print(f"找到 {tissue_type} 预训练模型: {model_path}")
        else:
            print(f"警告: 未找到 {tissue_type} 预训练模型")
    
    if not model_paths:
        print("错误: 未找到任何预训练模型。请先运行训练模式。")
        return
    
    # 训练融合模型
    fusion_model, best_val_acc = train_fusion_with_improved_models(
        data_loaders=data_loaders,
        device=device,
        model_paths=model_paths,
        tissue_types=tissue_types,
        save_dir=args.model_dir
    )
    
    print(f"\n融合模型训练完成，最佳验证准确率: {best_val_acc:.2f}%")

def test(args):
    """测试模型性能"""
    print(f"\n===== 开始测试模型 =====")
    
    # 准备数据
    data_loaders, tissue_types = prepare_data_loaders(args)
    
    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    if args.tissue_type == 'ALL':
        # 测试所有组织类型的模型和融合模型
        results = {}
        
        # 测试单个模型
        for tissue_type in tissue_types:
            model_path = f"{args.model_dir}/best_improved_resnet_{tissue_type}.pth"
            if os.path.exists(model_path):
                # 加载模型
                model = create_improved_resnet3d(in_channels=1, num_classes=2, device=device)
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # 测试模型
                val_acc = test_model(model, data_loaders[f'val_{tissue_type}'], device)
                
                # 保存结果
                results[tissue_type] = val_acc
                print(f"{tissue_type} 模型验证准确率: {val_acc:.2f}%")
        
        # 测试融合模型
        fusion_model_path = f"{args.model_dir}/best_improved_fusion.pth"
        if os.path.exists(fusion_model_path):
            print("\n测试融合模型...")
            
            # 创建基础模型
            base_models = {}
            for tissue_type in tissue_types:
                model_path = f"{args.model_dir}/best_improved_resnet_{tissue_type}.pth"
                if os.path.exists(model_path):
                    model = create_improved_resnet3d(in_channels=1, num_classes=2, device=device)
                    checkpoint = torch.load(model_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.eval()
                    base_models[tissue_type] = model
            
            # 测试融合结果
            if len(base_models) == len(tissue_types):
                # 使用验证数据测试融合性能
                fusion_acc = test_fusion(base_models, data_loaders, tissue_types, device)
                results['fusion'] = fusion_acc
                print(f"融合模型验证准确率: {fusion_acc:.2f}%")
        
        # 打印所有结果
        print("\n===== 测试结果汇总 =====")
        for model_name, acc in results.items():
            print(f"{model_name}: {acc:.2f}%")
    else:
        # 测试单个组织类型的模型
        model_path = f"{args.model_dir}/best_improved_resnet_{args.tissue_type}.pth"
        if os.path.exists(model_path):
            # 加载模型
            model = create_improved_resnet3d(in_channels=1, num_classes=2, device=device)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 测试模型
            val_acc = test_model(model, data_loaders[f'val_{args.tissue_type}'], device)
            print(f"{args.tissue_type} 模型验证准确率: {val_acc:.2f}%")
        else:
            print(f"错误: 未找到 {args.tissue_type} 预训练模型")

def test_model(model, data_loader, device):
    """测试单个模型的性能"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算准确率
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    # 计算总准确率
    accuracy = 100. * correct / total
    
    return accuracy

def test_fusion(base_models, data_loaders, tissue_types, device):
    """测试模型融合性能"""
    # 初始化统计
    correct = 0
    total = 0
    
    # 获取验证数据迭代器
    val_iterators = {t: iter(data_loaders[f'val_{t}']) for t in tissue_types}
    
    # 确保所有验证数据加载器的长度一致
    min_val_len = min(len(data_loaders[f'val_{t}']) for t in tissue_types)
    
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
                
                # 获取各模型的预测
                predictions = []
                for tissue_type, model in base_models.items():
                    if tissue_type in inputs_dict:
                        # 获取模型输出
                        output = model(inputs_dict[tissue_type])
                        predictions.append(output)
                
                # 简单平均融合
                ensemble_output = torch.zeros_like(predictions[0])
                for pred in predictions:
                    ensemble_output += pred
                ensemble_output /= len(predictions)
                
                # 计算准确率
                _, predicted = ensemble_output.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
            except StopIteration:
                break
    
    # 计算总准确率
    accuracy = 100. * correct / total if total > 0 else 0
    
    return accuracy

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(42)
    
    # 根据模式执行操作
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'ensemble':
        ensemble(args)
    else:
        print(f"错误: 不支持的模式 '{args.mode}'")

if __name__ == "__main__":
    main() 
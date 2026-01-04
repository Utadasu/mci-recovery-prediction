import random
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from scipy.ndimage import rotate

class SimpleDataset(Dataset):
    def __init__(self, data_path, target_size=(113, 137, 113)):
        self.data_path = data_path
        self.target_size = target_size
        self.samples = []
        self.labels = []
        self.patient_ids = []
        
        # 用于统计患者ID信息
        self.patient_stats = {
            'ad': {'total': 0, 'ids': set(), 'tissue_counts': {'CSF': 0, 'GRAY': 0, 'WHITE': 0}},
            'cn': {'total': 0, 'ids': set(), 'tissue_counts': {'CSF': 0, 'GRAY': 0, 'WHITE': 0}}
        }
        
        print(f"\n检查数据路径:")
        print(f"AD目录: {data_path['ad_dir']}")
        print(f"CN目录: {data_path['cn_dir']}")
        
        # 处理AD数据（标签0）
        ad_dir = data_path['ad_dir']
        print(f"\n检查AD目录:")
        print(f"AD主目录存在: {os.path.exists(ad_dir)}")
        
        if os.path.exists(ad_dir):
            # 处理AD图像文件
            print(f"\n处理AD图像文件:")
            for subdir in ["ADfinalCSF", "ADfinalGRAY", "ADfinalWHITE"]:
                subdir_path = os.path.join(os.path.dirname(ad_dir), subdir)
                print(f"\n检查{subdir}:")
                print(f"目录存在: {os.path.exists(subdir_path)}")
                if os.path.exists(subdir_path):
                    print(f"{subdir}中的文件:")
                    for img_name in os.listdir(subdir_path):
                        if img_name.endswith('.nii'):
                            parts = img_name.split('_')
                            if len(parts) >= 5:
                                patient_id = parts[3]
                                tissue_type = 'CSF' if 'CSF' in subdir else 'GRAY' if 'GRAY' in subdir else 'WHITE'
                                self.patient_stats['ad']['total'] += 1
                                self.patient_stats['ad']['tissue_counts'][tissue_type] += 1
                                self.samples.append(os.path.join(subdir_path, img_name))
                                self.labels.append(0)
                                self.patient_ids.append(patient_id)
                                self.patient_stats['ad']['ids'].add(patient_id)
                                print(f"  - {img_name} (患者ID: {patient_id})")
                            else:
                                print(f"  - {img_name} - 文件名格式无效")
        
        # 处理CN数据（标签1）
        cn_dir = data_path['cn_dir']
        print(f"\n检查CN目录:")
        print(f"CN主目录存在: {os.path.exists(cn_dir)}")
        
        if os.path.exists(cn_dir):
            # 处理CN图像文件
            print(f"\n处理CN图像文件:")
            for subdir in ["CNfinalCSF", "CNfinalGRAY", "CNfinalWHITE"]:
                subdir_path = os.path.join(os.path.dirname(cn_dir), subdir)
                print(f"\n检查{subdir}:")
                print(f"目录存在: {os.path.exists(subdir_path)}")
                if os.path.exists(subdir_path):
                    print(f"{subdir}中的文件:")
                    for img_name in os.listdir(subdir_path):
                        if img_name.endswith('.nii'):
                            parts = img_name.split('_')
                            if len(parts) >= 5:
                                patient_id = parts[3]
                                tissue_type = 'CSF' if 'CSF' in subdir else 'GRAY' if 'GRAY' in subdir else 'WHITE'
                                self.patient_stats['cn']['total'] += 1
                                self.patient_stats['cn']['tissue_counts'][tissue_type] += 1
                                self.samples.append(os.path.join(subdir_path, img_name))
                                self.labels.append(1)
                                self.patient_ids.append(patient_id)
                                self.patient_stats['cn']['ids'].add(patient_id)
                                print(f"  - {img_name} (患者ID: {patient_id})")
                            else:
                                print(f"  - {img_name} - 文件名格式无效")
        
        # 打印详细的统计信息
        print("\n=== 详细的数据分布统计 ===")
        print("\nAD患者统计:")
        print(f"唯一患者总数: {len(self.patient_stats['ad']['ids'])}")
        print(f"总图像数: {self.patient_stats['ad']['total']}")
        print("各组织类型图像数:")
        for tissue, count in self.patient_stats['ad']['tissue_counts'].items():
            print(f"  {tissue}: {count}")
        
        print("\nCN患者统计:")
        print(f"唯一患者总数: {len(self.patient_stats['cn']['ids'])}")
        print(f"总图像数: {self.patient_stats['cn']['total']}")
        print("各组织类型图像数:")
        for tissue, count in self.patient_stats['cn']['tissue_counts'].items():
            print(f"  {tissue}: {count}")
        
        # 检查每个患者的模态完整性
        print("\n检查每个患者的模态完整性:")
        for patient_id in self.patient_stats['ad']['ids']:
            modalities = [pid for pid in self.patient_ids if pid == patient_id]
            print(f"AD患者 {patient_id}: {len(modalities)} 个模态")
        
        for patient_id in self.patient_stats['cn']['ids']:
            modalities = [pid for pid in self.patient_ids if pid == patient_id]
            print(f"CN患者 {patient_id}: {len(modalities)} 个模态")
        
        print(f"\n加载的总样本数: {len(self.samples)}")
        if len(self.samples) == 0:
            raise ValueError("未找到有效样本！请检查数据路径和文件命名格式。")
        
        # 打印目标图像大小
        print(f"\n目标图像大小: {self.target_size}")
        print(f"每个体素总数: {np.prod(self.target_size)}")
    
    def __len__(self):
        return len(self.samples)
    
    def resize_3d(self, img):
        """将3D图像调整为目标大小"""
        zoom_factors = [t/s for t, s in zip(self.target_size, img.shape)]
        from scipy.ndimage import zoom
        return zoom(img, zoom_factors, order=1)
    
    def normalize_image(self, img):
        """改进的图像归一化，增加稳定性"""
        # 移除异常值 - 使用更保守的百分位数
        percentile_99 = np.percentile(img, 99.5)
        percentile_1 = np.percentile(img, 0.5)
        img = np.clip(img, percentile_1, percentile_99)
        
        # Z-score归一化
        mean = np.mean(img)
        std = np.std(img)
        if std > 0:
            img = (img - mean) / (std + 1e-8)  # 添加小量防止除零
        else:
            img = img - mean
        
        # 确保值在合理范围内
        img = np.clip(img, -10, 10)  # 限制极端值
        
        return img
    
    def random_gamma(self, img, gamma_range=(0.8, 1.2)):
        """随机伽马校正"""
        gamma = np.random.uniform(gamma_range[0], gamma_range[1])
        # 处理负值：保留符号，对绝对值应用伽马
        img_signed = np.sign(img)  # 获取符号 (-1, 0, 1)
        img_abs = np.abs(img) + 1e-8  # 获取绝对值并添加偏移
        return img_signed * np.power(img_abs, gamma)  # 保留原始符号
    
    def __getitem__(self, idx):
        try:
            # 加载图像
            img = nib.load(self.samples[idx]).get_fdata()
            
            # 检查原始图像大小
            original_shape = img.shape
            
            # 数据增强 - 减少增强强度，增加确定性
            if random.random() > 0.7:  # 降低增强概率
                # 随机旋转 - 减小角度范围
                angle = random.uniform(-10, 10)
                img = rotate(img, angle, axes=(0, 1), reshape=False)
            
            if random.random() > 0.7:  # 降低增强概率
                # 随机翻转
                img = np.flip(img, axis=0)
            
            if random.random() > 0.7:  # 降低增强概率
                # 随机翻转
                img = np.flip(img, axis=1)
            
            # 添加更多数据增强，但降低强度
            if random.random() > 0.7:  # 降低增强概率
                # 随机亮度 - 减小范围
                brightness_factor = random.uniform(0.9, 1.1)
                img = img * brightness_factor
            
            if random.random() > 0.7:  # 降低增强概率
                # 随机对比度 - 减小范围
                contrast_factor = random.uniform(0.9, 1.1)
                mean = np.mean(img)
                img = (img - mean) * contrast_factor + mean
            
            if random.random() > 0.8:  # 进一步降低噪声概率
                # 随机噪声 - 减小噪声强度
                noise_level = random.uniform(0.005, 0.02)
                noise = np.random.normal(0, noise_level, img.shape)
                img = img + noise
            
            if random.random() > 0.8:  # 降低裁剪概率
                # 随机裁剪和填充 - 减小裁剪范围
                crop_size = int(img.shape[0] * 0.95)  # 减小裁剪比例
                start = random.randint(0, img.shape[0] - crop_size)
                img = img[start:start+crop_size, start:start+crop_size, start:start+crop_size]
            
            # 归一化
            img = self.normalize_image(img)
            
            # 只有在图像形状与目标不一致时才调整大小
            if img.shape != self.target_size:
                # 记录调整大小操作
                if idx % 100 == 0:  # 仅记录部分样本以减少日志量
                    print(f"调整样本 {idx} 大小: {img.shape} -> {self.target_size}")
                img = self.resize_3d(img)
            
            # 添加通道维度
            img = img[np.newaxis, ...]
            
            # 转换为tensor并添加随机噪声 - 降低噪声强度
            img = torch.FloatTensor(img)
            if random.random() > 0.8:  # 降低噪声概率
                noise = torch.randn_like(img) * 0.005  # 减小噪声强度
                img = img + noise
            
            return img, self.labels[idx]
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {str(e)}")
            print(f"图像路径: {self.samples[idx]}")
            raise
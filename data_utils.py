import os
from torch.utils.data import DataLoader, Subset, ConcatDataset
from dataset import SimpleDataset
import random
import numpy as np
import nibabel as nib
from scipy.ndimage import rotate
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import logging
from typing import List, Tuple, Dict, Optional, Union
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, map_coordinates
import pandas as pd
import re
from tqdm import tqdm # Added for progress bar

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_default_data_path():
    """
    智能获取默认数据路径，兼容服务器和本地环境。
    遵循MCI_DATA数据规范，使用正确的根目录路径。
    """
    # 优先检查服务器路径 - 遵循MCI_DATA规范
    server_paths = [
        "/root/autodl-tmp/MCI_DATA/", # V3.5 更新: 指向规范的MCI_DATA根目录
        "/root/autodl-tmp/DATA_MCI/", # 兼容旧路径
        "/autodl-fs/data/ZM_Files/备份5.27/test_data/",
        "/autodl-fs/data/test_data/"
    ]
    for path in server_paths:
        if os.path.exists(path):
            logger.info(f"检测到服务器数据路径: {path}")
            return path
    
    # 检查本地调试路径
    local_paths = [
        "./test_data/",
        "../test_data/",
        "../../test_data/"
    ]
    for path in local_paths:
        if os.path.exists(os.path.abspath(path)):
            abs_path = os.path.abspath(path)
            logger.info(f"检测到本地数据路径: {abs_path}")
            return abs_path
            
    logger.warning("未找到任何预设的数据路径，请手动指定。将返回第一个服务器路径作为默认值。")
    return server_paths[0]

def create_tissue_specific_dataset(data_path, tissue_type):
    """Create dataset for specific tissue type"""
    # Create a copy of data_path with modified paths
    tissue_data_path = data_path.copy()
    
    # 直接构建到具体子目录的路径
    if tissue_type == 'CSF':
        tissue_data_path['ad_dir'] = os.path.join(data_path['ad_dir'], 'ADfinalCSF')
        tissue_data_path['cn_dir'] = os.path.join(data_path['cn_dir'], 'CNfinalCSF')
    elif tissue_type == 'GRAY':
        tissue_data_path['ad_dir'] = os.path.join(data_path['ad_dir'], 'ADfinalGRAY')
        tissue_data_path['cn_dir'] = os.path.join(data_path['cn_dir'], 'CNfinalGRAY')
    elif tissue_type == 'WHITE':
        tissue_data_path['ad_dir'] = os.path.join(data_path['ad_dir'], 'ADfinalWHITE')
        tissue_data_path['cn_dir'] = os.path.join(data_path['cn_dir'], 'CNfinalWHITE')
    else:
        raise ValueError(f"Unknown tissue type: {tissue_type}")
    
    # Verify paths exist
    print(f"\nVerifying paths for {tissue_type}:")
    print(f"AD path: {tissue_data_path['ad_dir']}")
    print(f"CN path: {tissue_data_path['cn_dir']}")
    
    # 检查路径是否存在
    if not os.path.exists(tissue_data_path['ad_dir']):
        # 尝试在当前目录查找
        if os.path.exists(data_path['ad_dir']):
            print(f"AD directory {tissue_data_path['ad_dir']} does not exist.")
            print(f"Available AD directories:")
            for item in os.listdir(data_path['ad_dir']):
                print(f"  - {item}")
        raise ValueError(f"AD directory does not exist: {tissue_data_path['ad_dir']}")
    
    if not os.path.exists(tissue_data_path['cn_dir']):
        # 尝试在当前目录查找
        if os.path.exists(data_path['cn_dir']):
            print(f"CN directory {tissue_data_path['cn_dir']} does not exist.")
            print(f"Available CN directories:")
            for item in os.listdir(data_path['cn_dir']):
                print(f"  - {item}")
        raise ValueError(f"CN directory does not exist: {tissue_data_path['cn_dir']}")
    
    # 只打印目录存在信息，不显示具体内容
    print(f"\nAD目录包含文件数: {len([f for f in os.listdir(tissue_data_path['ad_dir']) if f.endswith('.nii')])}")
    print(f"CN目录包含文件数: {len([f for f in os.listdir(tissue_data_path['cn_dir']) if f.endswith('.nii')])}")
    
    return SimpleDataset(tissue_data_path)

def create_data_loaders(dataset, batch_size=32, num_workers=8):
    """Create train and validation data loaders with patient-wise split"""
    # 获取所有唯一的患者ID
    patient_ids = list(set(dataset.patient_ids))
    random.shuffle(patient_ids)
    
    # 按患者ID划分训练集和验证集
    split_idx = int(len(patient_ids) * 0.8)
    train_patients = patient_ids[:split_idx]
    val_patients = patient_ids[split_idx:]
    
    # 创建训练集和验证集的索引
    train_indices = []
    val_indices = []
    
    for idx, patient_id in enumerate(dataset.patient_ids):
        if patient_id in train_patients:
            train_indices.append(idx)
        else:
            val_indices.append(idx)
    
    # 创建数据集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"\n数据集划分信息:")
    print(f"训练集患者数: {len(train_patients)}")
    print(f"验证集患者数: {len(val_patients)}")
    print(f"训练集样本数: {len(train_indices)}")
    print(f"验证集样本数: {len(val_indices)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

# 数据增强转换
class RandomRotation3D:
    def __init__(self, degrees=10):
        self.degrees = degrees

    def __call__(self, x):
        angle = random.uniform(-self.degrees, self.degrees)
        return F.rotate(x, angle, mode='bilinear')

class RandomFlip3D:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            x = torch.flip(x, [2])  # 水平翻转
        if random.random() < self.p:
            x = torch.flip(x, [3])  # 垂直翻转
        return x

class RandomBrightnessContrast:
    def __init__(self, brightness=0.2, contrast=0.2):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, x):
        brightness_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        contrast_factor = 1.0 + random.uniform(-self.contrast, self.contrast)
        x = x * brightness_factor
        x = (x - x.mean()) * contrast_factor + x.mean()
        return x

class GammaCorrection:
    def __init__(self, gamma_range=(0.8, 1.2)):
        self.gamma_range = gamma_range

    def __call__(self, x):
        gamma = random.uniform(*self.gamma_range)
        x = torch.pow(x, gamma)
        return x

# 增强型数据增强方法
class ElasticDeformation:
    """弹性变形增强方法，对3D医学图像特别有效"""
    def __init__(self, alpha=1, sigma=0.1, apply_prob=0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.apply_prob = apply_prob
        
    def __call__(self, img):
        if random.random() > self.apply_prob:
            return img
            
        # 确保我们正确处理包含通道维度的图像形状
        # img的形状应该是 (C, D, H, W) 或 (D, H, W)
        input_shape = img.shape
        
        # 判断输入是3D还是4D（带通道）
        has_channel_dim = len(input_shape) == 4
        
        if has_channel_dim:
            # 如果有通道维度，我们将其分离出来单独处理
            C, D, H, W = input_shape
            img_no_channel = img[0]  # 假设只有一个通道
        else:
            # 直接使用3D体数据
            D, H, W = input_shape
            img_no_channel = img
            
        # 为3D图像创建形变矢量场
        shape = (D, H, W)
        
        # 创建随机位移场并使用高斯滤波平滑它们
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        
        # 创建网格坐标
        z, y, x = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
        
        # 应用位移场，并扁平化坐标以用于map_coordinates
        indices = [
            np.reshape(z + dz, (-1,)), 
            np.reshape(y + dy, (-1,)), 
            np.reshape(x + dx, (-1,))
        ]
        
        # 应用形变
        distorted_img = map_coordinates(img_no_channel, indices, order=1, mode='reflect')
        
        # 恢复原始形状
        distorted_img = distorted_img.reshape(shape)
        
        # 如果原始输入有通道维度，则添加回去
        if has_channel_dim:
            distorted_img = distorted_img[np.newaxis, ...]
            
        return distorted_img

class RandomIntensityShift:
    """随机强度偏移增强"""
    def __init__(self, shift_range=0.1, apply_prob=0.3):
        self.shift_range = shift_range
        self.apply_prob = apply_prob
        
    def __call__(self, img):
        if random.random() > self.apply_prob:
            return img
            
        shift = random.uniform(-self.shift_range, self.shift_range)
        return img + shift

class RandomIntensityScale:
    """随机强度缩放增强"""
    def __init__(self, scale_range=(0.9, 1.1), apply_prob=0.3):
        self.scale_range = scale_range
        self.apply_prob = apply_prob
        
    def __call__(self, img):
        if random.random() > self.apply_prob:
            return img
            
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        return img * scale

# 数据集类
class SimpleDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.samples = []  # 文件路径
        self.labels = []   # 标签（AD=0，CN=1）
        self.patient_ids = []  # 患者ID
        self.modality_info = []  # 模态信息
        
        # 用于统计患者ID信息
        self.patient_stats = {
            'ad': {'total': 0, 'ids': set()},
            'cn': {'total': 0, 'ids': set()}
        }
        
        # 确定当前数据集的模态类型
        ad_dir = data_path['ad_dir']
        if 'CSF' in ad_dir:
            self.modality = 'CSF'
        elif 'GRAY' in ad_dir:
            self.modality = 'GRAY'
        elif 'WHITE' in ad_dir:
            self.modality = 'WHITE'
        else:
            self.modality = 'UNKNOWN'
        
        print(f"\n加载 {self.modality} 模态数据:")
        print(f"AD目录: {data_path['ad_dir']}")
        print(f"CN目录: {data_path['cn_dir']}")
        print(f"AD目录存在: {os.path.exists(data_path['ad_dir'])}")
        print(f"CN目录存在: {os.path.exists(data_path['cn_dir'])}")
        
        # 处理AD数据（标签0）
        ad_dir = data_path['ad_dir']
        if os.path.exists(ad_dir):
            print(f"\n处理AD {self.modality}图像文件:")
            file_count = 0
            for img_name in os.listdir(ad_dir):
                if img_name.endswith('.nii'):
                    file_count += 1
                    # 提取患者ID，规范化为一致的ID格式
                    patient_id = self._extract_patient_id(img_name)
                    
                    self.patient_stats['ad']['total'] += 1
                    self.samples.append(os.path.join(ad_dir, img_name))
                    self.labels.append(0)  # AD标签为0
                    self.patient_ids.append(patient_id)
                    self.modality_info.append(self.modality)
                    self.patient_stats['ad']['ids'].add(patient_id)
        
        # 处理CN数据（标签1）
        cn_dir = data_path['cn_dir']
        if os.path.exists(cn_dir):
            print(f"\n处理CN {self.modality}图像文件:")
            file_count = 0
            for img_name in os.listdir(cn_dir):
                if img_name.endswith('.nii'):
                    file_count += 1
                    # 提取患者ID，规范化为一致的ID格式
                    patient_id = self._extract_patient_id(img_name)
                    
                    self.patient_stats['cn']['total'] += 1
                    self.samples.append(os.path.join(cn_dir, img_name))
                    self.labels.append(1)  # CN标签为1
                    self.patient_ids.append(patient_id)
                    self.modality_info.append(self.modality)
                    self.patient_stats['cn']['ids'].add(patient_id)
        
        # 打印患者ID统计信息
        print(f"\n{self.modality}模态患者ID统计信息:")
        print("AD患者:")
        print(f"  唯一患者总数: {len(self.patient_stats['ad']['ids'])}")
        print(f"  总图像数: {self.patient_stats['ad']['total']}")
        print("CN患者:")
        print(f"  唯一患者总数: {len(self.patient_stats['cn']['ids'])}")
        print(f"  总图像数: {self.patient_stats['cn']['total']}")
        
        print(f"\n{self.modality}模态加载的总样本数: {len(self.samples)}")
        if len(self.samples) == 0:
            raise ValueError(f"未找到{self.modality}模态的有效样本！请检查数据路径和文件命名格式。")
    
    def _extract_patient_id(self, filename):
        """从文件名中提取规范化的患者ID - 修复NAME列对齐"""
        # 移除后缀
        basename = filename.split('.')[0]
        
        # 针对ADNI数据格式: "029_S_4385_3-2016-01-29_12_25_03.0.nii"或"mwp1MRI_002_S_0295_2006-04-18_08_51_20.0.nii"
        # 提取NAME部分: "029_S_4385"或"002_S_0295"
        if '_' in basename:
            parts = basename.split('_')
            # 寻找格式为 "数字_S_数字" 的部分
            for i in range(len(parts) - 2):
                if parts[i+1] == 'S' and parts[i].isdigit() and parts[i+2].isdigit():
                    return f"{parts[i]}_{parts[i+1]}_{parts[i+2]}"
            
            # 针对mwp*MRI格式: "mwp1MRI_002_S_0295_2006-04-18_08_51_20.0"
            if len(parts) >= 4 and parts[1] == 'S':
                return f"{parts[0]}_{parts[1]}_{parts[2]}"
            
            # 通用格式：取前3个部分
            elif len(parts) >= 3:
                return f"{parts[0]}_{parts[1]}_{parts[2]}"
            else:
                return basename  # 如果不符合预期格式，使用全名
        
        # 如果没有下划线，只保留数字作为ID
        numeric_id = re.sub(r'[^0-9]', '', basename)
        if numeric_id:
            return numeric_id
        
        # 最后的备选是整个文件名（不含后缀）
        return basename
    
    def __len__(self):
        return len(self.samples)
    
    # 移除resize_3d方法，保持图像原始尺寸
    
    def normalize_image(self, img):
        """简单的图像归一化"""
        if img.max() > img.min():
            img = (img - img.mean()) / img.std()  # Z-score归一化
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
            
            # 数据增强
            if random.random() > 0.5:
                # 随机旋转
                angle = random.uniform(-10, 10)
                img = rotate(img, angle, axes=(0, 1), reshape=False)
            
            if random.random() > 0.5:
                # 随机翻转
                img = np.flip(img, axis=0)
            
            if random.random() > 0.5:
                # 随机翻转
                img = np.flip(img, axis=1)
                
            # 基础数据增强
            if random.random() > 0.5:
                # 随机亮度
                brightness_factor = random.uniform(0.8, 1.2)
                img = img * brightness_factor
            
            # 模态特定增强
            if self.modality in ['GRAY', 'WHITE']:
                if random.random() > 0.5:
                    # 增强对比度，扩大参数范围以更好地增强这些模态的特征
                    contrast_factor = random.uniform(0.75, 1.25)  # 对比度范围扩大
                    mean = np.mean(img)
                    img = (img - mean) * contrast_factor + mean
                    
                if random.random() > 0.7:  # 为这些模态额外应用锐化
                    # 简单的锐化操作 - 拉普拉斯锐化
                    from scipy.ndimage import laplace
                    edge = laplace(img)
                    img = img - 0.2 * edge  # 弱锐化，避免过度强化噪声
            elif self.modality == 'CSF':
                # 为CSF使用正常对比度参数
                if random.random() > 0.5:
                    contrast_factor = random.uniform(0.8, 1.2)
                    mean = np.mean(img)
                    img = (img - mean) * contrast_factor + mean
            
            # 归一化
            img = self.normalize_image(img)
            
            # 添加通道维度
            img = img[np.newaxis, ...]
            
            # 应用高级增强
            if random.random() > 0.5:
                if random.random() > 0.7:  # 30%概率应用弹性变形
                    elastic_transform = ElasticDeformation(alpha=1, sigma=0.1)
                    img = elastic_transform(img)
                
                if random.random() > 0.7:  # 30%概率应用强度偏移
                    intensity_shift = RandomIntensityShift(shift_range=0.1)
                    img = intensity_shift(img)
                    
                if random.random() > 0.7:  # 30%概率应用强度缩放
                    intensity_scale = RandomIntensityScale(scale_range=(0.9, 1.1))
                    img = intensity_scale(img)
            
            return torch.FloatTensor(img), self.labels[idx], self.patient_ids[idx], self.modality_info[idx]
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {str(e)}")
            print(f"图像路径: {self.samples[idx]}")
            raise 

@staticmethod
def load_early_fusion_data(data_dir: str, max_samples: int = None):
        """
        加载用于早期融合的三模态MRI数据 (CSF, GREY, WHITE)。
        V3.5 更新: 适配新的 totalAD/totalCN 目录结构。
    
        Args:
            data_dir: 包含 totalAD 和 totalCN 子目录的数据根目录。
            max_samples: 每个类别的最大样本数（用于测试）。
    
        Returns:
            Tuple[np.ndarray, np.ndarray]: (images, labels)
        """
        logger.info(f"开始从 {data_dir} 加载早期融合数据 (新结构)...")
        
        # 定义AD和CN的数据目录
        ad_dir = os.path.join(data_dir, 'totalAD')
        cn_dir = os.path.join(data_dir, 'totalCN')
        
        if not os.path.exists(ad_dir) or not os.path.exists(cn_dir):
            raise FileNotFoundError(f"在 {data_dir} 中未找到 totalAD 或 totalCN 目录。")
    
        def extract_patient_id_from_filename(filename):
            # 匹配 mwp1MRI_002_S_0295_2006-04-18_08_51_20.0.nii 格式
            match = re.search(r'mwp\dMRI_(\d{3}_S_\d{4})_', filename)
            if match:
                return match.group(1)
            logger.warning(f"无法从文件名 {filename} 提取标准患者ID。")
            return None
    
        def load_nii_files_with_patient_id(directory):
            patient_files = {}
            if not os.path.exists(directory):
                logger.warning(f"目录不存在: {directory}")
                return {}
            for filename in os.listdir(directory):
                if filename.endswith((".nii", ".nii.gz")):
                    patient_id = extract_patient_id_from_filename(filename)
                    if patient_id:
                        patient_files[patient_id] = os.path.join(directory, filename)
            return patient_files
    
        # 加载每个模态的文件路径 - 遵循MCI_DATA规范
        ad_csf_files = load_nii_files_with_patient_id(os.path.join(ad_dir, 'ADfinalCSF'))
        ad_gray_files = load_nii_files_with_patient_id(os.path.join(ad_dir, 'ADfinalGRAY'))
        ad_white_files = load_nii_files_with_patient_id(os.path.join(ad_dir, 'ADfinalWHITE'))

        cn_csf_files = load_nii_files_with_patient_id(os.path.join(cn_dir, 'CNfinalCSF'))
        cn_gray_files = load_nii_files_with_patient_id(os.path.join(cn_dir, 'CNfinalGRAY'))
        cn_white_files = load_nii_files_with_patient_id(os.path.join(cn_dir, 'CNfinalWHITE'))
        
        logger.info(f"AD 模态文件数量: CSF={len(ad_csf_files)}, GRAY={len(ad_gray_files)}, WHITE={len(ad_white_files)}")
        logger.info(f"CN 模态文件数量: CSF={len(cn_csf_files)}, GRAY={len(cn_gray_files)}, WHITE={len(cn_white_files)}")
    
        all_images = []
        all_labels = []
    
        # 处理AD数据
        ad_patient_ids = set(ad_csf_files.keys()) & set(ad_gray_files.keys()) & set(ad_white_files.keys())
        logger.info(f"找到 {len(ad_patient_ids)} 个三模态完整的AD患者。")
        
        if max_samples is not None:
            ad_patient_ids = list(ad_patient_ids)[:max_samples]
    
        for patient_id in tqdm(ad_patient_ids, desc="处理 AD 数据"):
            try:
                csf_img = nib.load(ad_csf_files[patient_id]).get_fdata().astype(np.float32)
                gray_img = nib.load(ad_gray_files[patient_id]).get_fdata().astype(np.float32)
                white_img = nib.load(ad_white_files[patient_id]).get_fdata().astype(np.float32)

                if csf_img.shape == gray_img.shape == white_img.shape:
                    stacked_img = np.stack([csf_img, gray_img, white_img], axis=0)
                    all_images.append(stacked_img)
                    all_labels.append(1)
                else:
                    logger.warning(f"患者 {patient_id} (AD) 的模态形状不匹配，已跳过。")
            except Exception as e:
                logger.error(f"处理患者 {patient_id} (AD) 时出错: {e}")
    
        # 处理CN数据
        cn_patient_ids = set(cn_csf_files.keys()) & set(cn_gray_files.keys()) & set(cn_white_files.keys())
        logger.info(f"找到 {len(cn_patient_ids)} 个三模态完整的CN患者。")
    
        if max_samples is not None:
            cn_patient_ids = list(cn_patient_ids)[:max_samples]
    
        for patient_id in tqdm(cn_patient_ids, desc="处理 CN 数据"):
            try:
                csf_img = nib.load(cn_csf_files[patient_id]).get_fdata().astype(np.float32)
                gray_img = nib.load(cn_gray_files[patient_id]).get_fdata().astype(np.float32)
                white_img = nib.load(cn_white_files[patient_id]).get_fdata().astype(np.float32)

                if csf_img.shape == gray_img.shape == white_img.shape:
                    stacked_img = np.stack([csf_img, gray_img, white_img], axis=0)
                    all_images.append(stacked_img)
                    all_labels.append(0)
                else:
                    logger.warning(f"患者 {patient_id} (CN) 的模态形状不匹配，已跳过。")
            except Exception as e:
                logger.error(f"处理患者 {patient_id} (CN) 时出错: {e}")
    
        if not all_images:
            logger.error("未能加载任何有效的图像数据。")
            return np.array([]), np.array([])
    
        logger.info(f"数据加载完成。总样本数: {len(all_images)} (AD: {np.sum(all_labels)}, CN: {len(all_labels) - np.sum(all_labels)})")
        
        return np.array(all_images), np.array(all_labels)

def load_text_data_from_excel(text_data_dir: str = "./文本编码器") -> Tuple[List[str], List[int]]:
    """
    从Excel文件加载文本数据，并生成结构化的临床文本。
    
    数据规模:
    - AD患者: 414个样本 (final_AD_updated.xlsx, 24KB)
    - CN对照: 414个样本 (final_CN_updated.xlsx, 31KB)  
    - 总计: 828个样本
    
    文本模板:
    Age: [XX] years
    Sex: [Male/Female]
    Education: [XX] years
    Neuropsychological Scores:
    Mini-Mental State Examination (MMSE): [XX/30]
    Clinical Dementia Rating - Sum of Boxes (CDR-SB): [XX]
    Diagnosis: [Alzheimer's Disease (AD)/Cognitively Normal (CN)]
    
    Args:
        text_data_dir: 文本数据目录，包含final_AD_updated.xlsx和final_CN_updated.xlsx
    
    Returns:
        texts: 文本列表 (828个)
        labels: 标签列表 (0=CN, 1=AD)
    """
    print("📝 从Excel文件加载真实文本数据...")
    print("📊 预期数据规模: AD=414, CN=414, 总计=828")
    
    ad_file = os.path.join(text_data_dir, "final_AD_updated.xlsx")
    cn_file = os.path.join(text_data_dir, "final_CN_updated.xlsx")
    
    # 检查文件是否存在
    if not os.path.exists(ad_file):
        raise FileNotFoundError(f"❌ AD文本数据文件不存在: {ad_file}")
    if not os.path.exists(cn_file):
        raise FileNotFoundError(f"❌ CN文本数据文件不存在: {cn_file}")
    
    # 检查文件大小
    ad_size = os.path.getsize(ad_file) / 1024  # KB
    cn_size = os.path.getsize(cn_file) / 1024  # KB
    print(f"📁 文件大小检查: AD={ad_size:.1f}KB, CN={cn_size:.1f}KB")
    
    texts = []
    labels = []
    
    def create_clinical_text(row, diagnosis):
        """根据数据行创建临床文本模板"""
        # 性别映射
        gender_map = {0: 'Male', 1: 'Female', '0': 'Male', '1': 'Female'}
        gender = gender_map.get(row.get('Gender', 0), 'Unknown')
        
        # 获取各项数据，处理缺失值
        age = row.get('Age', 'Unknown')
        education = row.get('Edu', 'Unknown')
        mmse = row.get('MMSE', 'Unknown')
        cdrsb = row.get('CDRSB', 'Unknown')
        
        # 构建临床文本模板
        clinical_text = f"""Age: {age} years
Sex: {gender}
Education: {education} years
Neuropsychological Scores:
Mini-Mental State Examination (MMSE): {mmse}/30
Clinical Dementia Rating - Sum of Boxes (CDR-SB): {cdrsb}
Diagnosis: {diagnosis}"""
        
        return clinical_text
    
    try:
        # 加载AD患者数据
        print(f"📊 加载AD患者数据: {ad_file}")
        ad_df = pd.read_excel(ad_file)
        print(f"   ✅ AD数据形状: {ad_df.shape} (预期: 414行)")
        print(f"   📋 AD数据列: {list(ad_df.columns)}")
        
        # 加载CN对照组数据
        print(f"📊 加载CN对照组数据: {cn_file}")
        cn_df = pd.read_excel(cn_file)
        print(f"   ✅ CN数据形状: {cn_df.shape} (预期: 414行)")
        print(f"   📋 CN数据列: {list(cn_df.columns)}")
        
        # 验证数据规模
        if ad_df.shape[0] != 414:
            print(f"⚠️  AD数据行数异常: 实际{ad_df.shape[0]}行, 预期414行")
        if cn_df.shape[0] != 414:
            print(f"⚠️  CN数据行数异常: 实际{cn_df.shape[0]}行, 预期414行")
        
        # 验证必要列是否存在
        required_columns = ['NAME', 'Gender', 'Age', 'Edu', 'MMSE', 'CDRSB']
        
        # 检查AD数据列
        missing_ad_cols = [col for col in required_columns if col not in ad_df.columns]
        if missing_ad_cols:
            print(f"⚠️  AD数据缺少列: {missing_ad_cols}")
        
        # 检查CN数据列（忽略wholecode列）
        missing_cn_cols = [col for col in required_columns if col not in cn_df.columns]
        if missing_cn_cols:
            print(f"⚠️  CN数据缺少列: {missing_cn_cols}")
        
        print(f"🔍 使用临床文本模板构建特征...")
        
        # 处理AD数据 (414个样本)
        ad_count = 0
        for idx, row in ad_df.iterrows():
            try:
                # 构建临床文本
                clinical_text = create_clinical_text(row, "Alzheimer's Disease (AD)")
                
                # 清理文本
                clinical_text = clean_text(clinical_text)
                if len(clinical_text.strip()) > 0:  # 确保不是空文本
                    texts.append(clinical_text)
                    labels.append(1)  # AD标签
                    ad_count += 1
                    
            except Exception as e:
                print(f"⚠️  处理AD样本 {idx} 时出错: {e}")
                continue
        
        # 处理CN数据 (414个样本)
        cn_count = 0
        for idx, row in cn_df.iterrows():
            try:
                # 构建临床文本
                clinical_text = create_clinical_text(row, "Cognitively Normal (CN)")
                
                # 清理文本
                clinical_text = clean_text(clinical_text)
                if len(clinical_text.strip()) > 0:  # 确保不是空文本
                    texts.append(clinical_text)
                    labels.append(0)  # CN标签
                    cn_count += 1
                    
            except Exception as e:
                print(f"⚠️  处理CN样本 {idx} 时出错: {e}")
                continue
        
        print(f"✅ 文本数据加载完成:")
        print(f"   📊 总样本数: {len(texts)} (预期: 828)")
        print(f"   🔥 AD样本: {ad_count} (预期: 414)")
        print(f"   🔵 CN样本: {cn_count} (预期: 414)")
        print(f"   📝 平均文本长度: {sum(len(text.split()) for text in texts) / len(texts):.1f} 词")
        
        # 显示文本样例
        if len(texts) > 0:
            print(f"\n📋 文本样例 (AD患者):")
            ad_sample = next((text for i, text in enumerate(texts) if labels[i] == 1), None)
            if ad_sample:
                print(f"   {ad_sample[:200]}...")
            
            print(f"\n📋 文本样例 (CN对照):")
            cn_sample = next((text for i, text in enumerate(texts) if labels[i] == 0), None)
            if cn_sample:
                print(f"   {cn_sample[:200]}...")
        
        # 数据质量检查
        if len(texts) < 800:
            print(f"⚠️  文本样本数量偏少: {len(texts)}/828")
        if abs(ad_count - cn_count) > 50:
            print(f"⚠️  类别不平衡: AD={ad_count}, CN={cn_count}")
        
        return texts, labels
        
    except Exception as e:
        print(f"❌ Excel文件加载失败: {e}")
        print(f"💡 请检查Excel文件格式和内容")
        raise


def clean_text(text: str) -> str:
    """
    清理文本数据
    
    Args:
        text: 原始文本
    
    Returns:
        清理后的文本
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text)
    
    # 移除特殊字符（保留基本标点）
    text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
    
    # 转换为小写
    text = text.lower().strip()
    
    return text


def create_multimodal_dataset_from_excel(image_data_dir: str, 
                                        text_data_dir: str = "./文本编码器",
                                        max_samples: int = None) -> Tuple:
    """
    从Excel文件创建多模态数据集
    
    Args:
        image_data_dir: 图像数据目录
        text_data_dir: 文本数据目录
        max_samples: 最大样本数限制
    
    Returns:
        (image_data, texts, labels)
    """
    print("🔄 创建多模态数据集（从Excel文件）...")
    
    # 加载图像数据
    print("📸 加载图像数据...")
    image_data, image_labels = load_early_fusion_data(image_data_dir, max_samples=max_samples)
    
    # 加载文本数据
    print("📝 加载文本数据...")
    texts, text_labels = load_text_data_from_excel(text_data_dir)
    
    # 数据对齐检查
    print("🔍 数据对齐检查...")
    print(f"   图像数据: {len(image_data)} 样本 (AD={sum(image_labels)}, CN={len(image_labels)-sum(image_labels)})")
    print(f"   文本数据: {len(texts)} 样本 (AD={sum(text_labels)}, CN={len(text_labels)-sum(text_labels)})")
    
    # 如果数据量不匹配，需要进行对齐
    if len(image_data) != len(texts):
        print("⚠️  图像和文本数据数量不匹配，进行数据对齐...")
        
        # 取较小的数据集大小
        min_samples = min(len(image_data), len(texts))
        
        # 按类别平衡采样
        ad_image_indices = [i for i, label in enumerate(image_labels) if label == 1]
        cn_image_indices = [i for i, label in enumerate(image_labels) if label == 0]
        
        ad_text_indices = [i for i, label in enumerate(text_labels) if label == 1]
        cn_text_indices = [i for i, label in enumerate(text_labels) if label == 0]
        
        # 计算每类的样本数
        samples_per_class = min_samples // 2
        
        # 随机采样
        import random
        random.seed(42)
        
        selected_ad_image = random.sample(ad_image_indices, min(samples_per_class, len(ad_image_indices)))
        selected_cn_image = random.sample(cn_image_indices, min(samples_per_class, len(cn_image_indices)))
        
        selected_ad_text = random.sample(ad_text_indices, min(samples_per_class, len(ad_text_indices)))
        selected_cn_text = random.sample(cn_text_indices, min(samples_per_class, len(cn_text_indices)))
        
        # 重新组织数据
        aligned_image_data = []
        aligned_texts = []
        aligned_labels = []
        
        # AD样本
        for i, (img_idx, text_idx) in enumerate(zip(selected_ad_image, selected_ad_text)):
            aligned_image_data.append(image_data[img_idx])
            aligned_texts.append(texts[text_idx])
            aligned_labels.append(1)
        
        # CN样本
        for i, (img_idx, text_idx) in enumerate(zip(selected_cn_image, selected_cn_text)):
            aligned_image_data.append(image_data[img_idx])
            aligned_texts.append(texts[text_idx])
            aligned_labels.append(0)
        
        image_data = np.array(aligned_image_data)
        texts = aligned_texts
        labels = aligned_labels
        
        print(f"✅ 数据对齐完成: {len(image_data)} 样本")
    else:
        labels = image_labels  # 假设标签一致
    
    print(f"📊 最终数据集:")
    print(f"   样本数: {len(image_data)}")
    print(f"   图像形状: {image_data.shape}")
    print(f"   文本数: {len(texts)}")
    print(f"   标签分布: AD={sum(labels)}, CN={len(labels)-sum(labels)}")
    
    return image_data, texts, labels 

def load_image_data_from_nii(data_dir: str, max_samples_per_class: int = None):
    """
    从NII文件加载图像数据，返回图像数据、标签和患者ID
    
    Args:
        data_dir: 数据目录路径，应包含123-AD-MRI和123-CN-MRI子目录
        max_samples_per_class: 每个类别的最大样本数量限制（用于调试）
    
    Returns:
        tuple: (image_data, labels, patient_ids)
            - image_data: numpy数组，形状为[N, 3, D, H, W]
            - labels: numpy数组，形状为[N]，0=CN，1=AD
            - patient_ids: 患者ID列表
    """
    import nibabel as nib
    from scipy.ndimage import zoom
    
    print(f"🔄 开始加载图像数据从: {data_dir}")
    
    # 构建数据路径 - 遵循MCI_DATA规范
    ad_csf_dir = os.path.join(data_dir, "totalAD", "ADfinalCSF")
    ad_gray_dir = os.path.join(data_dir, "totalAD", "ADfinalGRAY")
    ad_white_dir = os.path.join(data_dir, "totalAD", "ADfinalWHITE")
    
    cn_csf_dir = os.path.join(data_dir, "totalCN", "CNfinalCSF")
    cn_gray_dir = os.path.join(data_dir, "totalCN", "CNfinalGRAY")
    cn_white_dir = os.path.join(data_dir, "totalCN", "CNfinalWHITE")
    
    # 验证路径存在
    required_dirs = [ad_csf_dir, ad_gray_dir, ad_white_dir, cn_csf_dir, cn_gray_dir, cn_white_dir]
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"❌ 路径不存在: {dir_path}")
    
    print("✅ 所有路径验证通过")
    
    def extract_patient_id_from_filename(filename):
        """从图像文件名提取患者ID"""
        basename = filename.split('.')[0]
        parts = basename.split('_')
        
        if len(parts) >= 4 and parts[0].startswith('mwp') and parts[2] == 'S':
            return f"{parts[1]}_{parts[2]}_{parts[3]}"
        
        for i in range(len(parts) - 2):
            if parts[i+1] == 'S' and parts[i].isdigit() and parts[i+2].isdigit():
                return f"{parts[i]}_{parts[i+1]}_{parts[i+2]}"
        
        return basename
    
    def load_nii_files_with_patient_id(directory):
        """加载目录中的所有.nii文件，并按患者ID排序"""
        files = [f for f in os.listdir(directory) if f.endswith('.nii')]
        
        # 按患者ID排序而不是文件名排序
        file_patient_pairs = []
        for file in files:
            patient_id = extract_patient_id_from_filename(file)
            file_patient_pairs.append((file, patient_id))
        
        # 按患者ID排序
        file_patient_pairs.sort(key=lambda x: x[1])
        sorted_files = [pair[0] for pair in file_patient_pairs]
        sorted_patient_ids = [pair[1] for pair in file_patient_pairs]
        
        data_list = []
        for file in sorted_files:
            file_path = os.path.join(directory, file)
            try:
                nii_img = nib.load(file_path)
                data = nii_img.get_fdata()
                
                # 标准化数据
                if data.std() > 0:
                    data = (data - data.mean()) / data.std()
                
                # 确保数据形状为[113, 137, 113]
                target_shape = (113, 137, 113)
                if data.shape != target_shape:
                    # 计算缩放因子
                    zoom_factors = [t/s for t, s in zip(target_shape, data.shape)]
                    data = zoom(data, zoom_factors, order=1)
                
                data_list.append(data.astype(np.float32))
                
            except Exception as e:
                print(f"⚠️  跳过文件 {file}: {e}")
                continue
        
        return data_list, sorted_patient_ids
    
    # 加载AD数据 - 按患者ID排序
    print("📊 加载AD数据...")
    ad_csf_data, ad_patient_ids_csf = load_nii_files_with_patient_id(ad_csf_dir)
    ad_gray_data, ad_patient_ids_gray = load_nii_files_with_patient_id(ad_gray_dir)
    ad_white_data, ad_patient_ids_white = load_nii_files_with_patient_id(ad_white_dir)
    
    print(f"   AD CSF: {len(ad_csf_data)} 文件")
    print(f"   AD GRAY: {len(ad_gray_data)} 文件")
    print(f"   AD WHITE: {len(ad_white_data)} 文件")
    
    # 验证AD数据的患者ID一致性
    if not (ad_patient_ids_csf == ad_patient_ids_gray == ad_patient_ids_white):
        print("⚠️  警告: AD数据中不同组织类型的患者ID顺序不一致")
        # 取交集确保一致性
        common_ad_ids = list(set(ad_patient_ids_csf) & set(ad_patient_ids_gray) & set(ad_patient_ids_white))
        common_ad_ids.sort()
        print(f"   使用共同患者ID: {len(common_ad_ids)} 个")
    else:
        common_ad_ids = ad_patient_ids_csf
    
    # 加载CN数据 - 按患者ID排序
    print("📊 加载CN数据...")
    cn_csf_data, cn_patient_ids_csf = load_nii_files_with_patient_id(cn_csf_dir)
    cn_gray_data, cn_patient_ids_gray = load_nii_files_with_patient_id(cn_gray_dir)
    cn_white_data, cn_patient_ids_white = load_nii_files_with_patient_id(cn_white_dir)
    
    print(f"   CN CSF: {len(cn_csf_data)} 文件")
    print(f"   CN GRAY: {len(cn_gray_data)} 文件")
    print(f"   CN WHITE: {len(cn_white_data)} 文件")
    
    # 验证CN数据的患者ID一致性
    if not (cn_patient_ids_csf == cn_patient_ids_gray == cn_patient_ids_white):
        print("⚠️  警告: CN数据中不同组织类型的患者ID顺序不一致")
        # 取交集确保一致性
        common_cn_ids = list(set(cn_patient_ids_csf) & set(cn_patient_ids_gray) & set(cn_patient_ids_white))
        common_cn_ids.sort()
        print(f"   使用共同患者ID: {len(common_cn_ids)} 个")
    else:
        common_cn_ids = cn_patient_ids_csf
    
    # 确定最终样本数量
    ad_count = len(common_ad_ids)
    cn_count = len(common_cn_ids)
    
    print(f"📈 每类有效样本数: AD={ad_count}, CN={cn_count}")
    
    # 应用样本数量限制
    if max_samples_per_class:
        ad_count = min(ad_count, max_samples_per_class)
        cn_count = min(cn_count, max_samples_per_class)
        print(f"🔧 应用样本限制: AD={ad_count}, CN={cn_count}")
    
    # 构建早期融合数据 - 确保患者ID对齐
    all_images = []
    all_labels = []
    all_patient_ids = []
    
    # 处理AD数据 - 按患者ID顺序
    print(f"🔄 构建AD数据...")
    for i in range(ad_count):
        patient_id = common_ad_ids[i]
        
        # 找到对应的数据索引
        csf_idx = ad_patient_ids_csf.index(patient_id)
        gray_idx = ad_patient_ids_gray.index(patient_id)
        white_idx = ad_patient_ids_white.index(patient_id)
        
        # 合并三种组织类型为3通道图像 [CSF, GRAY, WHITE]
        combined_image = np.stack([
            ad_csf_data[csf_idx],    # 通道0: CSF
            ad_gray_data[gray_idx],   # 通道1: GRAY  
            ad_white_data[white_idx]  # 通道2: WHITE
        ], axis=0)  # 结果形状: [3, 113, 137, 113]
        
        all_images.append(combined_image)
        all_labels.append(1)  # AD = 1
        all_patient_ids.append(patient_id)
    
    # 处理CN数据 - 按患者ID顺序
    print(f"🔄 构建CN数据...")
    for i in range(cn_count):
        patient_id = common_cn_ids[i]
        
        # 找到对应的数据索引
        csf_idx = cn_patient_ids_csf.index(patient_id)
        gray_idx = cn_patient_ids_gray.index(patient_id)
        white_idx = cn_patient_ids_white.index(patient_id)
        
        # 合并三种组织类型为3通道图像 [CSF, GRAY, WHITE]
        combined_image = np.stack([
            cn_csf_data[csf_idx],    # 通道0: CSF
            cn_gray_data[gray_idx],   # 通道1: GRAY
            cn_white_data[white_idx]  # 通道2: WHITE
        ], axis=0)  # 结果形状: [3, 113, 137, 113]
        
        all_images.append(combined_image)
        all_labels.append(0)  # CN = 0
        all_patient_ids.append(patient_id)
    
    # 转换为numpy数组
    image_data = np.array(all_images, dtype=np.float32)  # [N, 3, 113, 137, 113]
    labels = np.array(all_labels, dtype=np.int64)        # [N]
    
    print(f"✅ 图像数据加载完成:")
    print(f"   图像数据形状: {image_data.shape}")
    print(f"   标签形状: {labels.shape}")
    print(f"   标签分布: AD={np.sum(labels==1)}, CN={np.sum(labels==0)}")
    print(f"   患者ID示例: {all_patient_ids[:5]}")
    
    return image_data, labels, all_patient_ids 
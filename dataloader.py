import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging

class VideoFrameDataset(Dataset):
    """
    用于处理视频抽帧后的图像数据集类
    支持批量读取、预处理和路径管理
    """
    def __init__(self, data_folder, target_size=(640, 480)):
        """
        初始化数据集
        Args:
            data_folder (str): 包含视频帧图片的文件夹路径
            target_size (tuple): 目标图像尺寸 (width, height)
        """
        self.data_folder = data_folder
        self.target_size = target_size
        self.logger = logging.getLogger('VideoProcessor.Dataset')
        
        # 支持的图片格式
        self.supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        
        # 收集所有图片路径
        self.image_paths = []
        for filename in sorted(os.listdir(data_folder)):
            if filename.lower().endswith(self.supported_formats):
                self.image_paths.append(os.path.join(data_folder, filename))
        
        self.logger.info(f"数据集初始化完成: {len(self.image_paths)} 张图片")

    def __len__(self):
        """返回数据集中的图片数量"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        返回预处理后的图像和路径
        
        返回:
            tuple: (processed_image, image_path)
                   processed_image: 预处理后的numpy数组 (H, W, C)
                   image_path: 原始图片路径
        """
        img_path = self.image_paths[idx]
        
        try:
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"无法读取图像: {img_path}")
            
            # 颜色空间转换: BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 调整图像尺寸
            if img.shape[:2] != self.target_size[::-1]:
                img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
            
            # 确保数据类型为uint8
            img = img.astype(np.uint8)
            
            return img, img_path
            
        except Exception as e:
            self.logger.error(f"处理图像失败 {img_path}: {str(e)}")
            # 返回空图像和路径
            empty_img = np.zeros((*self.target_size[::-1], 3), dtype=np.uint8)
            return empty_img, img_path

    def get_dataloader(self, batch_size=16, num_workers=0, shuffle=False):
        """
        创建DataLoader实例
        Args:
            batch_size (int): 批次大小，默认16
            num_workers (int): 工作进程数
            shuffle (bool): 是否打乱数据
        Returns:
            DataLoader: 配置好的数据加载器
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False
        )

# 假设你的图片文件夹在当前目录下，名为 'data'
# 你需要先创建一个名为 'data' 的文件夹，并在其中放入一些图片
# 例如：
# data/image1.jpg
# data/image2.png
# ...

# 示例用法
data_folder = 'data'
if not os.path.exists(data_folder):
    print(f"文件夹 '{data_folder}' 不存在。请先创建该文件夹并放入图片。")
else:
    # 创建数据集实例
    dataset = MyImageFolderDataset(data_folder)
    
    # 创建 DataLoader 实例
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # 打印加载的数据
    print(f"数据集中共有 {len(dataset)} 张图片。")
    print("-" * 20)
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        print(f'Batch {batch_idx + 1}:')
        print(f'Inputs (Tensor shape): {inputs.shape}')
        print(f'Labels (Paths):')
        for label in labels:
            print(f'  - {label}')
        print("-" * 20)
        # 这里只打印一个批次作为示例，你可以根据需要调整
        if batch_idx >= 0:
            break

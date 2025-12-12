"""
数据集类 - 加载和处理鼠标轨迹数据
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config.config import Config
import math


class MouseTrajectoryDataset(Dataset):
    def __init__(self, csv_file, normalize=True):
        """
        初始化数据集
        Args:
            csv_file: CSV数据文件路径
            normalize: 是否归一化坐标
        """
        self.data = pd.read_csv(csv_file)
        self.normalize = normalize

        # 按group_id分组
        self.trajectories = []
        self.group_ids = self.data['group_id'].unique()

        # 统计信息用于归一化
        if self.normalize:
            self.coord_stats = self._compute_normalization_stats()

        # 处理每组轨迹
        for group_id in self.group_ids:
            group_data = self.data[self.data['group_id'] == group_id].sort_values('sequence_id')

            if len(group_data) < 2:  # 跳过太短的轨迹
                continue

            trajectory = self._process_trajectory(group_data)
            if trajectory is not None:
                self.trajectories.append(trajectory)

        print(f"加载了 {len(self.trajectories)} 条轨迹")

    def _compute_normalization_stats(self):
        """计算归一化统计信息"""
        coords_data = self.data[['start_x', 'start_y', 'end_x', 'end_y', 'current_x', 'current_y']]

        stats = {
            'coord_mean': coords_data.mean().mean(),
            'coord_std': coords_data.std().mean(),
            'velocity_mean': self.data['velocity'].mean(),
            'velocity_std': self.data['velocity'].std(),
            'acceleration_mean': self.data['acceleration'].mean(),
            'acceleration_std': self.data['acceleration'].std(),
            'direction_mean': self.data['direction'].mean(),
            'direction_std': self.data['direction'].std(),
        }

        return stats

    def _process_trajectory(self, group_data):
        """
        处理单条轨迹，添加额外特征
        Args:
            group_data: 单个group的DataFrame
        Returns:
            dict: 包含特征和目标的字典
        """
        features = []

        for idx, row in group_data.iterrows():
            # 基础坐标
            start_x = row['start_x']
            start_y = row['start_y']
            end_x = row['end_x']
            end_y = row['end_y']
            current_x = row['current_x']
            current_y = row['current_y']

            # 归一化坐标
            if self.normalize:
                start_x = (start_x - self.coord_stats['coord_mean']) / self.coord_stats['coord_std']
                start_y = (start_y - self.coord_stats['coord_mean']) / self.coord_stats['coord_std']
                end_x = (end_x - self.coord_stats['coord_mean']) / self.coord_stats['coord_std']
                end_y = (end_y - self.coord_stats['coord_mean']) / self.coord_stats['coord_std']
                current_x = (current_x - self.coord_stats['coord_mean']) / self.coord_stats['coord_std']
                current_y = (current_y - self.coord_stats['coord_mean']) / self.coord_stats['coord_std']

            # 速度、加速度、方向
            velocity = row['velocity']
            acceleration = row['acceleration']
            direction = row['direction']

            # 归一化速度、加速度、方向
            if self.normalize:
                velocity = (velocity - self.coord_stats['velocity_mean']) / (self.coord_stats['velocity_std'] + 1e-8)
                acceleration = (acceleration - self.coord_stats['acceleration_mean']) / (self.coord_stats['acceleration_std'] + 1e-8)
                direction = (direction - self.coord_stats['direction_mean']) / (self.coord_stats['direction_std'] + 1e-8)

            # 计算相对距离
            distance = math.sqrt((current_x - start_x)**2 + (current_y - start_y)**2)

            # 组合特征向量 [start_x, start_y, end_x, end_y, current_x, current_y, velocity, acceleration, direction, distance]
            feature_vector = [
                start_x, start_y,
                end_x, end_y,
                current_x, current_y,
                velocity, acceleration, direction, distance
            ]

            features.append(feature_vector)

        if len(features) == 0:
            return None

        features = np.array(features, dtype=np.float32)

        # 获取起点和终点信息
        first_row = group_data.iloc[0]
        start_point = np.array([first_row['start_x'], first_row['start_y']], dtype=np.float32)
        end_point = np.array([first_row['end_x'], first_row['end_y']], dtype=np.float32)

        # 归一化起点终点
        if self.normalize:
            start_point = (start_point - self.coord_stats['coord_mean']) / self.coord_stats['coord_std']
            end_point = (end_point - self.coord_stats['coord_mean']) / self.coord_stats['coord_std']

        return {
            'features': features,  # (seq_len, feature_dim)
            'start_point': start_point,  # (2,)
            'end_point': end_point,  # (2,)
            'length': len(features)  # 轨迹长度
        }

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]

    def get_stats(self):
        """返回归一化统计信息"""
        return self.coord_stats if self.normalize else None


def collate_fn(batch):
    """
    自定义collate函数，处理不同长度的序列
    使用padding将序列填充到batch中的最大长度
    """
    # 找到batch中最长的序列
    max_length = max([item['length'] for item in batch])

    batch_features = []
    batch_start_points = []
    batch_end_points = []
    batch_lengths = []
    batch_masks = []

    for item in batch:
        features = item['features']
        length = item['length']

        # Padding
        if length < max_length:
            padding = np.zeros((max_length - length, features.shape[1]), dtype=np.float32)
            features = np.vstack([features, padding])

        # 创建mask（True表示真实数据，False表示padding）
        mask = np.zeros(max_length, dtype=np.float32)
        mask[:length] = 1.0

        batch_features.append(features)
        batch_start_points.append(item['start_point'])
        batch_end_points.append(item['end_point'])
        batch_lengths.append(length)
        batch_masks.append(mask)

    return {
        'features': torch.FloatTensor(np.array(batch_features)),  # (batch, max_len, feature_dim)
        'start_point': torch.FloatTensor(np.array(batch_start_points)),  # (batch, 2)
        'end_point': torch.FloatTensor(np.array(batch_end_points)),  # (batch, 2)
        'length': torch.LongTensor(batch_lengths),  # (batch,)
        'mask': torch.FloatTensor(np.array(batch_masks))  # (batch, max_len)
    }


def create_data_loaders(csv_file, batch_size=32, train_split=0.8):
    """
    创建训练和验证数据加载器
    """
    # 加载完整数据集
    full_dataset = MouseTrajectoryDataset(csv_file, normalize=True)

    # 划分训练集和验证集
    train_size = int(len(full_dataset) * train_split)
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size]
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    print(f"训练集大小: {train_size}, 验证集大小: {val_size}")

    return train_loader, val_loader, full_dataset.get_stats()


if __name__ == '__main__':
    # 测试数据集加载
    dataset = MouseTrajectoryDataset(Config.DATA_FILE)
    print(f"数据集大小: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"样本特征形状: {sample['features'].shape}")
        print(f"起点: {sample['start_point']}")
        print(f"终点: {sample['end_point']}")
        print(f"轨迹长度: {sample['length']}")
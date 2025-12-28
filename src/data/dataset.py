import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler
import numpy as np
import math

class TrajectoryDataset(Dataset):
    def __init__(self, csv_file, max_seq_len=50, min_seq_len=5,
                 screen_width=1920, screen_height=1080):
        """
        初始化数据集，支持变长序列

        Args:
            csv_file: 数据文件路径
            max_seq_len: 最大序列长度（默认50）
            min_seq_len: 最小序列长度（默认5）
            screen_width: 屏幕宽度，用于归一化（默认1920）
            screen_height: 屏幕高度，用于归一化（默认1080）
        """
        # 读取数据
        df = pd.read_csv(csv_file)
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.screen_width = screen_width
        self.screen_height = screen_height

        # 1. 提取特征列
        # 输入特征：当前坐标(x,y)、速度、加速度、方向、位置编码。维度 = 6
        self.feat_cols = ['current_x', 'current_y', 'velocity', 'acceleration', 'direction']
        # 条件列：起点(x,y) 和 终点(x,y)。维度 = 4
        self.cond_cols = ['start_x', 'start_y', 'end_x', 'end_y']

        # 处理加速度异常值（clip到合理范围）
        # 鼠标移动的加速度通常在 [-1000, 1000] 范围内
        df['acceleration'] = df['acceleration'].clip(-1000, 1000)

        # 2. 归一化
        # ⚠️ 关键修复：坐标使用简单归一化，与推理时保持一致
        # 速度、加速度、方向使用RobustScaler（对异常值鲁棒）
        self.velocity_scaler = RobustScaler()

        # 坐标归一化：简单除以屏幕尺寸
        df['current_x'] = df['current_x'] / screen_width
        df['current_y'] = df['current_y'] / screen_height
        df['start_x'] = df['start_x'] / screen_width
        df['start_y'] = df['start_y'] / screen_height
        df['end_x'] = df['end_x'] / screen_width
        df['end_y'] = df['end_y'] / screen_height

        # 速度、加速度、方向使用RobustScaler
        velocity_features = df[['velocity', 'acceleration', 'direction']].values
        scaled_velocity_features = self.velocity_scaler.fit_transform(velocity_features)
        df[['velocity', 'acceleration', 'direction']] = scaled_velocity_features

        self.sequences = []
        self.conditions = []
        self.seq_lengths = []  # 存储每条轨迹的真实长度

        # 3. 按 group_id 组织序列
        for _, group in df.groupby('group_id'):
            # 获取起点终点（归一化后的坐标）
            cond_data = group[self.cond_cols].values[0]  # [4] 归一化后的条件 [0-1]

            actual_len = len(group)

            # 如果数据太短，跳过
            if actual_len < self.min_seq_len:
                continue

            # ⚠️ 关键修复：不再截断轨迹！
            # 如果轨迹太长（超过 max_seq_len），从末尾往回取 max_seq_len 个点
            # 这样可以确保终点始终在轨迹中
            if actual_len > self.max_seq_len:
                # 从末尾往前取 max_seq_len 个点
                seq_data = group[self.feat_cols].values[-self.max_seq_len:]  # [max_seq_len, 5]
                seq_len = self.max_seq_len
            else:
                # 使用完整轨迹
                seq_data = group[self.feat_cols].values  # [actual_len, 5]
                seq_len = actual_len

            # ⚠️ 关键修复：使用累积距离而不是线性时间步作为位置编码
            # 这样位置编码反映"空间进度"而不是"时间进度"
            # 计算每个点的累积移动距离
            cumulative_distances = [0.0]  # 起点距离为0
            coords = seq_data[:, 0:2]  # 提取 (x, y) 坐标
            for i in range(1, seq_len):
                # 计算当前点与前一个点的距离
                step_dist = np.sqrt(
                    (coords[i, 0] - coords[i-1, 0]) ** 2 +
                    (coords[i, 1] - coords[i-1, 1]) ** 2
                )
                cumulative_distances.append(cumulative_distances[-1] + step_dist)

            # 归一化到 [0, 1]
            total_distance = cumulative_distances[-1]
            if total_distance > 0:
                position_encoding = np.array(cumulative_distances).reshape(-1, 1) / total_distance
            else:
                # 如果总距离为0（所有点重合），使用线性编码作为后备
                position_encoding = np.linspace(0, 1, seq_len).reshape(-1, 1)

            seq_data_with_pos = np.concatenate([seq_data, position_encoding], axis=1)  # [seq_len, 6]

            self.sequences.append(torch.FloatTensor(seq_data_with_pos))
            self.conditions.append(torch.FloatTensor(cond_data))
            self.seq_lengths.append(seq_len)  # 记录真实长度

    @staticmethod
    def calculate_seq_len(start, end, pixels_per_step=30):
        """
        根据起点终点距离动态计算序列长度

        Args:
            start: (x, y) 起点坐标
            end: (x, y) 终点坐标
            pixels_per_step: 每步移动的平均像素数（默认30像素/步）

        Returns:
            seq_len: 合理的序列长度（最少5个点，最多50个点）
        """
        # 计算欧氏距离
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.sqrt(dx**2 + dy**2)

        # 根据距离计算点数：distance / pixels_per_step
        # 距离越远，点数越多
        seq_len = int(distance / pixels_per_step) + 1

        # 限制在合理范围：最少5个点（避免轨迹过于简单），最多50个点（避免计算开销）
        seq_len = max(5, min(50, seq_len))

        return seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        返回一条轨迹及其对应的起点终点约束和真实序列长度

        Returns:
            sequence: [seq_len, 6] 轨迹特征
            condition: [4] 起点终点条件
            seq_length: 标量，真实序列长度
        """
        return self.sequences[idx], self.conditions[idx], self.seq_lengths[idx]

    @staticmethod
    def collate_fn(batch):
        """
        自定义collate函数，用于处理变长序列的batch
        将序列padding到batch中的最大长度

        Args:
            batch: list of (sequence, condition, seq_length)

        Returns:
            padded_sequences: [batch, max_len, 6] padding后的序列
            conditions: [batch, 4] 条件
            seq_lengths: [batch] 真实序列长度
            masks: [batch, max_len] mask矩阵（1表示真实数据，0表示padding）
        """
        sequences, conditions, seq_lengths = zip(*batch)

        # 找到batch中的最大序列长度
        max_len = max(seq_lengths)
        batch_size = len(sequences)
        feat_dim = sequences[0].size(-1)

        # 创建padding后的tensor
        padded_sequences = torch.zeros(batch_size, max_len, feat_dim)
        masks = torch.zeros(batch_size, max_len)

        # 填充数据
        for i, (seq, seq_len) in enumerate(zip(sequences, seq_lengths)):
            padded_sequences[i, :seq_len, :] = seq
            masks[i, :seq_len] = 1

        # 转换条件和长度为tensor
        conditions = torch.stack(conditions)
        seq_lengths = torch.tensor(seq_lengths, dtype=torch.float32)

        return padded_sequences, conditions, seq_lengths, masks
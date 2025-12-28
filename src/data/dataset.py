import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler
import numpy as np
import math

class TrajectoryDataset(Dataset):
    def __init__(self, csv_file, max_seq_len=50, min_seq_len=5, pixels_per_step=30):
        """
        初始化数据集，支持变长序列

        Args:
            csv_file: 数据文件路径
            max_seq_len: 最大序列长度（默认50）
            min_seq_len: 最小序列长度（默认5）
            pixels_per_step: 每步平均像素数，用于计算目标序列长度（默认30）
        """
        # 读取数据
        df = pd.read_csv(csv_file)
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.pixels_per_step = pixels_per_step

        # 1. 提取特征列
        # 输入特征：当前坐标(x,y)、速度、加速度、方向、位置编码。维度 = 6
        self.feat_cols = ['current_x', 'current_y', 'velocity', 'acceleration', 'direction']
        # 条件列：起点(x,y) 和 终点(x,y)。维度 = 4
        self.cond_cols = ['start_x', 'start_y', 'end_x', 'end_y']

        # 处理加速度异常值（clip到合理范围）
        # 鼠标移动的加速度通常在 [-1000, 1000] 范围内
        df['acceleration'] = df['acceleration'].clip(-1000, 1000)

        # 2. 归一化（使用RobustScaler对异常值更鲁棒）
        # RobustScaler使用中位数和四分位数，不受极端值影响
        self.feat_scaler = RobustScaler()
        self.cond_scaler = RobustScaler()

        # 对整个数据集进行拟合和转换
        scaled_feats = self.feat_scaler.fit_transform(df[self.feat_cols])
        scaled_conds = self.cond_scaler.fit_transform(df[self.cond_cols])

        # 将归一化后的数据放回 DataFrame 方便按 group_id 分组
        df[self.feat_cols] = scaled_feats
        df[self.cond_cols] = scaled_conds

        self.sequences = []
        self.conditions = []
        self.seq_lengths = []  # 存储每条轨迹的真实长度

        # 3. 按 group_id 组织序列
        for _, group in df.groupby('group_id'):
            # 获取起点终点（使用原始未归一化的坐标计算距离）
            cond_data = group[self.cond_cols].values[0]  # [4] 归一化后的条件

            # 需要原始坐标来计算目标长度
            # 注意：这里需要逆变换获取原始坐标
            original_cond = self.cond_scaler.inverse_transform(cond_data.reshape(1, -1))[0]
            start = (original_cond[0], original_cond[1])
            end = (original_cond[2], original_cond[3])

            # 计算根据距离应该有的序列长度（目标长度）
            target_seq_len = self.calculate_seq_len(start, end, self.pixels_per_step)

            # 使用实际数据长度和目标长度的较小值（但不小于min_seq_len）
            actual_len = len(group)
            seq_len = max(self.min_seq_len, min(actual_len, target_seq_len, self.max_seq_len))

            # 如果数据太短，跳过
            if actual_len < self.min_seq_len:
                continue

            # 截取到计算出的长度
            seq_data = group[self.feat_cols].values[:seq_len]  # [seq_len, 5]

            # 添加位置编码：归一化的时间步 [0, 1/(seq_len-1), ..., 1]
            # 让模型知道当前在轨迹的什么阶段（开始/中间/结束）
            position_encoding = np.linspace(0, 1, seq_len).reshape(-1, 1)  # [seq_len, 1]
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
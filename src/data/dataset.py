import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class TrajectoryDataset(Dataset):
    def __init__(self, csv_file, seq_len=20):
        # 读取数据
        df = pd.read_csv(csv_file)
        self.seq_len = seq_len

        # 1. 提取特征列
        # 输入特征：当前坐标(x,y)、速度、加速度、方向。维度 = 5
        self.feat_cols = ['current_x', 'current_y', 'velocity', 'acceleration', 'direction']
        # 条件列：起点(x,y) 和 终点(x,y)。维度 = 4
        self.cond_cols = ['start_x', 'start_y', 'end_x', 'end_y']

        # 2. 归一化（神经网络对量级敏感，必须将坐标、速度等缩放到均值0方差1附近）
        self.feat_scaler = StandardScaler()
        self.cond_scaler = StandardScaler()

        # 对整个数据集进行拟合和转换
        scaled_feats = self.feat_scaler.fit_transform(df[self.feat_cols])
        scaled_conds = self.cond_scaler.fit_transform(df[self.cond_cols])

        # 将归一化后的数据放回 DataFrame 方便按 group_id 分组
        df[self.feat_cols] = scaled_feats
        df[self.cond_cols] = scaled_conds

        self.sequences = []
        self.conditions = []

        # 3. 按 group_id 组织序列
        for _, group in df.groupby('group_id'):
            # 如果某组数据太短，则跳过；如果太长，则截取前 seq_len 个点
            if len(group) >= seq_len:
                seq_data = group[self.feat_cols].values[:seq_len] # [seq_len, 5]
                cond_data = group[self.cond_cols].values[0]       # [4] (每组的起点终点是一致的)

                self.sequences.append(torch.FloatTensor(seq_data))
                self.conditions.append(torch.FloatTensor(cond_data))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # 返回一条轨迹及其对应的起点终点约束
        return self.sequences[idx], self.conditions[idx]
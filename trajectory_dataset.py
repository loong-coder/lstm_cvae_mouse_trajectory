import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import math

# 假设画布大小与您生成数据时相同
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
MAX_TRAJECTORY_LENGTH = 300  # 假设最大轨迹长度，需根据实际数据调整


class TrajectoryDataset(Dataset):
    def __init__(self, csv_file, max_len=MAX_TRAJECTORY_LENGTH):
        self.max_len = max_len
        self.data = self._load_and_process(csv_file)

    def _load_and_process(self, csv_file):
        df = pd.read_csv(csv_file)
        processed_data = []

        # 归一化函数
        def normalize_x(x):
            return x / CANVAS_WIDTH

        def normalize_y(y):
            return y / CANVAS_HEIGHT

        # 归一化距离 (使用对角线长度作为最大值)
        max_dist = math.sqrt(CANVAS_WIDTH ** 2 + CANVAS_HEIGHT ** 2)

        def normalize_dist(d):
            return d / max_dist

        for trial_id, group in df.groupby('Trial_ID'):
            # --- 1. 任务输入 (Task Context) ---
            start_x, start_y = group['Start_X'].iloc[0], group['Start_Y'].iloc[0]
            end_x, end_y = group['End_X'].iloc[0], group['End_Y'].iloc[0]
            dist = group['Distance_Euclidean'].iloc[0]

            task_context = np.array([
                normalize_x(start_x), normalize_y(start_y),
                normalize_x(end_x), normalize_y(end_y),
                normalize_dist(dist)
            ], dtype=np.float32)

            # --- 2. 序列输出 (Trajectory Sequence) ---
            # 我们预测的是相对于起点的归一化相对坐标
            traj_x = group['Mouse_X'].values - start_x
            traj_y = group['Mouse_Y'].values - start_y

            # 轨迹点归一化（除以画布尺寸）
            traj_x_norm = normalize_x(traj_x)
            traj_y_norm = normalize_y(traj_y)

            # 相对时间戳 (毫秒)
            timestamps = group['Timestamp_ms'].values / 1000.0  # 转换为秒

            # 计算移动方向（角度，弧度制）
            # 使用相对于起点到终点的方向作为参考
            target_direction = np.arctan2(end_y - start_y, end_x - start_x)

            # 计算每个点相对于起点的方向
            directions = np.zeros(len(traj_x), dtype=np.float32)
            for i in range(len(traj_x)):
                if i == 0:
                    # 第一个点使用目标方向
                    directions[i] = target_direction
                else:
                    # 计算移动方向
                    dx = traj_x[i] - traj_x[i-1]
                    dy = traj_y[i] - traj_y[i-1]
                    if dx != 0 or dy != 0:
                        directions[i] = np.arctan2(dy, dx)
                    else:
                        directions[i] = directions[i-1]  # 保持前一个方向

            # 归一化方向到 [-1, 1] (从 [-π, π] 映射)
            directions_norm = directions / np.pi

            # 计算速度 (像素/秒)
            speeds = np.zeros(len(traj_x), dtype=np.float32)
            for i in range(len(traj_x)):
                if i == 0:
                    speeds[i] = 0.0  # 起点速度为0
                else:
                    # 计算距离（像素）
                    dx = traj_x[i] - traj_x[i-1]
                    dy = traj_y[i] - traj_y[i-1]
                    distance = math.sqrt(dx**2 + dy**2)

                    # 计算时间差（秒）
                    time_diff = timestamps[i] - timestamps[i-1]

                    # 计算速度（像素/秒）
                    if time_diff > 0:
                        speeds[i] = distance / time_diff
                    else:
                        speeds[i] = speeds[i-1]  # 时间相同则保持前一个速度

            # 归一化速度（使用最大可能速度，假设最快速度为对角线距离/秒）
            max_speed = max_dist  # 使用对角线长度作为参考
            speeds_norm = np.clip(speeds / max_speed, 0, 1)  # 限制在 [0, 1]

            # 组合序列: (T, features) -> (T, 5)
            # 特征: [Relative_X_norm, Relative_Y_norm, Time_s, Direction_norm, Speed_norm]
            sequence = np.stack([traj_x_norm, traj_y_norm, timestamps, directions_norm, speeds_norm], axis=1).astype(np.float32)

            # --- 3. Padding (零填充) ---
            seq_len = len(sequence)
            if seq_len > self.max_len:
                sequence = sequence[:self.max_len]
                seq_len = self.max_len

            # 填充到最大长度
            padding_needed = self.max_len - seq_len
            if padding_needed > 0:
                padding = np.zeros((padding_needed, sequence.shape[1]), dtype=np.float32)
                sequence = np.concatenate([sequence, padding], axis=0)

            # Sequence: (L_max, 5)
            # Task Context: (5,)
            # Length: 标量
            processed_data.append((task_context, sequence, seq_len))

        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
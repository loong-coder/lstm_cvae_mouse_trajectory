"""
轨迹工具类 - 提取和处理模型生成的轨迹点信息
"""
import torch
import numpy as np
import math


class TrajectoryExtractor:
    """
    轨迹提取器：从模型生成的原始输出中提取轨迹点信息
    包括坐标、方向、速度、加速度等
    """

    def __init__(self, norm_stats=None):
        """
        初始化
        Args:
            norm_stats: 归一化统计信息（用于反归一化）
        """
        self.norm_stats = norm_stats

    def denormalize_coords(self, coords):
        """
        反归一化坐标
        Args:
            coords: 归一化后的坐标
        Returns:
            denormalized: 原始坐标
        """
        if self.norm_stats is None:
            return coords

        return coords * self.norm_stats['coord_std'] + self.norm_stats['coord_mean']

    def denormalize_velocity(self, velocity):
        """反归一化速度"""
        if self.norm_stats is None:
            return velocity

        return velocity * (self.norm_stats['velocity_std'] + 1e-8) + self.norm_stats['velocity_mean']

    def denormalize_acceleration(self, acceleration):
        """反归一化加速度"""
        if self.norm_stats is None:
            return acceleration

        return acceleration * (self.norm_stats['acceleration_std'] + 1e-8) + self.norm_stats['acceleration_mean']

    def denormalize_direction(self, direction):
        """反归一化方向"""
        if self.norm_stats is None:
            return direction

        return direction * (self.norm_stats['direction_std'] + 1e-8) + self.norm_stats['direction_mean']

    def extract_trajectory_points(self, generated_output):
        """
        从模型生成的输出中提取轨迹点信息
        Args:
            generated_output: (batch, seq_len, input_dim) 或 (seq_len, input_dim)
                             input_dim=10: [start_x, start_y, end_x, end_y, current_x, current_y,
                                           velocity, acceleration, direction, distance]
        Returns:
            trajectory_points: list of dict, 每个dict包含一个轨迹点的信息
        """
        # 处理batch维度
        if len(generated_output.shape) == 3:
            # 只处理第一个样本
            output = generated_output[0]  # (seq_len, input_dim)
        else:
            output = generated_output

        # 转换为numpy（如果是tensor）
        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()

        trajectory_points = []

        for i in range(output.shape[0]):
            point_data = output[i]  # (input_dim,)

            # 提取各个字段
            start_x, start_y = point_data[0], point_data[1]
            end_x, end_y = point_data[2], point_data[3]
            current_x, current_y = point_data[4], point_data[5]
            velocity = point_data[6]
            acceleration = point_data[7]
            direction = point_data[8]
            distance = point_data[9]

            # 反归一化
            start_x_denorm = self.denormalize_coords(start_x)
            start_y_denorm = self.denormalize_coords(start_y)
            end_x_denorm = self.denormalize_coords(end_x)
            end_y_denorm = self.denormalize_coords(end_y)
            current_x_denorm = self.denormalize_coords(current_x)
            current_y_denorm = self.denormalize_coords(current_y)

            velocity_denorm = self.denormalize_velocity(velocity)
            acceleration_denorm = self.denormalize_acceleration(acceleration)
            direction_denorm = self.denormalize_direction(direction)

            # 创建轨迹点字典
            point = {
                'index': i,
                'start_x': float(start_x_denorm),
                'start_y': float(start_y_denorm),
                'end_x': float(end_x_denorm),
                'end_y': float(end_y_denorm),
                'current_x': float(current_x_denorm),
                'current_y': float(current_y_denorm),
                'velocity': float(velocity_denorm),
                'acceleration': float(acceleration_denorm),
                'direction': float(direction_denorm),
                'distance': float(distance)
            }

            trajectory_points.append(point)

        return trajectory_points

    def extract_coordinates_only(self, generated_output):
        """
        只提取坐标序列（用于绘图）
        Args:
            generated_output: 模型输出
        Returns:
            coords: numpy array (seq_len, 2) - [x, y]坐标序列
        """
        points = self.extract_trajectory_points(generated_output)

        coords = np.array([[p['current_x'], p['current_y']] for p in points])

        return coords

    def calculate_trajectory_metrics(self, trajectory_points):
        """
        计算轨迹的统计指标
        Args:
            trajectory_points: extract_trajectory_points的输出
        Returns:
            metrics: dict 包含各种统计指标
        """
        if len(trajectory_points) == 0:
            return {}

        # 提取数据
        velocities = [p['velocity'] for p in trajectory_points]
        accelerations = [p['acceleration'] for p in trajectory_points]
        distances = [p['distance'] for p in trajectory_points]

        # 计算总路径长度
        total_distance = sum(distances)

        # 计算平均速度和加速度
        avg_velocity = np.mean(velocities)
        max_velocity = np.max(velocities)
        min_velocity = np.min(velocities)

        avg_acceleration = np.mean(accelerations)
        max_acceleration = np.max(accelerations)

        # 计算轨迹的平滑度（方向变化的标准差）
        directions = [p['direction'] for p in trajectory_points]
        direction_changes = [abs(directions[i] - directions[i-1]) for i in range(1, len(directions))]
        smoothness = np.std(direction_changes) if len(direction_changes) > 0 else 0

        # 起点到终点的直线距离
        start_point = (trajectory_points[0]['start_x'], trajectory_points[0]['start_y'])
        end_point = (trajectory_points[-1]['end_x'], trajectory_points[-1]['end_y'])
        straight_line_distance = math.sqrt(
            (end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2
        )

        # 路径效率（直线距离 / 实际路径长度）
        path_efficiency = straight_line_distance / total_distance if total_distance > 0 else 0

        metrics = {
            'num_points': len(trajectory_points),
            'total_distance': total_distance,
            'straight_line_distance': straight_line_distance,
            'path_efficiency': path_efficiency,
            'avg_velocity': avg_velocity,
            'max_velocity': max_velocity,
            'min_velocity': min_velocity,
            'avg_acceleration': avg_acceleration,
            'max_acceleration': max_acceleration,
            'smoothness': smoothness,
            'start_point': start_point,
            'end_point': end_point
        }

        return metrics

    def save_trajectory_to_csv(self, trajectory_points, filename):
        """
        将轨迹点保存到CSV文件
        Args:
            trajectory_points: extract_trajectory_points的输出
            filename: 保存的文件名
        """
        import pandas as pd

        df = pd.DataFrame(trajectory_points)
        df.to_csv(filename, index=False)
        print(f"轨迹已保存到: {filename}")

    def interpolate_trajectory(self, trajectory_points, num_points):
        """
        对轨迹进行插值，生成更密集或更稀疏的点
        Args:
            trajectory_points: 原始轨迹点
            num_points: 目标点数
        Returns:
            interpolated_points: 插值后的轨迹点坐标
        """
        if len(trajectory_points) == 0:
            return []

        # 提取坐标
        coords = np.array([[p['current_x'], p['current_y']] for p in trajectory_points])

        # 使用线性插值
        from scipy import interpolate

        # 原始索引
        original_indices = np.arange(len(coords))

        # 新索引
        new_indices = np.linspace(0, len(coords) - 1, num_points)

        # 对x和y分别插值
        f_x = interpolate.interp1d(original_indices, coords[:, 0], kind='linear')
        f_y = interpolate.interp1d(original_indices, coords[:, 1], kind='linear')

        interpolated_coords = np.column_stack([f_x(new_indices), f_y(new_indices)])

        return interpolated_coords


class TrajectoryComparator:
    """
    轨迹比较器：比较真实轨迹和生成轨迹
    """

    @staticmethod
    def compute_dtw_distance(traj1, traj2):
        """
        计算两条轨迹的DTW（动态时间规整）距离
        Args:
            traj1: (n, 2) 轨迹1的坐标
            traj2: (m, 2) 轨迹2的坐标
        Returns:
            distance: DTW距离
        """
        from scipy.spatial.distance import euclidean
        from fastdtw import fastdtw

        distance, path = fastdtw(traj1, traj2, dist=euclidean)
        return distance

    @staticmethod
    def compute_frechet_distance(traj1, traj2):
        """
        计算两条轨迹的Fréchet距离
        Args:
            traj1: (n, 2) 轨迹1
            traj2: (m, 2) 轨迹2
        Returns:
            distance: Fréchet距离
        """
        # 简化实现：使用平均欧氏距离
        # 完整实现需要使用专门的Fréchet距离算法
        min_len = min(len(traj1), len(traj2))

        if min_len == 0:
            return float('inf')

        # 对齐到相同长度
        from scipy import interpolate

        t1_indices = np.linspace(0, len(traj1) - 1, min_len)
        t2_indices = np.linspace(0, len(traj2) - 1, min_len)

        f1_x = interpolate.interp1d(np.arange(len(traj1)), traj1[:, 0], kind='linear')
        f1_y = interpolate.interp1d(np.arange(len(traj1)), traj1[:, 1], kind='linear')
        f2_x = interpolate.interp1d(np.arange(len(traj2)), traj2[:, 0], kind='linear')
        f2_y = interpolate.interp1d(np.arange(len(traj2)), traj2[:, 1], kind='linear')

        t1_resampled = np.column_stack([f1_x(t1_indices), f1_y(t1_indices)])
        t2_resampled = np.column_stack([f2_x(t2_indices), f2_y(t2_indices)])

        # 计算平均距离
        distances = np.sqrt(np.sum((t1_resampled - t2_resampled)**2, axis=1))
        avg_distance = np.mean(distances)

        return avg_distance


if __name__ == '__main__':
    # 测试轨迹提取器
    print("测试轨迹提取器...")

    # 创建模拟数据
    batch_size = 1
    seq_len = 50
    input_dim = 10

    fake_output = torch.randn(batch_size, seq_len, input_dim)

    # 创建提取器
    extractor = TrajectoryExtractor()

    # 提取轨迹点
    points = extractor.extract_trajectory_points(fake_output)
    print(f"提取了 {len(points)} 个轨迹点")
    print(f"第一个点: {points[0]}")

    # 提取坐标
    coords = extractor.extract_coordinates_only(fake_output)
    print(f"坐标形状: {coords.shape}")

    # 计算指标
    metrics = extractor.calculate_trajectory_metrics(points)
    print(f"轨迹指标: {metrics}")

    print("\n测试完成！")
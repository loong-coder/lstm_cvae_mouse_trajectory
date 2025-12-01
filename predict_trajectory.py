import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from cvae_model import TrajectoryPredictorCVAE
from trajectory_dataset import MAX_TRAJECTORY_LENGTH, CANVAS_WIDTH, CANVAS_HEIGHT
from trajectory_point import TrajectoryPoint, EnhancedTrajectory


class TrajectoryPredictor:
    """基于 CVAE 模型的轨迹预测器"""

    def __init__(self, model_path='cvae_trajectory_predictor.pth',
                 canvas_width=800, canvas_height=600):
        """
        初始化预测器

        Args:
            model_path: 模型权重文件路径
            canvas_width: 画布宽度（用于归一化）
            canvas_height: 画布高度（用于归一化）
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.max_dist = math.sqrt(canvas_width ** 2 + canvas_height ** 2)

        # 加载模型
        self.model = TrajectoryPredictorCVAE(max_len=MAX_TRAJECTORY_LENGTH)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ 模型加载成功: {model_path}")
        except FileNotFoundError:
            print(f"✗ 警告: 未找到模型文件 '{model_path}'")
            raise

    def predict(self, start_x, start_y, end_x, end_y, smooth=True, downsample=True, align_endpoints=True):
        """
        预测从起点到终点的轨迹

        Args:
            start_x: 起点 X 坐标
            start_y: 起点 Y 坐标
            end_x: 终点 X 坐标
            end_y: 终点 Y 坐标
            smooth: 是否平滑处理（移动平均）
            downsample: 是否降采样（减少点数）
            align_endpoints: 是否使用线性变换精确对齐起点和终点

        Returns:
            trajectory: 轨迹坐标列表 [(x1, y1), (x2, y2), ...]
        """
        # 1. 准备任务上下文
        distance = math.dist((start_x, start_y), (end_x, end_y))

        task_context = np.array([
            start_x / self.canvas_width,
            start_y / self.canvas_height,
            end_x / self.canvas_width,
            end_y / self.canvas_height,
            distance / self.max_dist
        ], dtype=np.float32)

        # 2. 转换为张量
        task_context_tensor = torch.from_numpy(task_context).unsqueeze(0).to(self.device)

        # 3. 模型预测
        with torch.no_grad():
            reconstruction, mu, logvar, predicted_length_ratio = self.model(task_context_tensor)
            prediction = reconstruction.cpu().squeeze(0).numpy()
            pred_length_ratio = predicted_length_ratio.cpu().squeeze(0).item()

        # 4. 确定使用的点数
        predicted_length = int(pred_length_ratio * MAX_TRAJECTORY_LENGTH)
        predicted_length = min(predicted_length, len(prediction))

        # 5. 转换为绝对坐标
        trajectory = [(start_x, start_y)]  # 起点

        for i in range(predicted_length):
            # 提取归一化的相对坐标
            rel_x_norm = prediction[i, 0]
            rel_y_norm = prediction[i, 1]

            # 反归一化
            rel_x = rel_x_norm * self.canvas_width
            rel_y = rel_y_norm * self.canvas_height

            # 转换为绝对坐标
            abs_x = rel_x + start_x
            abs_y = rel_y + start_y

            # 过滤超出范围的点
            if abs_x < -50 or abs_x > self.canvas_width + 50 or \
               abs_y < -50 or abs_y > self.canvas_height + 50:
                break

            trajectory.append((abs_x, abs_y))

        # 6. 后处理
        if smooth:
            trajectory = self._smooth_trajectory(trajectory, window_size=5)

        if downsample:
            trajectory = self._downsample_trajectory(trajectory, sample_rate=2)

        # 7. 线性变换对齐终点（保持轨迹形状，精确映射到目标起点和终点）
        if align_endpoints and len(trajectory) >= 2:
            trajectory = self._align_trajectory_to_endpoints(
                trajectory,
                target_start=(start_x, start_y),
                target_end=(end_x, end_y)
            )

        return trajectory

    def predict_enhanced(self, start_x, start_y, end_x, end_y,
                        smooth=True, downsample=True, align_endpoints=True) -> EnhancedTrajectory:
        """
        预测从起点到终点的轨迹（返回包含速度和时间信息的增强轨迹）

        Args:
            start_x: 起点 X 坐标
            start_y: 起点 Y 坐标
            end_x: 终点 X 坐标
            end_y: 终点 Y 坐标
            smooth: 是否平滑处理
            downsample: 是否降采样
            align_endpoints: 是否对齐终点

        Returns:
            EnhancedTrajectory: 包含完整时间和速度信息的轨迹对象
        """
        # 1. 准备任务上下文
        distance = math.dist((start_x, start_y), (end_x, end_y))

        task_context = np.array([
            start_x / self.canvas_width,
            start_y / self.canvas_height,
            end_x / self.canvas_width,
            end_y / self.canvas_height,
            distance / self.max_dist
        ], dtype=np.float32)

        # 2. 转换为张量
        task_context_tensor = torch.from_numpy(task_context).unsqueeze(0).to(self.device)

        # 3. 模型预测
        with torch.no_grad():
            reconstruction, mu, logvar, predicted_length_ratio = self.model(task_context_tensor)
            prediction = reconstruction.cpu().squeeze(0).numpy()
            pred_length_ratio = predicted_length_ratio.cpu().squeeze(0).item()

        # 4. 确定使用的点数
        predicted_length = int(pred_length_ratio * MAX_TRAJECTORY_LENGTH)
        predicted_length = min(predicted_length, len(prediction))

        # 5. 构建包含完整信息的轨迹点列表
        trajectory_points = []

        # 起点
        start_point = TrajectoryPoint(
            x=start_x,
            y=start_y,
            timestamp=0.0,
            speed=0.0,
            direction=0.0,
            duration=0.0
        )
        trajectory_points.append(start_point)

        # 反归一化最大速度
        max_speed = self.max_dist  # 使用对角线长度作为参考

        for i in range(predicted_length):
            # 提取模型预测的归一化数据
            # prediction[i] = [rel_x_norm, rel_y_norm, time, direction_norm, speed_norm]
            rel_x_norm = prediction[i, 0]
            rel_y_norm = prediction[i, 1]
            timestamp = prediction[i, 2]
            direction_norm = prediction[i, 3]
            speed_norm = prediction[i, 4]

            # 反归一化坐标
            rel_x = rel_x_norm * self.canvas_width
            rel_y = rel_y_norm * self.canvas_height
            abs_x = rel_x + start_x
            abs_y = rel_y + start_y

            # 过滤超出范围的点
            if abs_x < -50 or abs_x > self.canvas_width + 50 or \
               abs_y < -50 or abs_y > self.canvas_height + 50:
                break

            # 反归一化方向和速度
            direction = direction_norm * math.pi  # 从 [-1, 1] 映射到 [-π, π]
            speed = speed_norm * max_speed  # 从 [0, 1] 映射到实际速度

            # 计算时间增量（duration）
            if i > 0:
                prev_point = trajectory_points[-1]
                duration = timestamp - prev_point.timestamp
            else:
                duration = timestamp

            # 创建轨迹点
            point = TrajectoryPoint(
                x=abs_x,
                y=abs_y,
                timestamp=timestamp,
                speed=speed,
                direction=direction,
                duration=duration
            )
            trajectory_points.append(point)

        # 6. 创建增强轨迹对象
        enhanced_traj = EnhancedTrajectory(trajectory_points)

        # 7. 后处理（平滑、降采样）
        if smooth or downsample:
            # 先转换为坐标列表进行后处理
            coords = enhanced_traj.get_coordinates()

            if smooth:
                coords = self._smooth_trajectory(coords, window_size=5)

            if downsample:
                coords = self._downsample_trajectory(coords, sample_rate=2)

            # 重新构建轨迹点，保留速度信息
            enhanced_traj = self._rebuild_trajectory_from_coords(
                coords,
                original_trajectory=enhanced_traj
            )

        # 8. 对齐终点
        if align_endpoints and len(enhanced_traj) >= 2:
            coords = enhanced_traj.get_coordinates()
            aligned_coords = self._align_trajectory_to_endpoints(
                coords,
                target_start=(start_x, start_y),
                target_end=(end_x, end_y)
            )
            enhanced_traj = self._rebuild_trajectory_from_coords(
                aligned_coords,
                original_trajectory=enhanced_traj
            )

        # 9. 插值缺失的时间增量
        enhanced_traj.interpolate_missing_durations()

        return enhanced_traj

    def _rebuild_trajectory_from_coords(self, coordinates: list,
                                       original_trajectory: EnhancedTrajectory) -> EnhancedTrajectory:
        """
        从坐标列表重建轨迹，尽可能保留原始的速度和时间信息

        Args:
            coordinates: 坐标列表 [(x1, y1), (x2, y2), ...]
            original_trajectory: 原始轨迹（用于参考速度等信息）

        Returns:
            重建的增强轨迹
        """
        if len(coordinates) == 0:
            return EnhancedTrajectory()

        new_points = []
        original_points = original_trajectory.points

        # 计算平均速度作为默认值
        avg_speed = original_trajectory.average_speed if len(original_points) > 1 else 500.0

        cumulative_time = 0.0

        for i, (x, y) in enumerate(coordinates):
            if i == 0:
                # 起点
                point = TrajectoryPoint(
                    x=x, y=y,
                    timestamp=0.0,
                    speed=0.0,
                    direction=0.0,
                    duration=0.0
                )
            else:
                # 查找最接近的原始点以获取速度信息
                min_dist = float('inf')
                closest_idx = 0
                for j, orig_point in enumerate(original_points):
                    dist = math.sqrt((x - orig_point.x)**2 + (y - orig_point.y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = j

                # 使用最接近点的速度，如果找不到则使用平均速度
                if closest_idx < len(original_points):
                    speed = original_points[closest_idx].speed
                else:
                    speed = avg_speed

                # 确保速度不为0
                if speed <= 0:
                    speed = avg_speed

                # 计算距离和方向
                prev_x, prev_y = coordinates[i-1]
                dx = x - prev_x
                dy = y - prev_y
                distance = math.sqrt(dx**2 + dy**2)
                direction = math.atan2(dy, dx)

                # 计算时间增量
                duration = distance / speed if speed > 0 else 0.01
                cumulative_time += duration

                point = TrajectoryPoint(
                    x=x, y=y,
                    timestamp=cumulative_time,
                    speed=speed,
                    direction=direction,
                    duration=duration
                )

            new_points.append(point)

        return EnhancedTrajectory(new_points)

    def predict_multiple(self, start_x, start_y, end_x, end_y, num_samples=3,
                        smooth=True, downsample=True, align_endpoints=True):
        """
        生成多条可能的轨迹（展示 CVAE 的多样性）

        Args:
            start_x, start_y, end_x, end_y: 起点和终点坐标
            num_samples: 生成样本数量
            smooth: 是否平滑处理
            downsample: 是否降采样
            align_endpoints: 是否使用线性变换精确对齐起点和终点

        Returns:
            trajectories: 轨迹列表 [trajectory1, trajectory2, ...]
        """
        # 1. 准备任务上下文
        distance = math.dist((start_x, start_y), (end_x, end_y))

        task_context = np.array([
            start_x / self.canvas_width,
            start_y / self.canvas_height,
            end_x / self.canvas_width,
            end_y / self.canvas_height,
            distance / self.max_dist
        ], dtype=np.float32)

        task_context_tensor = torch.from_numpy(task_context).unsqueeze(0).to(self.device)

        # 2. 使用 sample 方法生成多个轨迹
        with torch.no_grad():
            trajectories_tensor = self.model.sample(task_context_tensor, num_samples=num_samples)
            trajectories_tensor = trajectories_tensor.squeeze(0).cpu().numpy()  # (num_samples, max_len, 5)

            _, _, _, predicted_length_ratio = self.model(task_context_tensor)
            pred_length_ratio = predicted_length_ratio.cpu().squeeze(0).item()

        predicted_length = int(pred_length_ratio * MAX_TRAJECTORY_LENGTH)

        # 3. 处理每个样本
        all_trajectories = []

        for sample_idx in range(num_samples):
            prediction = trajectories_tensor[sample_idx]
            trajectory = [(start_x, start_y)]

            for i in range(min(predicted_length, len(prediction))):
                rel_x_norm = prediction[i, 0]
                rel_y_norm = prediction[i, 1]

                rel_x = rel_x_norm * self.canvas_width
                rel_y = rel_y_norm * self.canvas_height

                abs_x = rel_x + start_x
                abs_y = rel_y + start_y

                if abs_x < -50 or abs_x > self.canvas_width + 50 or \
                   abs_y < -50 or abs_y > self.canvas_height + 50:
                    break

                trajectory.append((abs_x, abs_y))

            # 后处理
            if smooth:
                trajectory = self._smooth_trajectory(trajectory, window_size=5)
            if downsample:
                trajectory = self._downsample_trajectory(trajectory, sample_rate=2)

            # 线性变换对齐终点
            if align_endpoints and len(trajectory) >= 2:
                trajectory = self._align_trajectory_to_endpoints(
                    trajectory,
                    target_start=(start_x, start_y),
                    target_end=(end_x, end_y)
                )

            all_trajectories.append(trajectory)

        return all_trajectories

    def _smooth_trajectory(self, trajectory, window_size=5):
        """移动平均平滑"""
        if len(trajectory) < window_size:
            return trajectory

        smoothed = [trajectory[0]]

        for i in range(1, len(trajectory) - 1):
            half_window = window_size // 2
            start_idx = max(0, i - half_window)
            end_idx = min(len(trajectory), i + half_window + 1)

            window_points = trajectory[start_idx:end_idx]
            avg_x = sum(p[0] for p in window_points) / len(window_points)
            avg_y = sum(p[1] for p in window_points) / len(window_points)

            smoothed.append((avg_x, avg_y))

        smoothed.append(trajectory[-1])
        return smoothed

    def _downsample_trajectory(self, trajectory, sample_rate=2):
        """降采样减少点数"""
        if len(trajectory) <= 2:
            return trajectory

        sampled = [trajectory[0]]

        for i in range(sample_rate, len(trajectory) - 1, sample_rate):
            sampled.append(trajectory[i])

        if trajectory[-1] not in sampled:
            sampled.append(trajectory[-1])

        return sampled

    def _align_trajectory_to_endpoints(self, trajectory, target_start, target_end,
                                       ensure_monotonic=True):
        """
        通过线性变换将轨迹精确对齐到目标起点和终点

        算法：
        1. 计算当前轨迹的实际起点和终点
        2. 计算缩放和旋转变换
        3. 应用仿射变换将轨迹映射到目标起点和终点
        4. （可选）确保轨迹单调接近终点

        Args:
            trajectory: 原始轨迹 [(x1, y1), (x2, y2), ...]
            target_start: 目标起点 (x, y)
            target_end: 目标终点 (x, y)
            ensure_monotonic: 是否确保轨迹单调接近终点（防止超过后回转）

        Returns:
            aligned_trajectory: 对齐后的轨迹
        """
        if len(trajectory) < 2:
            return trajectory

        # 获取当前起点和终点
        current_start = trajectory[0]
        current_end = trajectory[-1]

        # 计算当前向量和目标向量
        current_vec = (current_end[0] - current_start[0], current_end[1] - current_start[1])
        target_vec = (target_end[0] - target_start[0], target_end[1] - target_start[1])

        # 计算当前长度和目标长度
        current_length = math.sqrt(current_vec[0]**2 + current_vec[1]**2)
        target_length = math.sqrt(target_vec[0]**2 + target_vec[1]**2)

        if current_length < 1e-6:
            # 如果当前轨迹几乎是一个点，直接返回起点到终点的直线
            return [target_start, target_end]

        # 计算缩放比例
        scale = target_length / current_length

        # 计算旋转角度
        current_angle = math.atan2(current_vec[1], current_vec[0])
        target_angle = math.atan2(target_vec[1], target_vec[0])
        rotation_angle = target_angle - current_angle

        # 对每个点应用变换：
        # 1. 平移到原点（减去当前起点）
        # 2. 缩放
        # 3. 旋转
        # 4. 平移到目标起点
        aligned_trajectory = []
        cos_theta = math.cos(rotation_angle)
        sin_theta = math.sin(rotation_angle)

        for x, y in trajectory:
            # 1. 平移到原点
            x_shifted = x - current_start[0]
            y_shifted = y - current_start[1]

            # 2. 缩放
            x_scaled = x_shifted * scale
            y_scaled = y_shifted * scale

            # 3. 旋转
            x_rotated = x_scaled * cos_theta - y_scaled * sin_theta
            y_rotated = x_scaled * sin_theta + y_scaled * cos_theta

            # 4. 平移到目标起点
            x_final = x_rotated + target_start[0]
            y_final = y_rotated + target_start[1]

            aligned_trajectory.append((x_final, y_final))

        # 5. 可选：确保轨迹单调接近终点（防止超过后回转）
        if ensure_monotonic and len(aligned_trajectory) > 2:
            aligned_trajectory = self._ensure_monotonic_approach(aligned_trajectory, target_end)

        return aligned_trajectory

    def _ensure_monotonic_approach(self, trajectory, endpoint):
        """
        确保轨迹单调接近终点，如果发现超过终点后再回转的情况，进行截断

        Args:
            trajectory: 轨迹 [(x1, y1), (x2, y2), ...]
            endpoint: 终点 (x, y)

        Returns:
            修正后的轨迹
        """
        if len(trajectory) < 3:
            return trajectory

        # 计算每个点到终点的距离
        distances = [math.dist(point, endpoint) for point in trajectory]

        # 找到距离最小的点
        min_dist_idx = distances.index(min(distances))

        # 如果最小距离点不是最后一个点，说明后面的点在"远离"终点
        # 这种情况下，我们截断到最小距离点
        if min_dist_idx < len(trajectory) - 1:
            # 截断轨迹
            return trajectory[:min_dist_idx + 1]

        return trajectory


def predict_trajectory(start_x, start_y, end_x, end_y, model_path='cvae_trajectory_predictor.pth'):
    """
    便捷函数：预测单条轨迹

    Args:
        start_x, start_y: 起点坐标
        end_x, end_y: 终点坐标
        model_path: 模型文件路径

    Returns:
        trajectory: 轨迹坐标列表 [(x1, y1), (x2, y2), ...]
    """
    predictor = TrajectoryPredictor(model_path=model_path)
    return predictor.predict(start_x, start_y, end_x, end_y)


# ========== 可视化函数 ==========

def plot_trajectory(trajectory, start_point=None, end_point=None,
                    title="CVAE 轨迹预测", save_path=None, show=True):
    """
    绘制单条轨迹

    Args:
        trajectory: 轨迹坐标列表 [(x1, y1), (x2, y2), ...]
        start_point: 起点坐标 (x, y)，如果为 None 则使用 trajectory[0]
        end_point: 终点坐标 (x, y)，如果为 None 则使用 trajectory[-1]
        title: 图表标题
        save_path: 保存路径（如果提供）
        show: 是否显示图表
    """
    if len(trajectory) == 0:
        print("轨迹为空，无法绘制")
        return

    # 提取 x 和 y 坐标
    x_coords = [float(p[0]) for p in trajectory]
    y_coords = [float(p[1]) for p in trajectory]

    # 确定起点和终点
    if start_point is None:
        start_point = trajectory[0]
    if end_point is None:
        end_point = trajectory[-1]

    # 创建图表
    plt.figure(figsize=(10, 8))

    # 设置黑色背景
    ax = plt.gca()
    ax.set_facecolor('black')
    plt.gcf().patch.set_facecolor('black')

    # 绘制轨迹线（洋红色）
    plt.plot(x_coords, y_coords, color='#FF00FF', linewidth=2, label='预测轨迹')

    # 绘制轨迹点
    plt.scatter(x_coords, y_coords, color='#FF00FF', s=20, alpha=0.6, zorder=3)

    # 绘制起点（蓝色）
    start_circle = Circle((float(start_point[0]), float(start_point[1])),
                          radius=15, color='#0088FF', label='起点', zorder=5)
    ax.add_patch(start_circle)

    # 绘制终点（红色）
    end_circle = Circle((float(end_point[0]), float(end_point[1])),
                        radius=15, color='#FF4444', label='终点', zorder=5)
    ax.add_patch(end_circle)

    # 设置坐标轴
    plt.xlim(0, 800)
    plt.ylim(0, 600)
    plt.gca().invert_yaxis()  # 反转 Y 轴（屏幕坐标系）

    # 设置标题和标签
    plt.title(title, color='white', fontsize=14, pad=20)
    plt.xlabel('X 坐标', color='white', fontsize=12)
    plt.ylabel('Y 坐标', color='white', fontsize=12)

    # 设置刻度颜色
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    # 图例
    plt.legend(loc='upper right', facecolor='black', edgecolor='white',
               labelcolor='white', fontsize=10)

    # 网格
    plt.grid(True, color='gray', alpha=0.3, linestyle='--')

    # 添加信息文本
    info_text = f"轨迹点数: {len(trajectory)}"
    plt.text(0.02, 0.98, info_text, transform=ax.transAxes,
             color='cyan', fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='cyan'))

    plt.tight_layout()

    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black', edgecolor='none')
        print(f"✓ 图表已保存到: {save_path}")

    # 显示图表
    if show:
        plt.show()

    plt.close()


def plot_multiple_trajectories(trajectories, start_point, end_point,
                               title="CVAE 多样本轨迹预测", save_path=None, show=True):
    """
    绘制多条轨迹（展示 CVAE 多样性）

    Args:
        trajectories: 轨迹列表 [trajectory1, trajectory2, ...]
        start_point: 起点坐标 (x, y)
        end_point: 终点坐标 (x, y)
        title: 图表标题
        save_path: 保存路径（如果提供）
        show: 是否显示图表
    """
    if len(trajectories) == 0:
        print("轨迹列表为空，无法绘制")
        return

    # 颜色列表
    colors = ['#FF00FF', '#FF88FF', '#FF00AA', '#AA00FF', '#FF44FF',
              '#FF00CC', '#CC00FF', '#FF66FF', '#DD00FF', '#FF22FF']

    # 创建图表
    plt.figure(figsize=(10, 8))

    # 设置黑色背景
    ax = plt.gca()
    ax.set_facecolor('black')
    plt.gcf().patch.set_facecolor('black')

    # 绘制每条轨迹
    for idx, trajectory in enumerate(trajectories):
        if len(trajectory) == 0:
            continue

        x_coords = [float(p[0]) for p in trajectory]
        y_coords = [float(p[1]) for p in trajectory]

        color = colors[idx % len(colors)]

        # 绘制轨迹线
        plt.plot(x_coords, y_coords, color=color, linewidth=2,
                alpha=0.8, label=f'样本 {idx + 1} ({len(trajectory)} 点)')

        # 绘制轨迹点
        plt.scatter(x_coords, y_coords, color=color, s=15, alpha=0.5, zorder=3)

    # 绘制起点（蓝色）
    start_circle = Circle((float(start_point[0]), float(start_point[1])),
                          radius=15, color='#0088FF', label='起点', zorder=5)
    ax.add_patch(start_circle)

    # 绘制终点（红色）
    end_circle = Circle((float(end_point[0]), float(end_point[1])),
                        radius=15, color='#FF4444', label='终点', zorder=5)
    ax.add_patch(end_circle)

    # 设置坐标轴
    plt.xlim(0, 800)
    plt.ylim(0, 600)
    plt.gca().invert_yaxis()  # 反转 Y 轴

    # 设置标题和标签
    plt.title(title, color='white', fontsize=14, pad=20)
    plt.xlabel('X 坐标', color='white', fontsize=12)
    plt.ylabel('Y 坐标', color='white', fontsize=12)

    # 设置刻度颜色
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    # 图例
    plt.legend(loc='upper right', facecolor='black', edgecolor='white',
               labelcolor='white', fontsize=9)

    # 网格
    plt.grid(True, color='gray', alpha=0.3, linestyle='--')

    # 添加信息文本
    info_text = f"样本数量: {len(trajectories)}"
    plt.text(0.02, 0.98, info_text, transform=ax.transAxes,
             color='cyan', fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='cyan'))

    plt.tight_layout()

    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='black', edgecolor='none')
        print(f"✓ 图表已保存到: {save_path}")

    # 显示图表
    if show:
        plt.show()

    plt.close()


# ========== 使用示例 ==========

if __name__ == '__main__':
    print("=" * 80)
    print("CVAE 轨迹预测示例")
    print("=" * 80)

    # 方法 1: 使用便捷函数（单次预测）
    print("\n方法 1: 使用便捷函数")
    print("-" * 80)

    start_x, start_y = 100, 100
    end_x, end_y = 700, 500

    trajectory = predict_trajectory(start_x, start_y, end_x, end_y)

    print(f"起点: ({start_x}, {start_y})")
    print(f"终点: ({end_x}, {end_y})")
    print(f"生成轨迹点数: {len(trajectory)}")
    print(f"前 5 个点: {trajectory[:5]}")
    print(f"后 5 个点: {trajectory[-5:]}")

    # 方法 2: 使用类（可以多次预测，不需要重复加载模型）
    print("\n方法 2: 使用预测器类（支持多样本）")
    print("-" * 80)

    predictor = TrajectoryPredictor(model_path='cvae_trajectory_predictor.pth')

    # 单条预测
    trajectory_single = predictor.predict(200, 300, 600, 400)
    print(f"\n单条预测: {len(trajectory_single)} 个点")

    # 多条预测（展示多样性）
    trajectories_multi = predictor.predict_multiple(200, 300, 600, 400, num_samples=3)
    print(f"\n多样本预测: 生成了 {len(trajectories_multi)} 条轨迹")
    for idx, traj in enumerate(trajectories_multi):
        print(f"  样本 {idx + 1}: {len(traj)} 个点")

    # 方法 3: 不平滑、不降采样（原始预测）
    print("\n方法 3: 原始预测（不平滑、不降采样）")
    print("-" * 80)

    trajectory_raw = predictor.predict(100, 100, 700, 500, smooth=False, downsample=False)
    print(f"原始轨迹点数: {len(trajectory_raw)}")

    trajectory_processed = predictor.predict(100, 100, 700, 500, smooth=True, downsample=True)
    print(f"处理后轨迹点数: {len(trajectory_processed)}")

    # 方法 4: 可视化单条轨迹
    print("\n方法 4: 可视化单条轨迹")
    print("-" * 80)

    trajectory_viz = predictor.predict(150, 150, 650, 450)
    print(f"生成轨迹: {len(trajectory_viz)} 个点")
    print("正在绘制图表...")

    # 绘制并保存
    plot_trajectory(
        trajectory_viz,
        start_point=(150, 150),
        end_point=(650, 450),
        title="CVAE 单条轨迹预测示例",
        save_path="trajectory_single.png",
        show=False  # 改为 True 可以显示图表窗口
    )

    # 方法 5: 可视化多条轨迹（展示 CVAE 多样性）
    print("\n方法 5: 可视化多条轨迹")
    print("-" * 80)

    start_point = (200, 200)
    end_point = (600, 400)

    trajectories_viz = predictor.predict_multiple(*start_point, *end_point, num_samples=5)
    print(f"生成了 {len(trajectories_viz)} 条轨迹")
    for i, traj in enumerate(trajectories_viz, 1):
        print(f"  样本 {i}: {len(traj)} 个点")

    print("正在绘制多样本图表...")

    # 绘制并保存
    plot_multiple_trajectories(
        trajectories_viz,
        start_point=start_point,
        end_point=end_point,
        title="CVAE 多样本轨迹预测 - 展示随机性",
        save_path="trajectory_multiple.png",
        show=False  # 改为 True 可以显示图表窗口
    )

    print("\n" + "=" * 80)
    print("✓ 示例运行完成！")
    print("✓ 图表已保存:")
    print("  - trajectory_single.png (单条轨迹)")
    print("  - trajectory_multiple.png (多条轨迹)")
    print("=" * 80)
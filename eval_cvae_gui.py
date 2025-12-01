import tkinter as tk
import random
import time
import math
import torch
import numpy as np
from torch import nn

from cvae_model import TrajectoryPredictorCVAE

# --- 常量 (与模型和数据处理保持一致) ---
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
MAX_TRAJECTORY_LENGTH = 300


class CVAETrajectoryEvaluator:
    def __init__(self, master, model_path='cvae_trajectory_predictor.pth'):
        self.master = master
        master.title("CVAE 轨迹预测评估器")

        # --- 配置 ---
        self.CANVAS_WIDTH = CANVAS_WIDTH
        self.CANVAS_HEIGHT = CANVAS_HEIGHT
        self.POINT_RADIUS = 15
        self.SAMPLE_RATE_MS = 10

        self.HUMAN_COLOR = "#00FF00"  # 真人轨迹颜色 (亮绿色)
        self.MODEL_COLOR = "#FF00FF"  # 模型预测轨迹颜色 (洋红色)
        self.START_COLOR = "#0088FF"  # 起点颜色 (亮蓝色)
        self.END_COLOR = "#FF4444"  # 终点颜色 (亮红色)
        self.MULTI_SAMPLE_COLORS = ["#FF00FF", "#FF88FF", "#FF00AA", "#AA00FF", "#FF44FF"]  # 多样本颜色

        # --- 模型加载 ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 实例化 CVAE 模型
        self.model = TrajectoryPredictorCVAE(max_len=MAX_TRAJECTORY_LENGTH)

        # 加载训练好的权重
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"CVAE 模型加载成功: {model_path}")
        except FileNotFoundError:
            print(f"警告: 未找到模型文件 '{model_path}'，将使用未训练的模型")
            self.model.to(self.device)
            self.model.eval()

        # --- 状态变量 ---
        self.step = 0  # 0: 等待点击起点; 1: 正在收集/等待点击终点; 2: 轨迹显示/评估
        self.start_x, self.start_y = 0, 0
        self.end_x, self.end_y = 0, 0
        self.human_trajectory = []
        self.start_time = 0
        self.record_job = None
        self.num_samples = 1  # 采样数量

        # --- GUI 组件 ---
        self.canvas = tk.Canvas(master, width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT, bg='black')
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_click)

        # 状态标签
        self.status_label = tk.Label(master, text="", fg='white', bg='black', font=('Arial', 11))
        self.status_label.pack(pady=5)

        # 统计信息标签
        self.stats_label = tk.Label(master, text="", fg='cyan', bg='black', font=('Courier', 10), justify='left')
        self.stats_label.pack(pady=5)

        # 控制面板
        control_frame = tk.Frame(master, bg='black')
        control_frame.pack(pady=5)

        # 采样数量控制
        tk.Label(control_frame, text="生成样本数:", fg='white', bg='black').pack(side=tk.LEFT, padx=5)
        self.sample_var = tk.IntVar(value=1)
        sample_spinbox = tk.Spinbox(control_frame, from_=1, to=5, textvariable=self.sample_var,
                                     width=5, command=self.on_sample_change)
        sample_spinbox.pack(side=tk.LEFT, padx=5)

        # 重新生成按钮
        regenerate_btn = tk.Button(control_frame, text="重新生成任务", command=self.generate_new_trial)
        regenerate_btn.pack(side=tk.LEFT, padx=10)

        # 仅预测按钮（不需要画真人轨迹）
        predict_btn = tk.Button(control_frame, text="仅查看预测", command=self.predict_only_mode)
        predict_btn.pack(side=tk.LEFT, padx=10)

        self.generate_new_trial()

    # --- 辅助方法 ---
    def _draw_point(self, x, y, r, color, tag):
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, tags=tag, outline='white')

    def on_sample_change(self):
        """采样数量改变时的回调"""
        self.num_samples = self.sample_var.get()

    def generate_new_trial(self):
        """生成新的随机任务，并绘制起点。"""
        margin = self.POINT_RADIUS * 2 + 10

        # 1. 生成起点和终点坐标
        self.start_x = random.randint(margin, self.CANVAS_WIDTH - margin)
        self.start_y = random.randint(margin, self.CANVAS_HEIGHT - margin)

        min_dist = 200
        while True:
            self.end_x = random.randint(margin, self.CANVAS_WIDTH - margin)
            self.end_y = random.randint(margin, self.CANVAS_HEIGHT - margin)
            self.distance_euclidean = math.dist((self.start_x, self.start_y), (self.end_x, self.end_y))
            if self.distance_euclidean > min_dist:
                break

        # 2. 清理画布
        self.canvas.delete("all")

        # 3. 绘制起点 (蓝色，等待点击)
        self._draw_point(self.start_x, self.start_y, self.POINT_RADIUS, self.START_COLOR, "start_point")
        self.status_label.config(text=f"状态: 请点击蓝色起点 ({self.start_x}, {self.start_y})", fg='blue')
        self.stats_label.config(text="")
        self.step = 0
        self.num_samples = self.sample_var.get()

    def predict_only_mode(self):
        """仅预测模式：不收集真人轨迹，直接显示模型预测"""
        self.canvas.delete("trajectory", "human_traj", "model_traj")

        # 绘制终点
        self._draw_point(self.end_x, self.end_y, self.POINT_RADIUS, self.END_COLOR, "end_point")

        # 预测轨迹
        if self.num_samples == 1:
            model_trajectory = self.predict_trajectory()
            self._draw_trajectory(model_trajectory, self.MODEL_COLOR, "model_traj", width=3)

            stats_text = f"CVAE 模型预测 (单次采样)\n"
            stats_text += f"轨迹点数: {len(model_trajectory)}\n"
            stats_text += f"起点: ({self.start_x:.0f}, {self.start_y:.0f})\n"
            stats_text += f"终点: ({self.end_x:.0f}, {self.end_y:.0f})"
        else:
            # 多样本模式
            model_trajectories = self.predict_multiple_trajectories(self.num_samples)

            stats_text = f"CVAE 模型预测 ({self.num_samples} 个样本)\n"
            for idx, traj in enumerate(model_trajectories):
                color = self.MULTI_SAMPLE_COLORS[idx % len(self.MULTI_SAMPLE_COLORS)]
                self._draw_trajectory(traj, color, f"model_traj_{idx}", width=2)
                stats_text += f"样本 {idx+1}: {len(traj)} 个点\n"

        self.stats_label.config(text=stats_text)
        self.status_label.config(text="状态: CVAE 预测完成 (蓝色=起点, 红色=终点)", fg='magenta')

    # --- 模型预测逻辑 ---
    def predict_trajectory(self):
        """使用 CVAE 模型单次预测完整轨迹"""
        # 1. 归一化函数
        def normalize_x(x):
            return x / self.CANVAS_WIDTH

        def normalize_y(y):
            return y / self.CANVAS_HEIGHT

        max_dist = math.sqrt(self.CANVAS_WIDTH ** 2 + self.CANVAS_HEIGHT ** 2)

        def normalize_dist(d):
            return d / max_dist

        # 2. 准备任务上下文
        task_context_np = np.array([
            normalize_x(self.start_x), normalize_y(self.start_y),
            normalize_x(self.end_x), normalize_y(self.end_y),
            normalize_dist(self.distance_euclidean)
        ], dtype=np.float32)

        # 3. 转换为 PyTorch 张量
        task_context_tensor = torch.from_numpy(task_context_np).unsqueeze(0).to(self.device)

        # 4. CVAE 模型预测（使用先验网络）
        with torch.no_grad():
            reconstruction, mu, logvar, predicted_length_ratio = self.model(task_context_tensor)
            prediction_tensor = reconstruction.cpu().squeeze(0).numpy()
            pred_length_ratio = predicted_length_ratio.cpu().squeeze(0).item()

        # 5. 根据预测的长度比例确定实际要使用的点数
        predicted_length = int(pred_length_ratio * MAX_TRAJECTORY_LENGTH)
        predicted_length = min(predicted_length, len(prediction_tensor))

        # 6. 提取预测点并转换为绝对坐标
        raw_trajectory = [(self.start_x, self.start_y)]  # 起点

        for i in range(predicted_length):
            # 提取特征 [X, Y, Time, Direction, Speed]
            rel_x_norm = prediction_tensor[i, 0]
            rel_y_norm = prediction_tensor[i, 1]

            # 反归一化
            rel_x = rel_x_norm * self.CANVAS_WIDTH
            rel_y = rel_y_norm * self.CANVAS_HEIGHT

            # 相对坐标转绝对坐标（相对于起点）
            abs_x = rel_x + self.start_x
            abs_y = rel_y + self.start_y

            # 检查坐标是否合理
            if abs_x < -50 or abs_x > self.CANVAS_WIDTH + 50 or abs_y < -50 or abs_y > self.CANVAS_HEIGHT + 50:
                break

            raw_trajectory.append((abs_x, abs_y))

        # 7. 平滑处理：移动平均滤波
        smoothed_trajectory = self._smooth_trajectory(raw_trajectory, window_size=5)

        # 8. 采样降低点数
        sampled_trajectory = self._sample_trajectory(smoothed_trajectory, target_sample_rate=2)

        # 9. 线性变换对齐终点（保持轨迹形状，精确映射到目标起点和终点）
        if len(sampled_trajectory) >= 2:
            sampled_trajectory = self._align_trajectory_to_endpoints(
                sampled_trajectory,
                target_start=(self.start_x, self.start_y),
                target_end=(self.end_x, self.end_y)
            )

        return sampled_trajectory

    def predict_multiple_trajectories(self, num_samples):
        """使用 CVAE 模型生成多个不同的轨迹样本（展示多样性）"""
        # 1. 归一化函数
        def normalize_x(x):
            return x / self.CANVAS_WIDTH

        def normalize_y(y):
            return y / self.CANVAS_HEIGHT

        max_dist = math.sqrt(self.CANVAS_WIDTH ** 2 + self.CANVAS_HEIGHT ** 2)

        def normalize_dist(d):
            return d / max_dist

        # 2. 准备任务上下文
        task_context_np = np.array([
            normalize_x(self.start_x), normalize_y(self.start_y),
            normalize_x(self.end_x), normalize_y(self.end_y),
            normalize_dist(self.distance_euclidean)
        ], dtype=np.float32)

        # 3. 转换为 PyTorch 张量
        task_context_tensor = torch.from_numpy(task_context_np).unsqueeze(0).to(self.device)

        # 4. 使用 CVAE 的 sample 方法生成多个样本
        with torch.no_grad():
            trajectories_tensor = self.model.sample(task_context_tensor, num_samples=num_samples)
            # trajectories_tensor: (1, num_samples, max_len, 5)
            trajectories_tensor = trajectories_tensor.squeeze(0).cpu().numpy()  # (num_samples, max_len, 5)

            # 同时获取长度预测
            _, _, _, predicted_length_ratio = self.model(task_context_tensor)
            pred_length_ratio = predicted_length_ratio.cpu().squeeze(0).item()

        # 5. 处理每个样本
        all_trajectories = []
        predicted_length = int(pred_length_ratio * MAX_TRAJECTORY_LENGTH)

        for sample_idx in range(num_samples):
            prediction_tensor = trajectories_tensor[sample_idx]  # (max_len, 5)
            predicted_length_sample = min(predicted_length, len(prediction_tensor))

            # 提取预测点并转换为绝对坐标
            raw_trajectory = [(self.start_x, self.start_y)]  # 起点

            for i in range(predicted_length_sample):
                rel_x_norm = prediction_tensor[i, 0]
                rel_y_norm = prediction_tensor[i, 1]

                rel_x = rel_x_norm * self.CANVAS_WIDTH
                rel_y = rel_y_norm * self.CANVAS_HEIGHT

                abs_x = rel_x + self.start_x
                abs_y = rel_y + self.start_y

                if abs_x < -50 or abs_x > self.CANVAS_WIDTH + 50 or abs_y < -50 or abs_y > self.CANVAS_HEIGHT + 50:
                    break

                raw_trajectory.append((abs_x, abs_y))

            # 平滑和采样
            smoothed_trajectory = self._smooth_trajectory(raw_trajectory, window_size=5)
            sampled_trajectory = self._sample_trajectory(smoothed_trajectory, target_sample_rate=2)

            # 线性变换对齐终点
            if len(sampled_trajectory) >= 2:
                sampled_trajectory = self._align_trajectory_to_endpoints(
                    sampled_trajectory,
                    target_start=(self.start_x, self.start_y),
                    target_end=(self.end_x, self.end_y)
                )

            all_trajectories.append(sampled_trajectory)

        return all_trajectories

    def _smooth_trajectory(self, trajectory, window_size=5):
        """使用移动平均对轨迹进行平滑处理"""
        if len(trajectory) < window_size:
            return trajectory

        smoothed = [trajectory[0]]  # 保持起点不变

        # 对中间点进行平滑
        for i in range(1, len(trajectory) - 1):
            # 确定窗口范围
            half_window = window_size // 2
            start_idx = max(0, i - half_window)
            end_idx = min(len(trajectory), i + half_window + 1)

            # 计算窗口内点的平均值
            window_points = trajectory[start_idx:end_idx]
            avg_x = sum(p[0] for p in window_points) / len(window_points)
            avg_y = sum(p[1] for p in window_points) / len(window_points)

            smoothed.append((avg_x, avg_y))

        smoothed.append(trajectory[-1])  # 保持终点不变
        return smoothed

    def _sample_trajectory(self, trajectory, target_sample_rate=2):
        """对轨迹进行采样，减少点数"""
        if len(trajectory) <= 2:
            return trajectory

        sampled = [trajectory[0]]  # 保持起点

        # 按采样率采样中间点
        for i in range(target_sample_rate, len(trajectory) - 1, target_sample_rate):
            sampled.append(trajectory[i])

        # 确保终点被包含
        if trajectory[-1] not in sampled:
            sampled.append(trajectory[-1])

        return sampled

    def calculate_losses(self, model_traj, human_traj):
        """计算模型轨迹和真实轨迹之间的各种损失"""
        # 确保至少有2个点
        if len(model_traj) < 2 or len(human_traj) < 2:
            return {
                'position': 0.0,
                'endpoint': 0.0,
                'length_diff': 0,
                'avg_distance': 0.0
            }

        # 1. 终点损失
        model_endpoint = model_traj[-1]
        human_endpoint = human_traj[-1]
        endpoint_loss = math.dist(model_endpoint, human_endpoint)

        # 2. 位置损失 (采样点匹配)
        total_position_error = 0.0
        num_samples = min(len(model_traj), len(human_traj), 50)

        for i in range(num_samples):
            model_idx = int(i * (len(model_traj) - 1) / (num_samples - 1)) if num_samples > 1 else 0
            human_idx = int(i * (len(human_traj) - 1) / (num_samples - 1)) if num_samples > 1 else 0

            model_point = model_traj[model_idx]
            human_point = human_traj[human_idx]

            total_position_error += math.dist(model_point, human_point)

        avg_position_loss = total_position_error / num_samples if num_samples > 0 else 0.0

        # 3. 长度差异
        length_diff = abs(len(model_traj) - len(human_traj))

        # 4. 平均点距离（整体偏差）
        model_center_x = sum(p[0] for p in model_traj) / len(model_traj)
        model_center_y = sum(p[1] for p in model_traj) / len(model_traj)
        human_center_x = sum(p[0] for p in human_traj) / len(human_traj)
        human_center_y = sum(p[1] for p in human_traj) / len(human_traj)

        avg_distance = math.dist((model_center_x, model_center_y), (human_center_x, human_center_y))

        return {
            'position': avg_position_loss,
            'endpoint': endpoint_loss,
            'length_diff': length_diff,
            'avg_distance': avg_distance
        }

    def _draw_trajectory(self, trajectory_list, color, tag, width=3):
        """绘制轨迹线"""
        if len(trajectory_list) < 2:
            return 0

        points = []
        total_length = 0
        for i in range(len(trajectory_list)):
            x, y = trajectory_list[i]
            points.append(x)
            points.append(y)

            if i > 0:
                total_length += math.dist(trajectory_list[i - 1], trajectory_list[i])

        self.canvas.create_line(points, fill=color, tags=tag, width=width, smooth=True)
        return total_length

    # --- 事件处理 ---
    def on_click(self, event):
        click_x, click_y = event.x, event.y

        dist_to_start = math.dist((click_x, click_y), (self.start_x, self.start_y))
        dist_to_end = math.dist((click_x, click_y), (self.end_x, self.end_y))

        if self.step == 0:
            # 状态 0: 等待点击起点
            if dist_to_start <= self.POINT_RADIUS:
                self.step = 1
                self.start_time = time.time()
                self.human_trajectory = [(self.start_x, self.start_y)]  # 记录起点

                # 绘制终点
                self._draw_point(self.end_x, self.end_y, self.POINT_RADIUS, self.END_COLOR, "end_point")
                self.status_label.config(text="状态: 正在收集... 移动鼠标并点击终点 (红色)", fg='red')
                self._record_trajectory()

        elif self.step == 1:
            # 状态 1: 正在收集轨迹，等待点击终点
            if dist_to_end <= self.POINT_RADIUS:
                self.step = 2

                if self.record_job:
                    self.master.after_cancel(self.record_job)

                # 记录最后点击点
                self.human_trajectory.append((click_x, click_y))

                # 绘制真人轨迹
                self._draw_trajectory(self.human_trajectory, self.HUMAN_COLOR, "human_traj", width=3)

                # 预测轨迹
                if self.num_samples == 1:
                    model_trajectory = self.predict_trajectory()
                    self._draw_trajectory(model_trajectory, self.MODEL_COLOR, "model_traj", width=3)

                    # 计算损失
                    losses = self.calculate_losses(model_trajectory, self.human_trajectory)

                    # 显示统计信息
                    stats_text = f"CVAE 评估结果:\n"
                    stats_text += f"真人轨迹: {len(self.human_trajectory)} 点 (绿色)\n"
                    stats_text += f"CVAE 预测: {len(model_trajectory)} 点 (洋红色)\n"
                    stats_text += f"点数差异: {losses['length_diff']}\n"
                    stats_text += f"终点误差: {losses['endpoint']:.2f} px\n"
                    stats_text += f"平均位置误差: {losses['position']:.2f} px\n"
                    stats_text += f"轨迹中心偏差: {losses['avg_distance']:.2f} px"

                    self.stats_label.config(text=stats_text)
                else:
                    # 多样本模式
                    model_trajectories = self.predict_multiple_trajectories(self.num_samples)

                    stats_text = f"CVAE 多样本评估:\n"
                    stats_text += f"真人轨迹: {len(self.human_trajectory)} 点 (绿色)\n"

                    for idx, traj in enumerate(model_trajectories):
                        color = self.MULTI_SAMPLE_COLORS[idx % len(self.MULTI_SAMPLE_COLORS)]
                        self._draw_trajectory(traj, color, f"model_traj_{idx}", width=2)
                        losses = self.calculate_losses(traj, self.human_trajectory)
                        stats_text += f"样本 {idx+1}: {len(traj)} 点, 终点误差: {losses['endpoint']:.1f}px\n"

                    self.stats_label.config(text=stats_text)

                self.status_label.config(text="状态: 评估完成！点击 '重新生成任务' 开始下一次", fg='green')

        elif self.step == 2:
            # 状态 2: 评估完成，等待重新开始
            pass

    def _record_trajectory(self):
        """定时记录鼠标位置"""
        if self.step != 1:
            return

        # 获取鼠标位置
        x = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx()
        y = self.canvas.winfo_pointery() - self.canvas.winfo_rooty()

        # 检查是否在画布内
        if 0 <= x <= self.CANVAS_WIDTH and 0 <= y <= self.CANVAS_HEIGHT:
            # 避免记录重复点
            if len(self.human_trajectory) == 0 or math.dist((x, y), self.human_trajectory[-1]) > 1:
                self.human_trajectory.append((x, y))

        # 继续记录
        self.record_job = self.master.after(self.SAMPLE_RATE_MS, self._record_trajectory)

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


def main():
    root = tk.Tk()
    root.configure(bg='black')

    # 创建评估器（默认加载 cvae_trajectory_predictor.pth）
    evaluator = CVAETrajectoryEvaluator(root, model_path='cvae_trajectory_predictor.pth')

    root.mainloop()


if __name__ == '__main__':
    main()
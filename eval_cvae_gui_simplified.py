"""
简化版 CVAE 轨迹预测评估器
直接使用 TrajectoryPredictor 类，避免重复代码
"""

import tkinter as tk
import random
import time
import math

from predict_trajectory import TrajectoryPredictor

# --- 常量 ---
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600


class CVAETrajectoryEvaluatorSimplified:
    def __init__(self, master, model_path='cvae_trajectory_predictor.pth'):
        self.master = master
        master.title("CVAE 轨迹预测评估器 (简化版)")

        # --- 配置 ---
        self.CANVAS_WIDTH = CANVAS_WIDTH
        self.CANVAS_HEIGHT = CANVAS_HEIGHT
        self.POINT_RADIUS = 15
        self.SAMPLE_RATE_MS = 10

        self.HUMAN_COLOR = "#00FF00"  # 真人轨迹颜色 (亮绿色)
        self.MODEL_COLOR = "#FF00FF"  # 模型预测轨迹颜色 (洋红色)
        self.START_COLOR = "#0088FF"  # 起点颜色 (亮蓝色)
        self.END_COLOR = "#FF4444"  # 终点颜色 (亮红色)
        self.MULTI_SAMPLE_COLORS = ["#FF00FF", "#FF88FF", "#FF00AA", "#AA00FF", "#FF44FF"]

        # --- 使用 TrajectoryPredictor 封装所有预测逻辑 ---
        self.predictor = TrajectoryPredictor(
            model_path=model_path,
            canvas_width=CANVAS_WIDTH,
            canvas_height=CANVAS_HEIGHT
        )

        # --- 状态变量 ---
        self.step = 0
        self.start_x, self.start_y = 0, 0
        self.end_x, self.end_y = 0, 0
        self.human_trajectory = []
        self.start_time = 0
        self.record_job = None
        self.num_samples = 1

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

        # 仅预测按钮
        predict_btn = tk.Button(control_frame, text="仅查看预测", command=self.predict_only_mode)
        predict_btn.pack(side=tk.LEFT, padx=10)

        self.generate_new_trial()

    def _draw_point(self, x, y, r, color, tag):
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, tags=tag, outline='white')

    def on_sample_change(self):
        self.num_samples = self.sample_var.get()

    def generate_new_trial(self):
        """生成新的随机任务"""
        margin = self.POINT_RADIUS * 2 + 10

        self.start_x = random.randint(margin, self.CANVAS_WIDTH - margin)
        self.start_y = random.randint(margin, self.CANVAS_HEIGHT - margin)

        min_dist = 200
        while True:
            self.end_x = random.randint(margin, self.CANVAS_WIDTH - margin)
            self.end_y = random.randint(margin, self.CANVAS_HEIGHT - margin)
            distance = math.dist((self.start_x, self.start_y), (self.end_x, self.end_y))
            if distance > min_dist:
                break

        self.canvas.delete("all")
        self._draw_point(self.start_x, self.start_y, self.POINT_RADIUS, self.START_COLOR, "start_point")
        self.status_label.config(text=f"状态: 请点击蓝色起点 ({self.start_x}, {self.start_y})", fg='blue')
        self.stats_label.config(text="")
        self.step = 0
        self.num_samples = self.sample_var.get()

    def predict_only_mode(self):
        """仅预测模式：直接显示模型预测"""
        self.canvas.delete("trajectory", "human_traj", "model_traj")
        self._draw_point(self.end_x, self.end_y, self.POINT_RADIUS, self.END_COLOR, "end_point")

        if self.num_samples == 1:
            # 使用 TrajectoryPredictor 预测单条轨迹
            model_trajectory = self.predictor.predict(
                self.start_x, self.start_y,
                self.end_x, self.end_y
            )
            self._draw_trajectory(model_trajectory, self.MODEL_COLOR, "model_traj", width=3)

            stats_text = f"CVAE 模型预测 (单次采样)\n"
            stats_text += f"轨迹点数: {len(model_trajectory)}\n"
            stats_text += f"起点: ({self.start_x:.0f}, {self.start_y:.0f})\n"
            stats_text += f"终点: ({self.end_x:.0f}, {self.end_y:.0f})"
        else:
            # 使用 TrajectoryPredictor 预测多条轨迹
            model_trajectories = self.predictor.predict_multiple(
                self.start_x, self.start_y,
                self.end_x, self.end_y,
                num_samples=self.num_samples
            )

            stats_text = f"CVAE 模型预测 ({self.num_samples} 个样本)\n"
            for idx, traj in enumerate(model_trajectories):
                color = self.MULTI_SAMPLE_COLORS[idx % len(self.MULTI_SAMPLE_COLORS)]
                self._draw_trajectory(traj, color, f"model_traj_{idx}", width=2)
                stats_text += f"样本 {idx+1}: {len(traj)} 个点\n"

        self.stats_label.config(text=stats_text)
        self.status_label.config(text="状态: CVAE 预测完成 (蓝色=起点, 红色=终点)", fg='magenta')

    def calculate_losses(self, model_traj, human_traj):
        """计算模型轨迹和真实轨迹之间的各种损失"""
        if len(model_traj) < 2 or len(human_traj) < 2:
            return {
                'position': 0.0,
                'endpoint': 0.0,
                'length_diff': 0,
                'avg_distance': 0.0
            }

        # 终点损失
        model_endpoint = model_traj[-1]
        human_endpoint = human_traj[-1]
        endpoint_loss = math.dist(model_endpoint, human_endpoint)

        # 位置损失 (采样点匹配)
        total_position_error = 0.0
        num_samples = min(len(model_traj), len(human_traj), 50)

        for i in range(num_samples):
            model_idx = int(i * (len(model_traj) - 1) / (num_samples - 1)) if num_samples > 1 else 0
            human_idx = int(i * (len(human_traj) - 1) / (num_samples - 1)) if num_samples > 1 else 0

            model_point = model_traj[model_idx]
            human_point = human_traj[human_idx]

            total_position_error += math.dist(model_point, human_point)

        avg_position_loss = total_position_error / num_samples if num_samples > 0 else 0.0

        # 长度差异
        length_diff = abs(len(model_traj) - len(human_traj))

        # 平均点距离
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

    def on_click(self, event):
        click_x, click_y = event.x, event.y

        dist_to_start = math.dist((click_x, click_y), (self.start_x, self.start_y))
        dist_to_end = math.dist((click_x, click_y), (self.end_x, self.end_y))

        if self.step == 0:
            # 等待点击起点
            if dist_to_start <= self.POINT_RADIUS:
                self.step = 1
                self.start_time = time.time()
                self.human_trajectory = [(self.start_x, self.start_y)]

                self._draw_point(self.end_x, self.end_y, self.POINT_RADIUS, self.END_COLOR, "end_point")
                self.status_label.config(text="状态: 正在收集... 移动鼠标并点击终点 (红色)", fg='red')
                self._record_trajectory()

        elif self.step == 1:
            # 正在收集轨迹，等待点击终点
            if dist_to_end <= self.POINT_RADIUS:
                self.step = 2

                if self.record_job:
                    self.master.after_cancel(self.record_job)

                self.human_trajectory.append((click_x, click_y))
                self._draw_trajectory(self.human_trajectory, self.HUMAN_COLOR, "human_traj", width=3)

                # 使用 TrajectoryPredictor 预测
                if self.num_samples == 1:
                    model_trajectory = self.predictor.predict(
                        self.start_x, self.start_y,
                        self.end_x, self.end_y
                    )
                    self._draw_trajectory(model_trajectory, self.MODEL_COLOR, "model_traj", width=3)

                    losses = self.calculate_losses(model_trajectory, self.human_trajectory)

                    stats_text = f"CVAE 评估结果:\n"
                    stats_text += f"真人轨迹: {len(self.human_trajectory)} 点 (绿色)\n"
                    stats_text += f"CVAE 预测: {len(model_trajectory)} 点 (洋红色)\n"
                    stats_text += f"点数差异: {losses['length_diff']}\n"
                    stats_text += f"终点误差: {losses['endpoint']:.2f} px\n"
                    stats_text += f"平均位置误差: {losses['position']:.2f} px\n"
                    stats_text += f"轨迹中心偏差: {losses['avg_distance']:.2f} px"

                    self.stats_label.config(text=stats_text)
                else:
                    model_trajectories = self.predictor.predict_multiple(
                        self.start_x, self.start_y,
                        self.end_x, self.end_y,
                        num_samples=self.num_samples
                    )

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
            pass

    def _record_trajectory(self):
        """定时记录鼠标位置"""
        if self.step != 1:
            return

        x = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx()
        y = self.canvas.winfo_pointery() - self.canvas.winfo_rooty()

        if 0 <= x <= self.CANVAS_WIDTH and 0 <= y <= self.CANVAS_HEIGHT:
            if len(self.human_trajectory) == 0 or math.dist((x, y), self.human_trajectory[-1]) > 1:
                self.human_trajectory.append((x, y))

        self.record_job = self.master.after(self.SAMPLE_RATE_MS, self._record_trajectory)


def main():
    root = tk.Tk()
    root.configure(bg='black')

    evaluator = CVAETrajectoryEvaluatorSimplified(root, model_path='cvae_trajectory_predictor.pth')

    root.mainloop()


if __name__ == '__main__':
    main()
"""
可视化评估GUI - 对比人类轨迹和模型生成的轨迹
"""
import tkinter as tk
import torch
import numpy as np
import math
import time

from config.config import Config
from src.models.lstm_cvae import LSTMCVAE, TrajectoryLengthPredictor
from src.utils.trajectory_utils import TrajectoryExtractor


class TrajectoryEvaluationGUI:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("鼠标轨迹评估 - 人类 vs AI")

        # 设置窗口大小
        self.window_width = 1200
        self.window_height = 800
        self.root.geometry(f"{self.window_width}x{self.window_height}")

        # 加载模型
        print("加载模型...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = Config()
        self.load_model(model_path)

        # 轨迹提取器
        self.extractor = TrajectoryExtractor(self.norm_stats)

        # 状态变量
        self.state = 'waiting_start'  # waiting_start, recording_human, waiting_ai
        self.start_point = None
        self.end_point = None
        self.point_radius = 20

        # 人类轨迹数据
        self.human_trajectory = []
        self.recording_start_time = None
        self.last_position = None
        self.last_velocity = 0
        self.last_time = None

        # AI轨迹数据
        self.ai_trajectory = None

        # 创建UI
        self.create_ui()

        # 显示初始绿色点
        self.show_start_point()

    def load_model(self, model_path):
        """加载训练好的模型"""
        checkpoint = torch.load(model_path, map_location=self.device)

        self.model = LSTMCVAE(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.length_predictor = TrajectoryLengthPredictor(
            input_dim=4,
            hidden_dim=self.config.LENGTH_PREDICTOR_HIDDEN_DIM,
            max_length=self.config.MAX_TRAJECTORY_LENGTH
        ).to(self.device)
        self.length_predictor.load_state_dict(checkpoint['length_predictor_state_dict'])
        self.length_predictor.eval()

        self.norm_stats = checkpoint.get('norm_stats', None)

        print("模型加载成功！")

    def create_ui(self):
        """创建用户界面"""
        # 顶部控制面板
        control_frame = tk.Frame(self.root, bg='lightgray', height=60)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.status_label = tk.Label(
            control_frame,
            text="点击绿色点开始",
            font=('Arial', 14),
            bg='lightgray'
        )
        self.status_label.pack(side=tk.LEFT, padx=20, pady=10)

        self.reset_button = tk.Button(
            control_frame,
            text="重置",
            font=('Arial', 12),
            command=self.reset,
            width=10
        )
        self.reset_button.pack(side=tk.RIGHT, padx=20, pady=10)

        # 画布
        self.canvas = tk.Canvas(self.root, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def show_start_point(self):
        """显示绿色起点"""
        self.canvas.delete('all')

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        if width <= 1:
            self.root.update()
            width = self.canvas.winfo_width()
            height = self.canvas.winfo_height()

        # 随机生成起点
        margin = 100
        x = int(np.random.uniform(margin, width - margin))
        y = int(np.random.uniform(margin, height - margin))

        self.start_point = (x, y)

        # 绘制绿色点
        self.canvas.create_oval(
            x - self.point_radius, y - self.point_radius,
            x + self.point_radius, y + self.point_radius,
            fill='green', outline='black', width=2, tags='start_point'
        )

        self.canvas.create_text(
            x, y + self.point_radius + 20,
            text='起点',
            fill='black', font=('Arial', 12), tags='hint'
        )

        self.canvas.tag_bind('start_point', '<Button-1>', self.on_start_point_click)
        self.state = 'waiting_start'
        self.status_label.config(text="点击绿色点开始")

    def on_start_point_click(self, event):
        """点击绿色起点"""
        if self.state != 'waiting_start':
            return

        x, y = self.start_point
        distance = math.sqrt((event.x - x)**2 + (event.y - y)**2)
        if distance > self.point_radius:
            return

        self.show_end_point()

    def show_end_point(self):
        """显示红色终点并开始记录"""
        self.canvas.delete('hint')

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        margin = 100
        min_distance = 300

        # 生成终点
        attempts = 0
        while attempts < 100:
            end_x = int(np.random.uniform(margin, width - margin))
            end_y = int(np.random.uniform(margin, height - margin))

            distance = math.sqrt((end_x - self.start_point[0])**2 + (end_y - self.start_point[1])**2)

            if distance >= min_distance:
                break
            attempts += 1

        self.end_point = (end_x, end_y)

        # 绘制红色点
        self.canvas.create_oval(
            end_x - self.point_radius, end_y - self.point_radius,
            end_x + self.point_radius, end_y + self.point_radius,
            fill='red', outline='black', width=2, tags='end_point'
        )

        self.canvas.create_text(
            end_x, end_y + self.point_radius + 20,
            text='终点',
            fill='black', font=('Arial', 12), tags='hint'
        )

        self.canvas.tag_bind('end_point', '<Button-1>', self.on_end_point_click)

        # 开始记录人类轨迹
        self.state = 'recording_human'
        self.human_trajectory = []
        self.recording_start_time = time.time()
        self.last_position = self.start_point
        self.last_velocity = 0
        self.last_time = self.recording_start_time

        self.canvas.bind('<Motion>', self.on_mouse_move)
        self.status_label.config(text="移动鼠标到红色点...")

    def on_mouse_move(self, event):
        """记录鼠标移动"""
        if self.state != 'recording_human':
            return

        current_time = time.time()
        timestamp = current_time - self.recording_start_time
        current_pos = (event.x, event.y)

        # 计算距离
        dx = current_pos[0] - self.last_position[0]
        dy = current_pos[1] - self.last_position[1]
        distance = math.sqrt(dx**2 + dy**2)

        dt = current_time - self.last_time

        if dt < 0.001:
            return

        # 计算速度和加速度
        velocity = distance / dt if dt > 0 else 0
        acceleration = (velocity - self.last_velocity) / dt if dt > 0 else 0

        # 计算方向
        direction = math.degrees(math.atan2(dy, dx))
        if direction < 0:
            direction += 360

        # 记录数据
        data_point = {
            'start_x': self.last_position[0],
            'start_y': self.last_position[1],
            'end_x': self.end_point[0],
            'end_y': self.end_point[1],
            'current_x': current_pos[0],
            'current_y': current_pos[1],
            'velocity': velocity,
            'acceleration': acceleration,
            'direction': direction,
            'distance': distance,
            'timestamp': timestamp
        }

        self.human_trajectory.append(data_point)

        # 绘制轨迹
        if len(self.human_trajectory) > 1:
            prev = self.human_trajectory[-2]
            self.canvas.create_line(
                prev['current_x'], prev['current_y'],
                current_pos[0], current_pos[1],
                fill='blue', width=2, tags='human_trajectory'
            )

        self.last_position = current_pos
        self.last_velocity = velocity
        self.last_time = current_time

    def on_end_point_click(self, event):
        """点击红色终点"""
        if self.state != 'recording_human':
            return

        x, y = self.end_point
        distance = math.sqrt((event.x - x)**2 + (event.y - y)**2)
        if distance > self.point_radius:
            return

        # 停止记录
        self.state = 'waiting_ai'
        self.canvas.unbind('<Motion>')

        self.status_label.config(text="人类轨迹记录完成，生成AI轨迹中...")

        # 生成AI轨迹
        self.root.after(500, self.generate_ai_trajectory)

    def generate_ai_trajectory(self):
        """使用模型生成AI轨迹"""
        # 归一化起点和终点
        start_np = np.array([self.start_point], dtype=np.float32)
        end_np = np.array([self.end_point], dtype=np.float32)

        if self.norm_stats:
            start_np = (start_np - self.norm_stats['coord_mean']) / self.norm_stats['coord_std']
            end_np = (end_np - self.norm_stats['coord_mean']) / self.norm_stats['coord_std']

        start_tensor = torch.FloatTensor(start_np).to(self.device)
        end_tensor = torch.FloatTensor(end_np).to(self.device)

        # 预测轨迹长度
        with torch.no_grad():
            predicted_length = self.length_predictor(start_tensor, end_tensor)
            trajectory_length = int(predicted_length.item())
            trajectory_length = max(10, min(trajectory_length, self.config.MAX_TRAJECTORY_LENGTH))

        print(f"预测轨迹长度: {trajectory_length}")

        # 生成轨迹
        with torch.no_grad():
            ai_output = self.model.generate(start_tensor, end_tensor, trajectory_length)

        # 提取轨迹点
        ai_coords = self.extractor.extract_coordinates_only(ai_output)

        # 绘制AI轨迹
        self.draw_ai_trajectory(ai_coords)

        self.status_label.config(text="完成！蓝色=人类轨迹，红色=AI轨迹")

    def draw_ai_trajectory(self, coords):
        """绘制AI生成的轨迹"""
        for i in range(len(coords) - 1):
            self.canvas.create_line(
                coords[i][0], coords[i][1],
                coords[i+1][0], coords[i+1][1],
                fill='red', width=2, tags='ai_trajectory'
            )

        # 添加图例
        legend_x = 20
        legend_y = 20

        self.canvas.create_line(
            legend_x, legend_y, legend_x + 30, legend_y,
            fill='blue', width=3
        )
        self.canvas.create_text(
            legend_x + 40, legend_y,
            text='人类轨迹', anchor='w', font=('Arial', 10)
        )

        self.canvas.create_line(
            legend_x, legend_y + 25, legend_x + 30, legend_y + 25,
            fill='red', width=3
        )
        self.canvas.create_text(
            legend_x + 40, legend_y + 25,
            text='AI轨迹', anchor='w', font=('Arial', 10)
        )

    def reset(self):
        """重置"""
        self.state = 'waiting_start'
        self.human_trajectory = []
        self.ai_trajectory = None
        self.show_start_point()


def main():
    """主函数"""
    config = Config()

    # 检查模型文件是否存在
    import os
    if not os.path.exists(config.BEST_MODEL_PATH):
        print(f"错误: 找不到模型文件 {config.BEST_MODEL_PATH}")
        print("请先运行 train.py 训练模型")
        return

    root = tk.Tk()
    app = TrajectoryEvaluationGUI(root, config.BEST_MODEL_PATH)
    root.mainloop()


if __name__ == '__main__':
    main()
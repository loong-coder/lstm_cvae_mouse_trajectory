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
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

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
            text='终点（点击结束）',
            fill='black', font=('Arial', 12), tags='hint'
        )

        # 绑定点击事件到红色点
        self.canvas.tag_bind('end_point', '<Button-1>', self.on_end_point_click)

        # 同时绑定全局点击事件作为备选
        self.canvas.bind('<Button-1>', self.on_canvas_click)

        # 开始记录人类轨迹
        self.state = 'recording_human'
        self.human_trajectory = []
        self.recording_start_time = time.time()
        self.last_position = self.start_point
        self.last_velocity = 0
        self.last_time = self.recording_start_time

        self.canvas.bind('<Motion>', self.on_mouse_move)
        self.status_label.config(text="移动鼠标到红色点，然后点击红色点结束...")

        print(f"\n已显示终点: {self.end_point}")
        print(f"状态设置为: {self.state}")

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
        print(f"\n检测到终点点击事件!")
        print(f"  当前状态: {self.state}")
        print(f"  点击位置: ({event.x}, {event.y})")
        print(f"  终点位置: {self.end_point}")

        if self.state != 'recording_human':
            print(f"  状态不对，需要 'recording_human'，当前是 '{self.state}'")
            return

        x, y = self.end_point
        distance = math.sqrt((event.x - x)**2 + (event.y - y)**2)
        print(f"  点击距离圆心: {distance:.1f} 像素 (半径: {self.point_radius})")

        # 放宽点击判断：允许点击圆圈附近（增加到1.5倍半径）
        if distance > self.point_radius * 1.5:
            print(f"  距离太远，忽略点击")
            return

        print(f"  ✓ 点击有效！停止记录并生成AI轨迹...")

        # 停止记录
        self.state = 'waiting_ai'
        self.canvas.unbind('<Motion>')
        self.canvas.unbind('<Button-1>')  # 解绑全局点击

        self.status_label.config(text="人类轨迹记录完成，生成AI轨迹中...")

        # 生成AI轨迹
        self.root.after(500, self.generate_ai_trajectory)

    def on_canvas_click(self, event):
        """全局点击事件处理（用于检测点击终点）"""
        if self.state != 'recording_human':
            return

        # 检查是否点击在红色终点附近
        if self.end_point is None:
            return

        x, y = self.end_point
        distance = math.sqrt((event.x - x)**2 + (event.y - y)**2)

        # 如果点击在终点附近，调用终点点击处理
        if distance <= self.point_radius * 1.5:
            self.on_end_point_click(event)

    def generate_ai_trajectory(self):
        """使用模型生成AI轨迹"""
        try:
            print(f"\n开始生成AI轨迹...")
            print(f"起点: {self.start_point}, 终点: {self.end_point}")

            # 归一化起点和终点
            start_np = np.array([self.start_point], dtype=np.float32)
            end_np = np.array([self.end_point], dtype=np.float32)

            if self.norm_stats:
                print(f"归一化前: start={start_np}, end={end_np}")
                start_np = (start_np - self.norm_stats['coord_mean']) / self.norm_stats['coord_std']
                end_np = (end_np - self.norm_stats['coord_mean']) / self.norm_stats['coord_std']
                print(f"归一化后: start={start_np}, end={end_np}")

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
                print(f"生成轨迹形状: {ai_output.shape}")

            # 提取轨迹点
            ai_coords = self.extractor.extract_coordinates_only(ai_output)
            print(f"提取坐标数量: {len(ai_coords)}")
            print(f"坐标范围: x=[{ai_coords[:,0].min():.1f}, {ai_coords[:,0].max():.1f}], y=[{ai_coords[:,1].min():.1f}, {ai_coords[:,1].max():.1f}]")

            # 绘制AI轨迹
            self.draw_ai_trajectory(ai_coords)

            self.status_label.config(text="完成！蓝色=人类轨迹，红色=AI轨迹")
            print("AI轨迹生成完成！")

        except Exception as e:
            print(f"生成AI轨迹时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            self.status_label.config(text=f"错误: {str(e)}")

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
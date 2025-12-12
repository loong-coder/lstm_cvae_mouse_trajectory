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
from src.utils.trajectory_utils import TrajectoryExtractor, TrajectoryComparator


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

        # AI轨迹数据（支持多条）
        self.ai_trajectories = []  # 存储多条AI轨迹
        self.num_trajectories = tk.IntVar(value=3)  # 默认生成3条轨迹

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

        # 轨迹数量设置
        settings_frame = tk.Frame(control_frame, bg='lightgray')
        settings_frame.pack(side=tk.LEFT, padx=20)

        tk.Label(
            settings_frame,
            text="AI轨迹数量:",
            font=('Arial', 10),
            bg='lightgray'
        ).pack(side=tk.LEFT, padx=5)

        self.num_spinbox = tk.Spinbox(
            settings_frame,
            from_=1,
            to=10,
            textvariable=self.num_trajectories,
            width=5,
            font=('Arial', 10)
        )
        self.num_spinbox.pack(side=tk.LEFT, padx=5)

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

        # 标记停顿点（速度很低的点）
        if velocity < 50:  # 速度阈值：50像素/秒
            self.canvas.create_oval(
                current_pos[0] - 3, current_pos[1] - 3,
                current_pos[0] + 3, current_pos[1] + 3,
                fill='darkblue', outline='blue', width=1, tags='pause_point'
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
        """使用模型生成多条AI轨迹"""
        try:
            num_traj = self.num_trajectories.get()
            print(f"\n开始生成{num_traj}条AI轨迹...")
            print(f"起点: {self.start_point}, 终点: {self.end_point}")

            # 清空之前的轨迹
            self.ai_trajectories = []

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

            # 生成多条轨迹
            for i in range(num_traj):
                print(f"  生成第 {i+1}/{num_traj} 条轨迹...")
                with torch.no_grad():
                    ai_output = self.model.generate(start_tensor, end_tensor, trajectory_length)

                # 提取轨迹点
                ai_coords = self.extractor.extract_coordinates_only(ai_output)
                self.ai_trajectories.append(ai_coords)
                print(f"    提取坐标数量: {len(ai_coords)}")

            # 绘制所有AI轨迹
            self.draw_ai_trajectories(self.ai_trajectories)

            # 计算并显示对比指标
            self.compute_and_display_metrics(self.ai_trajectories)

            self.status_label.config(text=f"完成！蓝色=人类轨迹，其他颜色=AI轨迹(共{num_traj}条)")
            print(f"所有AI轨迹生成完成！")

        except Exception as e:
            print(f"生成AI轨迹时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            self.status_label.config(text=f"错误: {str(e)}")

    def compute_and_display_metrics(self, ai_trajectories_list):
        """计算并显示人类轨迹和多条AI轨迹的对比指标"""
        try:
            print(f"\n计算对比指标...")

            # 提取人类轨迹的坐标
            human_coords = np.array([[p['current_x'], p['current_y']] for p in self.human_trajectory])

            print(f"人类轨迹点数: {len(human_coords)}")

            #计算多条AI轨迹的平均指标
            dtw_distances = []
            frechet_distances = []
            ai_total_dists = []

            for i, ai_coords in enumerate(ai_trajectories_list):
                print(f"AI轨迹{i+1}点数: {len(ai_coords)}")

                # 计算DTW距离
                dtw_dist = TrajectoryComparator.compute_dtw_distance(human_coords, ai_coords)
                dtw_distances.append(dtw_dist)

                # 计算Fréchet距离
                frechet_dist = TrajectoryComparator.compute_frechet_distance(human_coords, ai_coords)
                frechet_distances.append(frechet_dist)

                # 计算轨迹长度
                ai_total_dist = np.sum(np.sqrt(np.sum(np.diff(ai_coords, axis=0)**2, axis=1)))
                ai_total_dists.append(ai_total_dist)

            # 计算平均值
            avg_dtw = np.mean(dtw_distances)
            avg_frechet = np.mean(frechet_distances)
            avg_ai_dist = np.mean(ai_total_dists)

            print(f"平均DTW距离: {avg_dtw:.2f}")
            print(f"平均Fréchet距离: {avg_frechet:.2f}")

            # 计算人类轨迹长度
            human_total_dist = sum([p['distance'] for p in self.human_trajectory])

            # 计算直线距离
            straight_dist = math.sqrt(
                (self.end_point[0] - self.start_point[0])**2 +
                (self.end_point[1] - self.start_point[1])**2
            )

            # 计算路径效率
            human_efficiency = straight_dist / human_total_dist if human_total_dist > 0 else 0
            avg_ai_efficiency = straight_dist / avg_ai_dist if avg_ai_dist > 0 else 0

            # 在画布右下角显示指标
            metrics_text = [
                "═══ 轨迹对比指标 ═══",
                f"AI轨迹数量: {len(ai_trajectories_list)}",
                f"平均DTW距离: {avg_dtw:.1f}",
                f"平均Fréchet距离: {avg_frechet:.1f}",
                "",
                f"人类轨迹点数: {len(human_coords)}",
                f"平均AI点数: {int(np.mean([len(t) for t in ai_trajectories_list]))}",
                "",
                f"人类路径长度: {human_total_dist:.1f}",
                f"平均AI路径长度: {avg_ai_dist:.1f}",
                "",
                f"直线距离: {straight_dist:.1f}",
                f"人类路径效率: {human_efficiency:.2%}",
                f"平均AI效率: {avg_ai_efficiency:.2%}"
            ]

            # 在画布右侧显示
            x_pos = self.canvas.winfo_width() - 260
            y_start = 100

            # 绘制白色背景
            self.canvas.create_rectangle(
                x_pos - 10, y_start - 10,
                x_pos + 240, y_start + len(metrics_text) * 18 + 10,
                fill='white', outline='black', width=2, tags='metrics'
            )

            # 绘制每行文本
            for i, line in enumerate(metrics_text):
                if line == "":
                    continue
                color = 'darkblue' if '═' in line else 'black'
                font = ('Arial', 10, 'bold') if '═' in line else ('Arial', 9)

                self.canvas.create_text(
                    x_pos, y_start + i * 18,
                    text=line,
                    anchor='w',
                    font=font,
                    fill=color,
                    tags='metrics'
                )

            print("指标显示完成！")

        except Exception as e:
            print(f"计算指标时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def draw_ai_trajectories(self, trajectories_list):
        """绘制多条AI生成的轨迹，使用不同颜色，并标记停顿点"""
        # 定义颜色列表
        colors = ['red', 'orange', 'purple', 'brown', 'pink',
                  'cyan', 'magenta', 'olive', 'navy', 'maroon']

        # 深色系列（用于停顿点）
        dark_colors = ['darkred', 'darkorange', 'purple', 'saddlebrown', 'hotpink',
                       'darkcyan', 'darkmagenta', 'darkolivegreen', 'darkblue', 'maroon']

        for traj_idx, coords in enumerate(trajectories_list):
            color = colors[traj_idx % len(colors)]
            dark_color = dark_colors[traj_idx % len(dark_colors)]

            # 绘制轨迹线
            for i in range(len(coords) - 1):
                self.canvas.create_line(
                    coords[i][0], coords[i][1],
                    coords[i+1][0], coords[i+1][1],
                    fill=color, width=2, tags='ai_trajectory'
                )

            # 标记停顿点（相邻点之间距离很小的点）
            for i in range(1, len(coords)):
                # 计算与前一个点的距离
                dist = np.sqrt((coords[i][0] - coords[i-1][0])**2 +
                              (coords[i][1] - coords[i-1][1])**2)

                # 如果距离小于阈值，标记为停顿点
                if dist < 5:  # 距离阈值：5像素
                    self.canvas.create_oval(
                        coords[i][0] - 3, coords[i][1] - 3,
                        coords[i][0] + 3, coords[i][1] + 3,
                        fill=dark_color, outline=color, width=1, tags='ai_pause_point'
                    )

        # 添加图例
        legend_x = 20
        legend_y = 20

        # 人类轨迹图例
        self.canvas.create_line(
            legend_x, legend_y, legend_x + 30, legend_y,
            fill='blue', width=3
        )
        self.canvas.create_text(
            legend_x + 40, legend_y,
            text='人类轨迹', anchor='w', font=('Arial', 10)
        )

        # AI轨迹图例（显示前几条）
        for i in range(min(len(trajectories_list), 5)):  # 最多显示5条图例
            color = colors[i % len(colors)]
            y_pos = legend_y + (i + 1) * 20
            self.canvas.create_line(
                legend_x, y_pos, legend_x + 30, y_pos,
                fill=color, width=3
            )
            self.canvas.create_text(
                legend_x + 40, y_pos,
                text=f'AI轨迹{i+1}', anchor='w', font=('Arial', 9)
            )

        # 如果轨迹太多，显示省略号
        if len(trajectories_list) > 5:
            self.canvas.create_text(
                legend_x + 40, legend_y + 6 * 20,
                text='...', anchor='w', font=('Arial', 9)
            )

        # 添加停顿点图例说明
        pause_y = legend_y + min(len(trajectories_list) + 1, 6) * 20 + 10
        self.canvas.create_oval(
            legend_x + 10, pause_y - 3,
            legend_x + 10 + 6, pause_y + 3,
            fill='darkblue', outline='blue', width=1
        )
        self.canvas.create_text(
            legend_x + 40, pause_y,
            text='停顿点', anchor='w', font=('Arial', 9), fill='gray'
        )

    def reset(self):
        """重置"""
        self.state = 'waiting_start'
        self.human_trajectory = []
        self.ai_trajectories = []
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
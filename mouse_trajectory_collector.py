import tkinter as tk
import random
import csv
import time
import os
import math


class MouseTrajectoryCollector:
    def __init__(self, master):
        self.master = master
        master.title("鼠标轨迹数据收集器 (黑色背景)")

        # --- 配置 ---
        self.CANVAS_WIDTH = 800
        self.CANVAS_HEIGHT = 600
        self.POINT_RADIUS = 15  # 目标点半径 (R)
        self.SAVE_FILE_PATH = "mouse_trajectories_black_bg.csv"
        self.SAMPLE_RATE_MS = 10  # 采样频率 (10ms 记录一次)
        self.TRAJECTORY_COLOR = "yellow"  # 用于绘制轨迹的颜色

        # --- 状态变量 ---
        # step: 0: 等待点击起点; 1: 正在收集/等待点击终点; 2: 轨迹显示/等待确认继续
        self.step = 0
        self.start_x, self.start_y = 0, 0
        self.end_x, self.end_y = 0, 0
        self.current_trajectory = []
        self.start_time = 0
        self.trial_id_counter = self._get_initial_trial_id()
        self.record_job = None

        # --- GUI 组件 ---
        self.canvas = tk.Canvas(master, width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT, bg='black')
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_click)

        self.status_label = tk.Label(master, text="", fg='white', bg='black')
        self.status_label.pack(pady=5)

        self._initialize_csv()
        # 初始状态：只绘制起点
        self.generate_new_trial(stage='start_only')

    # --- 辅助方法 (数据持久化) ---
    def _get_initial_trial_id(self):
        """读取现有文件，找到最大的 Trial ID，确保新的 ID 不重复。"""
        if not os.path.exists(self.SAVE_FILE_PATH):
            return 1

        max_id = 0
        try:
            with open(self.SAVE_FILE_PATH, mode='r', newline='') as file:
                reader = csv.reader(file)
                next(reader, None)
                for row in reader:
                    if row and row[0].isdigit():
                        max_id = max(max_id, int(row[0]))
            return max_id + 1
        except Exception:
            return 1

    def _initialize_csv(self):
        """初始化或检查 CSV 文件，确保表头存在。"""
        if not os.path.exists(self.SAVE_FILE_PATH):
            with open(self.SAVE_FILE_PATH, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    "Trial_ID",
                    "Start_X", "Start_Y",
                    "End_X", "End_Y",
                    "Target_R",
                    "Distance_Euclidean",
                    "Timestamp_ms",
                    "Mouse_X", "Mouse_Y"
                ])
            print(f"创建新的数据文件: {self.SAVE_FILE_PATH}")
        else:
            print(f"数据将追加到现有文件: {self.SAVE_FILE_PATH}")

    # --- 试次管理 ---

    def generate_new_trial(self, stage):
        """生成新的试次，并根据 stage 绘制相应的点。"""
        margin = self.POINT_RADIUS * 2 + 10

        if stage == 'start_only':
            # 1. 随机生成新的起点和终点坐标
            self.start_x = random.randint(margin, self.CANVAS_WIDTH - margin)
            self.start_y = random.randint(margin, self.CANVAS_HEIGHT - margin)

            min_dist = 200  # 最小距离要求
            while True:
                self.end_x = random.randint(margin, self.CANVAS_WIDTH - margin)
                self.end_y = random.randint(margin, self.CANVAS_HEIGHT - margin)
                distance = math.dist((self.start_x, self.start_y), (self.end_x, self.end_y))
                if distance > min_dist:
                    self.distance_euclidean = distance
                    break

            # 2. 清理画布
            self.canvas.delete("all")

            # 3. 绘制起点 (亮绿色，等待点击)
            self._draw_point(self.start_x, self.start_y, self.POINT_RADIUS, 'lime', "start_point_lime")
            self.status_label.config(
                text=f"状态: Trial {self.trial_id_counter} - 请点击亮绿色起点 ({self.start_x}, {self.start_y})",
                fg='lime')
            self.step = 0  # 重置为等待点击起点状态

        elif stage == 'collecting':
            # 绘制已点击的起点 (亮蓝色) 和终点 (亮红色)
            self.canvas.delete("start_point_lime")  # 移除绿色起点
            self._draw_point(self.start_x, self.start_y, self.POINT_RADIUS, 'blue', "start_point_blue")  # 绘制蓝色起点
            self._draw_point(self.end_x, self.end_y, self.POINT_RADIUS, 'red', "end_point")  # 绘制终点

    def _draw_point(self, x, y, r, color, tag):
        """在画布上绘制圆形目标点。"""
        # 使用白色描边，在黑色背景上更清晰
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, tags=tag, outline='white')

    def _draw_trajectory(self):
        """将当前收集到的轨迹点绘制在画布上。"""
        if len(self.current_trajectory) < 2:
            return

        # 提取轨迹点的坐标 (只取 X, Y)
        points = []
        for _, x, y in self.current_trajectory:
            points.append(x)
            points.append(y)

        # 使用 create_line 绘制轨迹，tag 为 'trajectory' 方便后续清除
        self.canvas.create_line(points, fill=self.TRAJECTORY_COLOR, tags='trajectory', width=2)

        # 重新绘制起点和终点，确保它们在轨迹之上，不被线条遮挡
        self._draw_point(self.start_x, self.start_y, self.POINT_RADIUS, 'blue', "start_point_blue")
        self._draw_point(self.end_x, self.end_y, self.POINT_RADIUS, 'red', "end_point")

    # --- 事件处理 ---

    def on_click(self, event):
        """处理鼠标点击事件。"""
        click_x, click_y = event.x, event.y

        dist_to_start = math.dist((click_x, click_y), (self.start_x, self.start_y))
        dist_to_end = math.dist((click_x, click_y), (self.end_x, self.end_y))

        if self.step == 0:
            # 状态 0: 等待点击起点
            if dist_to_start <= self.POINT_RADIUS:
                # 点击起点成功，进入状态 1
                self.step = 1
                self.start_time = time.time()
                self.current_trajectory = []

                # 绘制已点击的起点和终点
                self.generate_new_trial(stage='collecting')

                self.status_label.config(text="状态: 正在收集... 移动鼠标并点击终点 (亮红色)", fg='red')
                self._record_trajectory()

        elif self.step == 1:
            # 状态 1: 正在收集轨迹，等待点击终点
            if dist_to_end <= self.POINT_RADIUS:
                # 点击终点成功，进入轨迹显示状态
                self.step = 2

                if self.record_job:
                    self.master.after_cancel(self.record_job)

                # 记录最后点击点
                relative_timestamp_ms = int((time.time() - self.start_time) * 1000)
                self.current_trajectory.append((relative_timestamp_ms, click_x, click_y))

                # 绘制轨迹，起点和终点保持显示
                self._draw_trajectory()

                self.status_label.config(
                    text=f"状态: Trial {self.trial_id_counter} 完成！轨迹已绘制 (黄色)。请再次点击**屏幕任意位置**继续。",
                    fg='yellow')

        elif self.step == 2:
            # 状态 2: 轨迹显示/等待用户确认继续

            # 1. 保存数据
            self._save_data()

            # 2. 准备下一个试次
            self.trial_id_counter += 1
            self.generate_new_trial(stage='start_only')  # 清除所有并绘制下一个起点
            self.step = 0  # 切换回等待点击起点状态

    def _record_trajectory(self):
        """周期性地记录鼠标的当前位置和时间戳。"""
        if self.step == 1:
            try:
                # 获取鼠标在画布上的相对坐标
                x = self.master.winfo_pointerx() - self.master.winfo_rootx() - self.canvas.winfo_x()
                y = self.master.winfo_pointery() - self.master.winfo_rooty() - self.canvas.winfo_y()
            except Exception:
                self.record_job = self.master.after(self.SAMPLE_RATE_MS, self._record_trajectory)
                return

            x = max(0, min(x, self.CANVAS_WIDTH))
            y = max(0, min(y, self.CANVAS_HEIGHT))

            relative_timestamp_ms = int((time.time() - self.start_time) * 1000)

            self.current_trajectory.append((relative_timestamp_ms, x, y))

            self.record_job = self.master.after(self.SAMPLE_RATE_MS, self._record_trajectory)

    # --- 数据保存 ---

    def _save_data(self):
        """将当前试次的轨迹数据和参数保存到 CSV 文件中。"""
        if not self.current_trajectory:
            self.status_label.config(text="状态: 警告 - 轨迹为空，未保存。", fg='orange')
            return

        total_time = self.current_trajectory[-1][0] if self.current_trajectory else 0

        with open(self.SAVE_FILE_PATH, mode='a', newline='') as file:
            writer = csv.writer(file)

            for timestamp, x, y in self.current_trajectory:
                writer.writerow([
                    self.trial_id_counter,
                    self.start_x, self.start_y,
                    self.end_x, self.end_y,
                    self.POINT_RADIUS,
                    f"{self.distance_euclidean:.2f}",
                    timestamp,
                    x, y
                ])

        print(f"Trial {self.trial_id_counter} 收集成功！")


# --- 主程序 ---
if __name__ == "__main__":
    root = tk.Tk()
    root.configure(bg='black')
    app = MouseTrajectoryCollector(root)
    root.mainloop()
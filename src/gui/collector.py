"""
鼠标轨迹收集程序
功能：收集鼠标从起点到终点的移动轨迹数据
"""
import tkinter as tk
import csv
import os
import time
import math
import random
from datetime import datetime


class MouseTrajectoryCollector:
    def __init__(self, root):
        self.root = root
        self.root.title("鼠标轨迹收集器")

        # 设置全屏
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg='black')

        # 绑定ESC键退出
        self.root.bind('<Escape>', lambda e: self.root.quit())

        # 创建画布
        self.canvas = tk.Canvas(self.root, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 数据文件
        self.data_file = 'mouse_trajectories.csv'
        self.group_id = self.load_last_group_id() + 1

        # 状态变量
        self.state = 'waiting_start'  # waiting_start, recording, waiting_end
        self.start_point = None
        self.end_point = None
        self.trajectory_data = []
        self.sequence_id = 0
        self.recording_start_time = None
        self.last_position = None
        self.last_velocity = 0
        self.last_time = None

        # 圆点大小
        self.point_radius = 20

        # 初始化CSV文件
        self.initialize_csv()

        # 显示初始绿色点
        self.show_start_point()

    def load_last_group_id(self):
        """从CSV文件加载最后一个组ID"""
        if not os.path.exists(self.data_file):
            return 0

        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                last_group_id = 0
                for row in reader:
                    try:
                        group_id = int(row['group_id'])
                        if group_id > last_group_id:
                            last_group_id = group_id
                    except (ValueError, KeyError):
                        continue
                return last_group_id
        except Exception as e:
            print(f"读取文件错误: {e}")
            return 0

    def initialize_csv(self):
        """初始化CSV文件（如果不存在）"""
        if not os.path.exists(self.data_file):
            with open(self.data_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'group_id', 'sequence_id',
                    'start_x', 'start_y',
                    'end_x', 'end_y',
                    'current_x', 'current_y',
                    'velocity', 'acceleration', 'direction',
                    'timestamp'
                ])

    def show_start_point(self):
        """显示绿色起点"""
        self.canvas.delete('all')

        # 在随机位置显示绿色点
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        # 确保窗口已经渲染
        if width <= 1:
            self.root.update()
            width = self.canvas.winfo_width()
            height = self.canvas.winfo_height()

        # 留出边距，随机生成绿色点位置
        margin = 100
        x = int(random.uniform(margin, width - margin))
        y = int(random.uniform(margin, height - margin))

        self.start_point = (x, y)

        # 绘制绿色圆点
        self.canvas.create_oval(
            x - self.point_radius, y - self.point_radius,
            x + self.point_radius, y + self.point_radius,
            fill='green', outline='white', width=2, tags='start_point'
        )

        # 添加提示文字
        self.canvas.create_text(
            x, y + self.point_radius + 20,
            text='点击绿色点开始',
            fill='white', font=('Arial', 14), tags='hint'
        )

        # 显示当前组ID
        self.canvas.create_text(
            20, 20, text=f'当前组ID: {self.group_id}',
            fill='white', font=('Arial', 12), anchor='nw', tags='info'
        )

        self.canvas.create_text(
            20, 45, text='按ESC退出',
            fill='gray', font=('Arial', 10), anchor='nw', tags='info'
        )

        # 绑定点击事件
        self.canvas.tag_bind('start_point', '<Button-1>', self.on_start_point_click)
        self.state = 'waiting_start'

    def on_start_point_click(self, event):
        """点击绿色起点"""
        if self.state != 'waiting_start':
            return

        # 检查是否真的点击在绿色点上
        x, y = self.start_point
        distance = math.sqrt((event.x - x)**2 + (event.y - y)**2)
        if distance > self.point_radius:
            return

        # 显示红色终点并开始记录
        self.show_end_point()

    def show_end_point(self):
        """显示红色终点并开始记录"""
        # 清除提示
        self.canvas.delete('hint')

        # 在随机位置显示红色点（确保距离起点足够远）
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        margin = 100
        min_distance = 300  # 最小距离，确保有足够的轨迹数据

        # 生成随机终点，确保与起点距离足够远
        attempts = 0
        max_attempts = 100
        while attempts < max_attempts:
            end_x = random.uniform(margin, width - margin)
            end_y = random.uniform(margin, height - margin)

            # 计算与起点的距离
            distance = math.sqrt((end_x - self.start_point[0])**2 + (end_y - self.start_point[1])**2)

            if distance >= min_distance:
                break
            attempts += 1

        # 如果尝试多次仍未找到合适位置，使用固定策略
        if attempts >= max_attempts:
            start_x, start_y = self.start_point
            end_x = width - start_x if start_x < width / 2 else margin
            end_y = height - start_y if start_y < height / 2 else margin

        # 确保终点坐标为整数
        self.end_point = (int(end_x), int(end_y))

        # 绘制红色圆点
        self.canvas.create_oval(
            end_x - self.point_radius, end_y - self.point_radius,
            end_x + self.point_radius, end_y + self.point_radius,
            fill='red', outline='white', width=2, tags='end_point'
        )

        # 添加提示文字
        self.canvas.create_text(
            end_x, end_y + self.point_radius + 20,
            text='移动到红色点并点击',
            fill='white', font=('Arial', 14), tags='hint'
        )

        # 绑定点击事件
        self.canvas.tag_bind('end_point', '<Button-1>', self.on_end_point_click)

        # 开始记录
        self.state = 'recording'
        self.trajectory_data = []
        self.sequence_id = 0
        self.recording_start_time = time.time()
        self.last_position = self.start_point
        self.last_velocity = 0
        self.last_time = self.recording_start_time

        # 绑定鼠标移动事件
        self.canvas.bind('<Motion>', self.on_mouse_move)

    def on_mouse_move(self, event):
        """记录鼠标移动"""
        if self.state != 'recording':
            return

        current_time = time.time()
        timestamp = current_time - self.recording_start_time
        current_pos = (event.x, event.y)

        # 计算距离
        dx = current_pos[0] - self.last_position[0]
        dy = current_pos[1] - self.last_position[1]
        distance = math.sqrt(dx**2 + dy**2)

        # 计算时间差
        dt = current_time - self.last_time

        # 避免除以零
        if dt < 0.001:
            return

        # 计算速度 (像素/秒)
        velocity = distance / dt if dt > 0 else 0

        # 计算加速度 (像素/秒²)
        acceleration = (velocity - self.last_velocity) / dt if dt > 0 else 0

        # 计算运动方向（角度，0-360度，0度为正右方向）
        direction = math.degrees(math.atan2(dy, dx))
        if direction < 0:
            direction += 360

        # 记录数据（起点是上一个鼠标位置，当前是现在的位置）
        self.sequence_id += 1
        data_point = {
            'group_id': self.group_id,
            'sequence_id': self.sequence_id,
            'start_x': self.last_position[0],  # 从上一个位置
            'start_y': self.last_position[1],
            'end_x': self.end_point[0],        # 到目标终点
            'end_y': self.end_point[1],
            'current_x': current_pos[0],       # 当前位置
            'current_y': current_pos[1],
            'velocity': velocity,
            'acceleration': acceleration,
            'direction': direction,
            'timestamp': timestamp
        }

        self.trajectory_data.append(data_point)

        # 更新状态
        self.last_position = current_pos
        self.last_velocity = velocity
        self.last_time = current_time

    def on_end_point_click(self, event):
        """点击红色终点"""
        if self.state != 'recording':
            return

        # 检查是否真的点击在红色点上
        x, y = self.end_point
        distance = math.sqrt((event.x - x)**2 + (event.y - y)**2)
        if distance > self.point_radius:
            return

        # 停止记录
        self.state = 'waiting_end'
        self.canvas.unbind('<Motion>')

        # 保存数据
        self.save_trajectory_data()

        # 准备下一组
        self.group_id += 1
        self.show_start_point()

    def save_trajectory_data(self):
        """保存轨迹数据到CSV"""
        if not self.trajectory_data:
            print("没有数据可保存")
            return

        try:
            with open(self.data_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'group_id', 'sequence_id',
                    'start_x', 'start_y',
                    'end_x', 'end_y',
                    'current_x', 'current_y',
                    'velocity', 'acceleration', 'direction',
                    'timestamp'
                ])

                for data_point in self.trajectory_data:
                    # 确保坐标以整数形式保存
                    data_point['start_x'] = int(data_point['start_x'])
                    data_point['start_y'] = int(data_point['start_y'])
                    data_point['end_x'] = int(data_point['end_x'])
                    data_point['end_y'] = int(data_point['end_y'])
                    data_point['current_x'] = int(data_point['current_x'])
                    data_point['current_y'] = int(data_point['current_y'])
                    writer.writerow(data_point)

            print(f"已保存组 {self.group_id - 1} 的数据，共 {len(self.trajectory_data)} 条记录")

        except Exception as e:
            print(f"保存数据错误: {e}")


def main():
    root = tk.Tk()
    app = MouseTrajectoryCollector(root)
    root.mainloop()


if __name__ == '__main__':
    main()
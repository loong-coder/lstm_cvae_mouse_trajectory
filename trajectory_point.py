"""
轨迹点数据结构
包含坐标、时间、速度等信息，用于模拟真实的人类鼠标移动
"""
from dataclasses import dataclass
from typing import List, Tuple
import math


@dataclass
class TrajectoryPoint:
    """
    轨迹点数据结构

    Attributes:
        x: X坐标（像素）
        y: Y坐标（像素）
        timestamp: 相对时间戳（秒）
        speed: 移动速度（像素/秒）
        direction: 移动方向（弧度，相对于水平向右为0）
        duration: 到达此点需要的时间增量（秒）
    """
    x: float
    y: float
    timestamp: float
    speed: float
    direction: float
    duration: float = 0.0

    def distance_to(self, other: 'TrajectoryPoint') -> float:
        """计算到另一个点的欧氏距离"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def __repr__(self):
        return (f"TrajectoryPoint(x={self.x:.1f}, y={self.y:.1f}, "
                f"t={self.timestamp:.3f}s, speed={self.speed:.1f}px/s, "
                f"dir={math.degrees(self.direction):.1f}°)")


class EnhancedTrajectory:
    """
    增强的轨迹类，包含完整的时间和速度信息
    """

    def __init__(self, points: List[TrajectoryPoint] = None):
        """
        初始化轨迹

        Args:
            points: 轨迹点列表
        """
        self.points = points if points is not None else []

    def add_point(self, point: TrajectoryPoint):
        """添加轨迹点"""
        self.points.append(point)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx]

    def __iter__(self):
        return iter(self.points)

    @property
    def start_point(self) -> TrajectoryPoint:
        """起点"""
        return self.points[0] if self.points else None

    @property
    def end_point(self) -> TrajectoryPoint:
        """终点"""
        return self.points[-1] if self.points else None

    @property
    def total_duration(self) -> float:
        """总时长（秒）"""
        if not self.points:
            return 0.0
        return self.points[-1].timestamp

    @property
    def total_distance(self) -> float:
        """总距离（像素）"""
        if len(self.points) < 2:
            return 0.0
        distance = 0.0
        for i in range(1, len(self.points)):
            distance += self.points[i-1].distance_to(self.points[i])
        return distance

    @property
    def average_speed(self) -> float:
        """平均速度（像素/秒）"""
        if self.total_duration <= 0:
            return 0.0
        return self.total_distance / self.total_duration

    def get_coordinates(self) -> List[Tuple[float, float]]:
        """获取坐标列表（兼容旧接口）"""
        return [(p.x, p.y) for p in self.points]

    def get_speeds(self) -> List[float]:
        """获取速度列表"""
        return [p.speed for p in self.points]

    def get_timestamps(self) -> List[float]:
        """获取时间戳列表"""
        return [p.timestamp for p in self.points]

    def get_durations(self) -> List[float]:
        """获取时间增量列表"""
        return [p.duration for p in self.points]

    def summary(self) -> str:
        """获取轨迹摘要信息"""
        if not self.points:
            return "Empty trajectory"

        return (
            f"Trajectory Summary:\n"
            f"  Points: {len(self.points)}\n"
            f"  Start: ({self.start_point.x:.1f}, {self.start_point.y:.1f})\n"
            f"  End: ({self.end_point.x:.1f}, {self.end_point.y:.1f})\n"
            f"  Duration: {self.total_duration:.3f}s\n"
            f"  Distance: {self.total_distance:.1f}px\n"
            f"  Avg Speed: {self.average_speed:.1f}px/s\n"
            f"  Speed Range: [{min(self.get_speeds()):.1f}, {max(self.get_speeds()):.1f}]px/s"
        )

    def interpolate_missing_durations(self):
        """
        插值缺失的时间增量信息
        如果某些点的duration为0，根据速度和距离计算
        """
        if len(self.points) < 2:
            return

        for i in range(1, len(self.points)):
            current = self.points[i]
            previous = self.points[i-1]

            # 如果duration未设置或为0，根据距离和速度计算
            if current.duration <= 0:
                distance = previous.distance_to(current)
                if current.speed > 0:
                    current.duration = distance / current.speed
                else:
                    # 如果速度也是0，使用前一个点的时间间隔
                    if i > 1:
                        current.duration = self.points[i-1].duration
                    else:
                        current.duration = 0.01  # 默认10ms

    @classmethod
    def from_coordinates(cls, coordinates: List[Tuple[float, float]],
                        default_speed: float = 500.0) -> 'EnhancedTrajectory':
        """
        从坐标列表创建轨迹（为了兼容旧接口）
        使用默认速度和方向

        Args:
            coordinates: 坐标列表 [(x1, y1), (x2, y2), ...]
            default_speed: 默认速度（像素/秒）

        Returns:
            EnhancedTrajectory对象
        """
        points = []
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
                # 计算距离和方向
                prev_x, prev_y = coordinates[i-1]
                dx = x - prev_x
                dy = y - prev_y
                distance = math.sqrt(dx**2 + dy**2)
                direction = math.atan2(dy, dx)

                # 计算时间增量
                duration = distance / default_speed if default_speed > 0 else 0.01
                cumulative_time += duration

                point = TrajectoryPoint(
                    x=x, y=y,
                    timestamp=cumulative_time,
                    speed=default_speed,
                    direction=direction,
                    duration=duration
                )

            points.append(point)

        return cls(points)
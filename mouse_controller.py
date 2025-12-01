"""
PyAutoGUI 鼠标控制器
使用增强轨迹（包含速度和时间信息）来模拟真实的人类鼠标移动
"""
import time
import pyautogui
from typing import List, Tuple, Optional
from trajectory_point import EnhancedTrajectory, TrajectoryPoint
import math


class HumanMouseController:
    """
    人类化的鼠标控制器
    根据轨迹的速度和时间信息来模拟真实的人类鼠标移动
    """

    def __init__(self, failsafe=True, pause_between_moves=0.0):
        """
        初始化鼠标控制器

        Args:
            failsafe: 是否启用pyautogui的故障保护功能（移动到屏幕角落终止）
            pause_between_moves: 移动之间的额外暂停时间（秒）
        """
        pyautogui.FAILSAFE = failsafe
        pyautogui.PAUSE = pause_between_moves

        # 获取屏幕尺寸
        self.screen_width, self.screen_height = pyautogui.size()

    def move_along_trajectory(self, trajectory: EnhancedTrajectory,
                             start_immediately=False,
                             speed_multiplier=1.0,
                             min_duration=0.001,
                             max_duration=0.5,
                             use_tweening=False) -> float:
        """
        沿着增强轨迹移动鼠标

        Args:
            trajectory: EnhancedTrajectory对象，包含完整的速度和时间信息
            start_immediately: 是否立即开始（否则先移动到起点）
            speed_multiplier: 速度倍数（<1.0减速，>1.0加速）
            min_duration: 最小移动时间（秒），防止移动过快
            max_duration: 最大移动时间（秒），防止移动过慢
            use_tweening: 是否使用pyautogui的缓动函数（实验性）

        Returns:
            total_time: 实际执行的总时间（秒）
        """
        if len(trajectory) == 0:
            return 0.0

        # 如果不立即开始，先移动到起点
        if not start_immediately and len(trajectory) > 0:
            start_point = trajectory[0]
            pyautogui.moveTo(int(start_point.x), int(start_point.y))

        # 记录开始时间
        start_time = time.time()

        # 遍历轨迹点（跳过起点，因为已经在起点了）
        for i in range(1, len(trajectory)):
            point = trajectory[i]

            # 计算移动持续时间（考虑速度倍数）
            duration = point.duration / speed_multiplier

            # 限制在合理范围内
            duration = max(min_duration, min(max_duration, duration))

            # 移动到目标点
            if use_tweening:
                # 使用pyautogui的缓动函数（更平滑但可能不太真实）
                pyautogui.moveTo(
                    int(point.x),
                    int(point.y),
                    duration=duration,
                    tween=pyautogui.easeInOutQuad
                )
            else:
                # 直接移动（更真实）
                pyautogui.moveTo(
                    int(point.x),
                    int(point.y),
                    duration=duration
                )

        # 计算实际总时间
        total_time = time.time() - start_time

        return total_time

    def move_to_target(self, target_x: int, target_y: int,
                      trajectory: EnhancedTrajectory,
                      speed_multiplier=1.0) -> float:
        """
        使用轨迹移动到目标位置（自动从当前位置开始）

        Args:
            target_x: 目标X坐标
            target_y: 目标Y坐标
            trajectory: 预测的轨迹
            speed_multiplier: 速度倍数

        Returns:
            total_time: 实际执行时间（秒）
        """
        # 获取当前鼠标位置
        current_x, current_y = pyautogui.position()

        # 如果轨迹的起点不是当前位置，需要调整轨迹
        if len(trajectory) > 0:
            traj_start = trajectory[0]

            # 计算偏移量
            offset_x = current_x - traj_start.x
            offset_y = current_y - traj_start.y

            # 平移轨迹
            adjusted_points = []
            for point in trajectory:
                adjusted_point = TrajectoryPoint(
                    x=point.x + offset_x,
                    y=point.y + offset_y,
                    timestamp=point.timestamp,
                    speed=point.speed,
                    direction=point.direction,
                    duration=point.duration
                )
                adjusted_points.append(adjusted_point)

            adjusted_trajectory = EnhancedTrajectory(adjusted_points)

            # 沿着调整后的轨迹移动
            return self.move_along_trajectory(
                adjusted_trajectory,
                start_immediately=True,
                speed_multiplier=speed_multiplier
            )
        else:
            # 如果轨迹为空，直接移动到目标
            pyautogui.moveTo(target_x, target_y)
            return 0.0

    def click_at(self, x: int, y: int, button='left', clicks=1, interval=0.1):
        """
        在指定位置点击鼠标

        Args:
            x: X坐标
            y: Y坐标
            button: 鼠标按钮 ('left', 'right', 'middle')
            clicks: 点击次数
            interval: 多次点击之间的间隔（秒）
        """
        pyautogui.click(x, y, clicks=clicks, interval=interval, button=button)

    def move_and_click(self, target_x: int, target_y: int,
                      trajectory: EnhancedTrajectory,
                      button='left', clicks=1,
                      speed_multiplier=1.0,
                      click_delay=0.1) -> float:
        """
        移动到目标位置并点击

        Args:
            target_x: 目标X坐标
            target_y: 目标Y坐标
            trajectory: 移动轨迹
            button: 鼠标按钮
            clicks: 点击次数
            speed_multiplier: 速度倍数
            click_delay: 到达后点击前的延迟（秒）

        Returns:
            total_time: 移动+点击的总时间（秒）
        """
        # 移动到目标
        move_time = self.move_to_target(target_x, target_y, trajectory, speed_multiplier)

        # 短暂延迟（模拟人类反应时间）
        time.sleep(click_delay)

        # 点击
        self.click_at(target_x, target_y, button=button, clicks=clicks)

        return move_time + click_delay

    def get_current_position(self) -> Tuple[int, int]:
        """获取当前鼠标位置"""
        return pyautogui.position()

    def drag_along_trajectory(self, trajectory: EnhancedTrajectory,
                             button='left',
                             speed_multiplier=1.0) -> float:
        """
        按住鼠标按钮并沿轨迹拖动

        Args:
            trajectory: 拖动轨迹
            button: 鼠标按钮
            speed_multiplier: 速度倍数

        Returns:
            total_time: 拖动总时间（秒）
        """
        if len(trajectory) == 0:
            return 0.0

        # 移动到起点
        start_point = trajectory[0]
        pyautogui.moveTo(int(start_point.x), int(start_point.y))

        # 按下鼠标
        pyautogui.mouseDown(button=button)

        start_time = time.time()

        try:
            # 拖动到每个点
            for i in range(1, len(trajectory)):
                point = trajectory[i]
                duration = max(0.001, point.duration / speed_multiplier)

                pyautogui.moveTo(
                    int(point.x),
                    int(point.y),
                    duration=duration
                )
        finally:
            # 确保释放鼠标按钮
            pyautogui.mouseUp(button=button)

        total_time = time.time() - start_time
        return total_time


class SimpleMouseController:
    """
    简化的鼠标控制器
    直接使用坐标列表（向后兼容旧接口）
    """

    def __init__(self):
        """初始化简化鼠标控制器"""
        pyautogui.FAILSAFE = True

    def move_along_coordinates(self, coordinates: List[Tuple[float, float]],
                              total_duration: float = 1.0,
                              pause_at_end: float = 0.0):
        """
        沿坐标列表移动鼠标

        Args:
            coordinates: 坐标列表 [(x1, y1), (x2, y2), ...]
            total_duration: 总移动时间（秒）
            pause_at_end: 到达终点后的暂停时间（秒）
        """
        if len(coordinates) == 0:
            return

        # 计算每步的持续时间
        steps = len(coordinates) - 1
        if steps <= 0:
            steps = 1

        duration_per_step = total_duration / steps

        # 移动到每个点
        for x, y in coordinates:
            pyautogui.moveTo(int(x), int(y), duration=duration_per_step)

        # 暂停
        if pause_at_end > 0:
            time.sleep(pause_at_end)

    def move_to(self, x: int, y: int, duration: float = 0.5):
        """
        移动到指定位置

        Args:
            x: 目标X坐标
            y: 目标Y坐标
            duration: 移动持续时间（秒）
        """
        pyautogui.moveTo(x, y, duration=duration)

    def click_at(self, x: int, y: int, button='left'):
        """在指定位置点击"""
        pyautogui.click(x, y, button=button)


def move_mouse_humanlike(trajectory: EnhancedTrajectory,
                         speed_multiplier: float = 1.0,
                         click_at_end: bool = False) -> float:
    """
    便捷函数：使用增强轨迹进行人类化的鼠标移动

    Args:
        trajectory: 包含完整速度和时间信息的轨迹
        speed_multiplier: 速度倍数
        click_at_end: 是否在终点点击

    Returns:
        total_time: 移动总时间（秒）
    """
    controller = HumanMouseController()

    if click_at_end and len(trajectory) > 0:
        end_point = trajectory[-1]
        return controller.move_and_click(
            int(end_point.x),
            int(end_point.y),
            trajectory,
            speed_multiplier=speed_multiplier
        )
    else:
        return controller.move_along_trajectory(
            trajectory,
            speed_multiplier=speed_multiplier
        )


# ========== 使用示例 ==========

if __name__ == '__main__':
    print("=" * 80)
    print("PyAutoGUI 鼠标控制器示例")
    print("=" * 80)
    print("\n警告：此脚本将控制你的鼠标！")
    print("将鼠标移动到屏幕角落可以中断程序（Failsafe功能）")
    print("\n按回车键继续，或按 Ctrl+C 取消...")
    input()

    from predict_trajectory import TrajectoryPredictor

    # 加载预测器
    predictor = TrajectoryPredictor(model_path='cvae_trajectory_predictor.pth')
    controller = HumanMouseController()

    print("\n示例 1: 简单移动")
    print("-" * 80)

    # 获取当前位置
    current_x, current_y = controller.get_current_position()
    print(f"当前鼠标位置: ({current_x}, {current_y})")

    # 预测轨迹
    target_x, target_y = 500, 400
    print(f"目标位置: ({target_x}, {target_y})")

    print("正在预测轨迹...")
    trajectory = predictor.predict_enhanced(current_x, current_y, target_x, target_y)

    print(f"\n{trajectory.summary()}")

    print("\n3秒后开始移动...")
    time.sleep(3)

    # 执行移动
    print("正在移动鼠标...")
    elapsed_time = controller.move_to_target(target_x, target_y, trajectory)

    print(f"移动完成！实际耗时: {elapsed_time:.3f}秒")
    print(f"预计耗时: {trajectory.total_duration:.3f}秒")

    print("\n示例 2: 移动并点击")
    print("-" * 80)

    time.sleep(2)

    # 新目标
    target_x, target_y = 700, 300
    print(f"目标位置: ({target_x}, {target_y})")

    current_x, current_y = controller.get_current_position()
    trajectory = predictor.predict_enhanced(current_x, current_y, target_x, target_y)

    print(f"\n{trajectory.summary()}")
    print("\n2秒后开始移动并点击...")
    time.sleep(2)

    elapsed_time = controller.move_and_click(target_x, target_y, trajectory)
    print(f"移动并点击完成！耗时: {elapsed_time:.3f}秒")

    print("\n示例 3: 不同速度")
    print("-" * 80)

    time.sleep(2)

    target_x, target_y = 300, 500
    current_x, current_y = controller.get_current_position()
    trajectory = predictor.predict_enhanced(current_x, current_y, target_x, target_y)

    print(f"原始速度预计耗时: {trajectory.total_duration:.3f}秒")

    print("\n2秒后以2倍速移动...")
    time.sleep(2)

    elapsed_time = controller.move_to_target(
        target_x, target_y,
        trajectory,
        speed_multiplier=2.0  # 2倍速
    )

    print(f"2倍速移动完成！实际耗时: {elapsed_time:.3f}秒")

    print("\n示例 4: 慢速移动")
    print("-" * 80)

    time.sleep(2)

    target_x, target_y = 600, 200
    current_x, current_y = controller.get_current_position()
    trajectory = predictor.predict_enhanced(current_x, current_y, target_x, target_y)

    print(f"原始速度预计耗时: {trajectory.total_duration:.3f}秒")

    print("\n2秒后以0.5倍速移动...")
    time.sleep(2)

    elapsed_time = controller.move_to_target(
        target_x, target_y,
        trajectory,
        speed_multiplier=0.5  # 半速
    )

    print(f"0.5倍速移动完成！实际耗时: {elapsed_time:.3f}秒")

    print("\n" + "=" * 80)
    print("示例运行完成！")
    print("=" * 80)
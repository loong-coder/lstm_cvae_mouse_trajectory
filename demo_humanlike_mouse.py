"""
人类化鼠标移动完整演示
展示如何使用CVAE模型预测轨迹，并通过pyautogui控制鼠标进行真实的人类化移动
"""
import time
from predict_trajectory import TrajectoryPredictor
from mouse_controller import HumanMouseController
from trajectory_point import EnhancedTrajectory


def demo_basic_movement():
    """演示基本的人类化鼠标移动"""
    print("\n" + "=" * 80)
    print("演示 1: 基本的人类化鼠标移动")
    print("=" * 80)

    # 初始化预测器和控制器
    predictor = TrajectoryPredictor(model_path='cvae_trajectory_predictor.pth')
    controller = HumanMouseController()

    # 获取当前位置
    current_x, current_y = controller.get_current_position()
    print(f"\n当前鼠标位置: ({current_x}, {current_y})")

    # 定义目标位置
    target_x, target_y = 700, 400
    print(f"目标位置: ({target_x}, {target_y})")

    # 预测轨迹（包含完整的速度和时间信息）
    print("\n正在使用CVAE模型预测轨迹...")
    trajectory = predictor.predict_enhanced(
        current_x, current_y,
        target_x, target_y,
        smooth=True,
        downsample=True,
        align_endpoints=True
    )

    # 显示轨迹信息
    print(f"\n{trajectory.summary()}")

    # 显示前5个点的详细信息
    print("\n前5个轨迹点详情:")
    for i, point in enumerate(trajectory.points[:5]):
        print(f"  点 {i}: {point}")

    # 执行移动
    print("\n3秒后开始移动...")
    time.sleep(3)

    print("正在移动鼠标（使用预测的速度和时间）...")
    elapsed_time = controller.move_to_target(
        target_x, target_y,
        trajectory,
        speed_multiplier=1.0
    )

    print(f"\n移动完成！")
    print(f"  预计耗时: {trajectory.total_duration:.3f}秒")
    print(f"  实际耗时: {elapsed_time:.3f}秒")
    print(f"  时间差异: {abs(elapsed_time - trajectory.total_duration):.3f}秒")


def demo_speed_variations():
    """演示不同速度的移动"""
    print("\n" + "=" * 80)
    print("演示 2: 不同速度的鼠标移动")
    print("=" * 80)

    predictor = TrajectoryPredictor(model_path='cvae_trajectory_predictor.pth')
    controller = HumanMouseController()

    # 定义一系列目标和速度
    targets_and_speeds = [
        ((300, 300), 1.0, "正常速度"),
        ((700, 500), 1.5, "1.5倍速"),
        ((500, 200), 0.7, "0.7倍速"),
        ((200, 500), 2.0, "2倍速")
    ]

    for target, speed_mult, description in targets_and_speeds:
        current_x, current_y = controller.get_current_position()
        target_x, target_y = target

        print(f"\n从 ({current_x}, {current_y}) 移动到 ({target_x}, {target_y})")
        print(f"速度: {description} (速度倍数: {speed_mult})")

        # 预测轨迹
        trajectory = predictor.predict_enhanced(
            current_x, current_y,
            target_x, target_y
        )

        print(f"预计耗时: {trajectory.total_duration:.3f}秒")
        print(f"调整后预计: {trajectory.total_duration / speed_mult:.3f}秒")

        print("2秒后开始移动...")
        time.sleep(2)

        # 执行移动
        elapsed_time = controller.move_to_target(
            target_x, target_y,
            trajectory,
            speed_multiplier=speed_mult
        )

        print(f"实际耗时: {elapsed_time:.3f}秒")


def demo_move_and_click():
    """演示移动并点击"""
    print("\n" + "=" * 80)
    print("演示 3: 移动并点击")
    print("=" * 80)

    predictor = TrajectoryPredictor(model_path='cvae_trajectory_predictor.pth')
    controller = HumanMouseController()

    # 定义多个点击目标
    click_targets = [
        (400, 300, "左键单击", 'left', 1),
        (600, 400, "左键双击", 'left', 2),
        (350, 500, "右键单击", 'right', 1)
    ]

    for target_x, target_y, description, button, clicks in click_targets:
        current_x, current_y = controller.get_current_position()

        print(f"\n从 ({current_x}, {current_y}) 移动到 ({target_x}, {target_y})")
        print(f"操作: {description}")

        # 预测轨迹
        trajectory = predictor.predict_enhanced(
            current_x, current_y,
            target_x, target_y
        )

        print(f"预计耗时: {trajectory.total_duration:.3f}秒")

        print("2秒后开始移动并点击...")
        time.sleep(2)

        # 移动并点击
        elapsed_time = controller.move_and_click(
            target_x, target_y,
            trajectory,
            button=button,
            clicks=clicks,
            click_delay=0.1  # 到达后等待0.1秒再点击（模拟人类反应时间）
        )

        print(f"实际耗时: {elapsed_time:.3f}秒")


def demo_trajectory_analysis():
    """演示轨迹分析和可视化"""
    print("\n" + "=" * 80)
    print("演示 4: 轨迹详细分析")
    print("=" * 80)

    predictor = TrajectoryPredictor(model_path='cvae_trajectory_predictor.pth')

    # 生成轨迹但不移动鼠标
    start_x, start_y = 100, 100
    end_x, end_y = 700, 500

    print(f"\n分析从 ({start_x}, {start_y}) 到 ({end_x}, {end_y}) 的轨迹")

    trajectory = predictor.predict_enhanced(
        start_x, start_y,
        end_x, end_y
    )

    print(f"\n{trajectory.summary()}")

    # 详细分析
    print("\n速度分布:")
    speeds = trajectory.get_speeds()
    print(f"  最小速度: {min(speeds):.1f} px/s")
    print(f"  最大速度: {max(speeds):.1f} px/s")
    print(f"  平均速度: {trajectory.average_speed:.1f} px/s")

    print("\n时间分布:")
    durations = trajectory.get_durations()
    print(f"  最小时间间隔: {min(d for d in durations if d > 0):.4f}秒")
    print(f"  最大时间间隔: {max(durations):.4f}秒")
    print(f"  平均时间间隔: {sum(durations)/len(durations):.4f}秒")

    print("\n轨迹点详细信息（前10个点）:")
    for i, point in enumerate(trajectory.points[:10]):
        print(f"  {i}: x={point.x:.1f}, y={point.y:.1f}, "
              f"t={point.timestamp:.3f}s, speed={point.speed:.1f}px/s, "
              f"Δt={point.duration:.4f}s")


def demo_multiple_trajectories():
    """演示生成多条轨迹（展示CVAE的多样性）"""
    print("\n" + "=" * 80)
    print("演示 5: 多样化轨迹生成（不移动鼠标）")
    print("=" * 80)

    predictor = TrajectoryPredictor(model_path='cvae_trajectory_predictor.pth')

    start_x, start_y = 200, 200
    end_x, end_y = 600, 400

    print(f"\n从 ({start_x}, {start_y}) 到 ({end_x}, {end_y})")
    print("生成3条不同的轨迹（展示CVAE的多样性）...")

    # 生成多条轨迹
    for i in range(3):
        trajectory = predictor.predict_enhanced(
            start_x, start_y,
            end_x, end_y
        )

        print(f"\n轨迹 {i+1}:")
        print(f"  点数: {len(trajectory)}")
        print(f"  总距离: {trajectory.total_distance:.1f}px")
        print(f"  总时长: {trajectory.total_duration:.3f}s")
        print(f"  平均速度: {trajectory.average_speed:.1f}px/s")


def main():
    """主函数"""
    print("=" * 80)
    print("人类化鼠标移动完整演示")
    print("=" * 80)
    print("\n这个演示将展示如何使用CVAE模型预测的轨迹来控制鼠标")
    print("轨迹包含完整的速度和时间信息，可以精确模拟人类的移动模式")
    print("\n警告：部分演示会实际控制你的鼠标！")
    print("你可以随时将鼠标移动到屏幕角落来中断程序（Failsafe功能）")

    while True:
        print("\n" + "=" * 80)
        print("请选择演示:")
        print("  1 - 基本的人类化鼠标移动（会控制鼠标）")
        print("  2 - 不同速度的移动（会控制鼠标）")
        print("  3 - 移动并点击（会控制鼠标和点击）")
        print("  4 - 轨迹详细分析（不会控制鼠标）")
        print("  5 - 多样化轨迹生成（不会控制鼠标）")
        print("  0 - 退出")
        print("=" * 80)

        choice = input("\n请输入选项 (0-5): ").strip()

        if choice == '0':
            print("\n再见！")
            break
        elif choice == '1':
            print("\n警告：3秒后将开始控制鼠标！")
            if input("按回车继续，或输入 'n' 取消: ").lower() != 'n':
                demo_basic_movement()
        elif choice == '2':
            print("\n警告：2秒后将开始控制鼠标！")
            if input("按回车继续，或输入 'n' 取消: ").lower() != 'n':
                demo_speed_variations()
        elif choice == '3':
            print("\n警告：2秒后将开始控制鼠标并点击！")
            if input("按回车继续，或输入 'n' 取消: ").lower() != 'n':
                demo_move_and_click()
        elif choice == '4':
            demo_trajectory_analysis()
        elif choice == '5':
            demo_multiple_trajectories()
        else:
            print("\n无效的选项，请重新选择")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n\n发生错误: {e}")
        import traceback
        traceback.print_exc()
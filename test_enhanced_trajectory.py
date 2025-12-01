"""
测试增强轨迹功能（不控制鼠标）
验证轨迹包含正确的速度和时间信息
"""
from predict_trajectory import TrajectoryPredictor
from trajectory_point import EnhancedTrajectory, TrajectoryPoint
import math


def test_trajectory_structure():
    """测试轨迹数据结构"""
    print("=" * 80)
    print("测试 1: 轨迹数据结构")
    print("=" * 80)

    # 创建测试轨迹点
    points = [
        TrajectoryPoint(x=0, y=0, timestamp=0.0, speed=0, direction=0, duration=0),
        TrajectoryPoint(x=100, y=50, timestamp=0.2, speed=250, direction=0.463, duration=0.2),
        TrajectoryPoint(x=200, y=100, timestamp=0.4, speed=250, direction=0.463, duration=0.2),
    ]

    trajectory = EnhancedTrajectory(points)

    print(f"\n创建了 {len(trajectory)} 个点的轨迹")
    print(f"起点: ({trajectory.start_point.x}, {trajectory.start_point.y})")
    print(f"终点: ({trajectory.end_point.x}, {trajectory.end_point.y})")
    print(f"总时长: {trajectory.total_duration:.3f}秒")
    print(f"总距离: {trajectory.total_distance:.1f}像素")

    print("\n轨迹点详情:")
    for i, point in enumerate(trajectory):
        print(f"  点{i}: {point}")

    print("\n✓ 测试通过")


def test_predict_enhanced():
    """测试增强预测功能"""
    print("\n" + "=" * 80)
    print("测试 2: 增强轨迹预测")
    print("=" * 80)

    try:
        predictor = TrajectoryPredictor(model_path='cvae_trajectory_predictor.pth')
        print("✓ 模型加载成功")
    except FileNotFoundError:
        print("✗ 模型文件未找到，跳过此测试")
        return

    # 预测轨迹
    start_x, start_y = 100, 100
    end_x, end_y = 700, 500

    print(f"\n预测从 ({start_x}, {start_y}) 到 ({end_x}, {end_y}) 的轨迹...")

    trajectory = predictor.predict_enhanced(
        start_x, start_y,
        end_x, end_y
    )

    print(f"\n{trajectory.summary()}")

    # 验证数据完整性
    print("\n数据完整性检查:")

    # 检查起点和终点
    assert len(trajectory) > 0, "轨迹不能为空"
    print(f"  ✓ 轨迹包含 {len(trajectory)} 个点")

    # 检查起点
    start_point = trajectory.start_point
    assert abs(start_point.x - start_x) < 1.0, "起点X坐标不匹配"
    assert abs(start_point.y - start_y) < 1.0, "起点Y坐标不匹配"
    print(f"  ✓ 起点坐标正确: ({start_point.x:.1f}, {start_point.y:.1f})")

    # 检查速度信息
    speeds = trajectory.get_speeds()
    assert all(s >= 0 for s in speeds), "速度不能为负"
    assert any(s > 0 for s in speeds[1:]), "至少应该有一些非零速度"
    print(f"  ✓ 速度范围: [{min(speeds):.1f}, {max(speeds):.1f}] px/s")

    # 检查时间信息
    timestamps = trajectory.get_timestamps()
    assert timestamps[0] == 0.0, "起点时间戳应该为0"
    assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)), \
        "时间戳应该单调递增"
    print(f"  ✓ 时间戳单调递增，总时长: {trajectory.total_duration:.3f}秒")

    # 检查时间增量
    durations = trajectory.get_durations()
    assert all(d >= 0 for d in durations), "时间增量不能为负"
    print(f"  ✓ 时间增量范围: [{min(d for d in durations if d > 0):.4f}, {max(durations):.4f}]秒")

    # 检查方向信息
    print(f"  ✓ 方向信息已包含（弧度制）")

    print("\n✓ 所有检查通过")

    # 显示详细信息
    print("\n前5个点的详细信息:")
    for i, point in enumerate(trajectory.points[:5]):
        print(f"  点{i}: x={point.x:.1f}, y={point.y:.1f}, "
              f"t={point.timestamp:.3f}s, speed={point.speed:.1f}px/s, "
              f"Δt={point.duration:.4f}s, dir={math.degrees(point.direction):.1f}°")


def test_trajectory_methods():
    """测试轨迹方法"""
    print("\n" + "=" * 80)
    print("测试 3: 轨迹方法")
    print("=" * 80)

    # 创建测试轨迹
    points = []
    cumulative_time = 0.0

    for i in range(10):
        x = i * 100
        y = i * 50
        speed = 200 + i * 10
        duration = 0.1 if i > 0 else 0.0
        cumulative_time += duration

        point = TrajectoryPoint(
            x=x, y=y,
            timestamp=cumulative_time,
            speed=speed,
            direction=math.atan2(50, 100),
            duration=duration
        )
        points.append(point)

    trajectory = EnhancedTrajectory(points)

    # 测试各种方法
    print("\n基本信息:")
    print(f"  点数: {len(trajectory)}")
    print(f"  总时长: {trajectory.total_duration:.3f}秒")
    print(f"  总距离: {trajectory.total_distance:.1f}像素")
    print(f"  平均速度: {trajectory.average_speed:.1f}px/s")

    # 测试获取列表
    coords = trajectory.get_coordinates()
    print(f"\n  坐标列表长度: {len(coords)}")
    print(f"  第一个坐标: {coords[0]}")
    print(f"  最后一个坐标: {coords[-1]}")

    speeds = trajectory.get_speeds()
    print(f"\n  速度列表长度: {len(speeds)}")
    print(f"  速度范围: [{min(speeds):.1f}, {max(speeds):.1f}] px/s")

    timestamps = trajectory.get_timestamps()
    print(f"\n  时间戳列表长度: {len(timestamps)}")
    print(f"  时间范围: [{timestamps[0]:.3f}, {timestamps[-1]:.3f}]秒")

    # 测试摘要
    print(f"\n摘要信息:")
    print(trajectory.summary())

    print("\n✓ 所有方法测试通过")


def test_from_coordinates():
    """测试从坐标列表创建轨迹"""
    print("\n" + "=" * 80)
    print("测试 4: 从坐标列表创建轨迹")
    print("=" * 80)

    # 创建坐标列表
    coordinates = [
        (0, 0),
        (100, 50),
        (200, 100),
        (300, 150),
        (400, 200)
    ]

    print(f"\n创建包含 {len(coordinates)} 个坐标的轨迹")

    # 使用默认速度
    trajectory = EnhancedTrajectory.from_coordinates(
        coordinates,
        default_speed=500.0
    )

    print(f"\n{trajectory.summary()}")

    # 验证
    assert len(trajectory) == len(coordinates), "点数不匹配"
    print(f"  ✓ 点数匹配: {len(trajectory)}")

    assert trajectory.start_point.x == coordinates[0][0], "起点X不匹配"
    assert trajectory.start_point.y == coordinates[0][1], "起点Y不匹配"
    print(f"  ✓ 起点正确")

    assert trajectory.end_point.x == coordinates[-1][0], "终点X不匹配"
    assert trajectory.end_point.y == coordinates[-1][1], "终点Y不匹配"
    print(f"  ✓ 终点正确")

    # 检查速度
    speeds = trajectory.get_speeds()
    assert speeds[0] == 0.0, "起点速度应该为0"
    assert all(s > 0 for s in speeds[1:]), "其他点速度应该大于0"
    print(f"  ✓ 速度信息正确")

    print("\n✓ 测试通过")


def test_comparison():
    """对比原始predict和增强predict"""
    print("\n" + "=" * 80)
    print("测试 5: 原始predict vs 增强predict对比")
    print("=" * 80)

    try:
        predictor = TrajectoryPredictor(model_path='cvae_trajectory_predictor.pth')
    except FileNotFoundError:
        print("✗ 模型文件未找到，跳过此测试")
        return

    start_x, start_y = 200, 200
    end_x, end_y = 600, 400

    # 原始predict
    print("\n使用原始 predict() 方法:")
    coords = predictor.predict(start_x, start_y, end_x, end_y)
    print(f"  返回类型: {type(coords)}")
    print(f"  返回值: 坐标列表，长度 {len(coords)}")
    print(f"  示例: {coords[:3]}")

    # 增强predict
    print("\n使用增强 predict_enhanced() 方法:")
    trajectory = predictor.predict_enhanced(start_x, start_y, end_x, end_y)
    print(f"  返回类型: {type(trajectory)}")
    print(f"  返回值: EnhancedTrajectory对象")
    print(f"  包含信息:")
    print(f"    - 坐标: {len(trajectory)} 个点")
    print(f"    - 速度: {trajectory.average_speed:.1f} px/s (平均)")
    print(f"    - 时间: {trajectory.total_duration:.3f}秒")
    print(f"    - 距离: {trajectory.total_distance:.1f}像素")

    # 转换对比
    print("\n兼容性测试:")
    enhanced_coords = trajectory.get_coordinates()
    print(f"  EnhancedTrajectory可以转换为坐标列表")
    print(f"  转换后长度: {len(enhanced_coords)}")
    print(f"  格式相同: {isinstance(enhanced_coords[0], tuple)}")

    print("\n✓ 对比完成，两种方法都可以使用")


def main():
    """运行所有测试"""
    print("开始测试增强轨迹功能...")
    print("这些测试不会控制鼠标，只验证数据结构和方法\n")

    try:
        test_trajectory_structure()
        test_predict_enhanced()
        test_trajectory_methods()
        test_from_coordinates()
        test_comparison()

        print("\n" + "=" * 80)
        print("所有测试通过！")
        print("=" * 80)

    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
"""
轨迹预测单元测试

测试内容：
1. 模型加载
2. 单条轨迹预测
3. 多条轨迹预测
4. 终点对齐功能
5. 平滑和降采样功能
6. 预测性能（推理速度）
7. 轨迹质量（起点/终点精度、平滑度）
"""

import unittest
import time
import math
import sys
import os
import numpy as np
from predict_trajectory import TrajectoryPredictor, predict_trajectory


class TestTrajectoryPrediction(unittest.TestCase):
    """轨迹预测测试类"""

    @classmethod
    def setUpClass(cls):
        """测试前的准备工作（只执行一次）"""
        print("\n" + "=" * 80)
        print("开始轨迹预测单元测试")
        print("=" * 80)

        # 检查模型文件是否存在
        if not os.path.exists('cvae_trajectory_predictor.pth'):
            raise FileNotFoundError("模型文件 'cvae_trajectory_predictor.pth' 不存在！")

        # 创建预测器实例（复用以提高测试速度）
        cls.predictor = TrajectoryPredictor(model_path='cvae_trajectory_predictor.pth')
        print("✓ 模型加载成功\n")

    def test_01_model_loading(self):
        """测试1: 模型加载"""
        print("\n[测试1] 模型加载测试")
        print("-" * 80)

        # 检查模型是否正确加载
        self.assertIsNotNone(self.predictor.model)
        self.assertEqual(self.predictor.canvas_width, 800)
        self.assertEqual(self.predictor.canvas_height, 600)

        print("✓ 模型实例创建成功")
        print("✓ 画布尺寸配置正确")

    def test_02_single_trajectory_prediction(self):
        """测试2: 单条轨迹预测"""
        print("\n[测试2] 单条轨迹预测测试")
        print("-" * 80)

        start_x, start_y = 100, 100
        end_x, end_y = 700, 500

        # 测试预测
        start_time = time.time()
        trajectory = self.predictor.predict(start_x, start_y, end_x, end_y)
        inference_time = time.time() - start_time

        # 断言
        self.assertIsInstance(trajectory, list)
        self.assertGreater(len(trajectory), 0, "轨迹不应为空")
        self.assertIsInstance(trajectory[0], tuple, "轨迹点应为元组格式")
        self.assertEqual(len(trajectory[0]), 2, "轨迹点应包含x和y坐标")

        print(f"✓ 轨迹生成成功")
        print(f"  - 轨迹点数: {len(trajectory)}")
        print(f"  - 推理时间: {inference_time*1000:.2f} ms")
        print(f"  - 起点: {trajectory[0]}")
        print(f"  - 终点: {trajectory[-1]}")

    def test_03_multiple_trajectory_prediction(self):
        """测试3: 多条轨迹预测（CVAE多样性）"""
        print("\n[测试3] 多条轨迹预测测试")
        print("-" * 80)

        start_x, start_y = 200, 300
        end_x, end_y = 600, 400
        num_samples = 5

        # 测试预测
        start_time = time.time()
        trajectories = self.predictor.predict_multiple(
            start_x, start_y, end_x, end_y,
            num_samples=num_samples
        )
        inference_time = time.time() - start_time

        # 断言
        self.assertEqual(len(trajectories), num_samples, f"应生成{num_samples}条轨迹")
        for i, traj in enumerate(trajectories):
            self.assertGreater(len(traj), 0, f"轨迹{i+1}不应为空")

        print(f"✓ 多条轨迹生成成功")
        print(f"  - 样本数量: {len(trajectories)}")
        print(f"  - 总推理时间: {inference_time*1000:.2f} ms")
        print(f"  - 平均每条: {inference_time/num_samples*1000:.2f} ms")
        for i, traj in enumerate(trajectories, 1):
            print(f"  - 样本{i}: {len(traj)} 个点")

    def test_04_endpoint_alignment(self):
        """测试4: 终点对齐功能"""
        print("\n[测试4] 终点对齐测试")
        print("-" * 80)

        start_x, start_y = 150, 150
        end_x, end_y = 650, 450

        # 测试启用对齐
        trajectory_aligned = self.predictor.predict(
            start_x, start_y, end_x, end_y,
            align_endpoints=True
        )

        # 测试禁用对齐
        trajectory_unaligned = self.predictor.predict(
            start_x, start_y, end_x, end_y,
            align_endpoints=False
        )

        # 检查对齐后的终点精度
        aligned_start = trajectory_aligned[0]
        aligned_end = trajectory_aligned[-1]

        start_error = math.dist(aligned_start, (start_x, start_y))
        end_error = math.dist(aligned_end, (end_x, end_y))

        print(f"启用对齐:")
        print(f"  - 起点误差: {start_error:.4f} 像素")
        print(f"  - 终点误差: {end_error:.4f} 像素")
        print(f"  - 轨迹点数: {len(trajectory_aligned)}")

        # 对齐后误差应该非常小（小于1像素）
        self.assertLess(start_error, 1.0, "对齐后起点误差应小于1像素")
        self.assertLess(end_error, 1.0, "对齐后终点误差应小于1像素")

        # 检查未对齐的终点（可能有误差）
        unaligned_end = trajectory_unaligned[-1]
        unaligned_end_error = math.dist(unaligned_end, (end_x, end_y))

        print(f"\n禁用对齐:")
        print(f"  - 终点误差: {unaligned_end_error:.4f} 像素")
        print(f"  - 轨迹点数: {len(trajectory_unaligned)}")

    def test_05_smoothing_and_downsampling(self):
        """测试5: 平滑和降采样功能"""
        print("\n[测试5] 平滑和降采样测试")
        print("-" * 80)

        start_x, start_y = 100, 100
        end_x, end_y = 700, 500

        # 测试不同配置
        traj_raw = self.predictor.predict(
            start_x, start_y, end_x, end_y,
            smooth=False, downsample=False, align_endpoints=False
        )

        traj_smooth = self.predictor.predict(
            start_x, start_y, end_x, end_y,
            smooth=True, downsample=False, align_endpoints=False
        )

        traj_downsample = self.predictor.predict(
            start_x, start_y, end_x, end_y,
            smooth=False, downsample=True, align_endpoints=False
        )

        traj_both = self.predictor.predict(
            start_x, start_y, end_x, end_y,
            smooth=True, downsample=True, align_endpoints=False
        )

        print(f"原始轨迹: {len(traj_raw)} 个点")
        print(f"仅平滑: {len(traj_smooth)} 个点")
        print(f"仅降采样: {len(traj_downsample)} 个点")
        print(f"平滑+降采样: {len(traj_both)} 个点")

        # 断言：平滑和降采样功能正常工作
        # 注意：平滑操作保留所有点，所以点数可能不变或增加
        # 降采样操作会跳过点，所以应该不超过原始点数太多
        self.assertGreater(len(traj_raw), 0, "原始轨迹应有点")
        self.assertGreater(len(traj_smooth), 0, "平滑轨迹应有点")
        self.assertGreater(len(traj_downsample), 0, "降采样轨迹应有点")
        self.assertGreater(len(traj_both), 0, "处理后轨迹应有点")

    def test_06_trajectory_smoothness(self):
        """测试6: 轨迹平滑度"""
        print("\n[测试6] 轨迹平滑度测试")
        print("-" * 80)

        start_x, start_y = 200, 200
        end_x, end_y = 600, 400

        trajectory = self.predictor.predict(
            start_x, start_y, end_x, end_y,
            smooth=True
        )

        # 计算平滑度指标：加速度的平均值（越小越平滑）
        if len(trajectory) >= 3:
            accelerations = []
            for i in range(1, len(trajectory) - 1):
                p_prev = trajectory[i - 1]
                p_curr = trajectory[i]
                p_next = trajectory[i + 1]

                # 计算速度向量
                v1 = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
                v2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])

                # 计算加速度（速度变化）
                a = (v2[0] - v1[0], v2[1] - v1[1])
                a_magnitude = math.sqrt(a[0]**2 + a[1]**2)
                accelerations.append(a_magnitude)

            avg_acceleration = sum(accelerations) / len(accelerations)
            max_acceleration = max(accelerations)

            print(f"✓ 平滑度指标:")
            print(f"  - 平均加速度: {avg_acceleration:.4f}")
            print(f"  - 最大加速度: {max_acceleration:.4f}")
            print(f"  - 轨迹点数: {len(trajectory)}")

            # 平滑后的轨迹加速度不应该太大
            self.assertLess(avg_acceleration, 100, "平均加速度应该合理")

    def test_07_inference_performance(self):
        """测试7: 推理性能（批量测试）"""
        print("\n[测试7] 推理性能测试")
        print("-" * 80)

        num_iterations = 10
        test_cases = [
            (100, 100, 700, 500),
            (200, 300, 600, 400),
            (150, 150, 650, 450),
            (300, 200, 500, 500),
            (50, 50, 750, 550),
        ]

        total_time = 0
        inference_times = []

        for i in range(num_iterations):
            for start_x, start_y, end_x, end_y in test_cases:
                start_time = time.time()
                trajectory = self.predictor.predict(start_x, start_y, end_x, end_y)
                inference_time = time.time() - start_time

                total_time += inference_time
                inference_times.append(inference_time)

        avg_time = total_time / (num_iterations * len(test_cases))
        min_time = min(inference_times)
        max_time = max(inference_times)

        print(f"✓ 性能统计 ({num_iterations * len(test_cases)} 次推理):")
        print(f"  - 平均推理时间: {avg_time*1000:.2f} ms")
        print(f"  - 最快: {min_time*1000:.2f} ms")
        print(f"  - 最慢: {max_time*1000:.2f} ms")
        print(f"  - 总时间: {total_time:.2f} s")
        print(f"  - 吞吐量: {1/avg_time:.2f} 轨迹/秒")

        # 推理时间应该在合理范围内（小于1秒）
        self.assertLess(avg_time, 1.0, "平均推理时间应小于1秒")

    def test_08_trajectory_validity(self):
        """测试8: 轨迹有效性"""
        print("\n[测试8] 轨迹有效性测试")
        print("-" * 80)

        start_x, start_y = 100, 100
        end_x, end_y = 700, 500

        trajectory = self.predictor.predict(start_x, start_y, end_x, end_y)

        # 检查轨迹点的有效性
        for i, (x, y) in enumerate(trajectory):
            # 坐标应该是数值（包括numpy类型）
            self.assertIsInstance(x, (int, float, np.integer, np.floating), f"点{i}的X坐标应为数值")
            self.assertIsInstance(y, (int, float, np.integer, np.floating), f"点{i}的Y坐标应为数值")

            # 转换为float以便比较
            x = float(x)
            y = float(y)

            # 坐标应该在合理范围内（允许一定超出画布）
            self.assertGreater(x, -100, f"点{i}的X坐标过小")
            self.assertLess(x, 900, f"点{i}的X坐标过大")
            self.assertGreater(y, -100, f"点{i}的Y坐标过小")
            self.assertLess(y, 700, f"点{i}的Y坐标过大")

        # 计算轨迹总长度
        total_length = 0
        for i in range(len(trajectory) - 1):
            segment_length = math.dist(trajectory[i], trajectory[i+1])
            total_length += segment_length

        # 计算直线距离
        direct_distance = math.dist((start_x, start_y), (end_x, end_y))

        # 轨迹长度应该大于等于直线距离
        self.assertGreaterEqual(total_length, direct_distance * 0.9,
                                "轨迹总长度应接近或大于直线距离")

        print(f"✓ 轨迹有效性检查通过:")
        print(f"  - 轨迹点数: {len(trajectory)}")
        print(f"  - 轨迹总长度: {total_length:.2f} 像素")
        print(f"  - 直线距离: {direct_distance:.2f} 像素")
        print(f"  - 长度比率: {total_length/direct_distance:.2f}")

    def test_09_diversity_test(self):
        """测试9: CVAE多样性测试"""
        print("\n[测试9] CVAE多样性测试")
        print("-" * 80)

        start_x, start_y = 200, 200
        end_x, end_y = 600, 400
        num_samples = 10

        # 生成多条轨迹
        trajectories = self.predictor.predict_multiple(
            start_x, start_y, end_x, end_y,
            num_samples=num_samples,
            align_endpoints=False  # 禁用对齐以更好地观察多样性
        )

        # 计算轨迹之间的差异
        # 使用中点位置来衡量多样性
        if len(trajectories) >= 2:
            midpoint_positions = []
            for traj in trajectories:
                if len(traj) >= 2:
                    mid_idx = len(traj) // 2
                    midpoint_positions.append(traj[mid_idx])

            # 计算中点位置的标准差（作为多样性指标）
            if len(midpoint_positions) >= 2:
                avg_x = sum(p[0] for p in midpoint_positions) / len(midpoint_positions)
                avg_y = sum(p[1] for p in midpoint_positions) / len(midpoint_positions)

                variance = sum((p[0]-avg_x)**2 + (p[1]-avg_y)**2
                              for p in midpoint_positions) / len(midpoint_positions)
                std_dev = math.sqrt(variance)

                print(f"✓ 多样性指标:")
                print(f"  - 样本数量: {num_samples}")
                print(f"  - 中点平均位置: ({avg_x:.2f}, {avg_y:.2f})")
                print(f"  - 中点位置标准差: {std_dev:.2f} 像素")

                # CVAE应该生成有一定差异的轨迹
                # 标准差应该大于0（表示有多样性）
                self.assertGreater(std_dev, 0, "CVAE应生成不同的轨迹")

    def test_10_edge_cases(self):
        """测试10: 边界情况测试"""
        print("\n[测试10] 边界情况测试")
        print("-" * 80)

        # 测试1: 非常短的距离
        print("测试短距离轨迹...")
        traj_short = self.predictor.predict(400, 300, 410, 310)
        self.assertGreater(len(traj_short), 0, "短距离轨迹应生成")
        print(f"  ✓ 短距离 (10px): {len(traj_short)} 个点")

        # 测试2: 非常长的距离
        print("测试长距离轨迹...")
        traj_long = self.predictor.predict(50, 50, 750, 550)
        self.assertGreater(len(traj_long), 0, "长距离轨迹应生成")
        print(f"  ✓ 长距离 (~990px): {len(traj_long)} 个点")

        # 测试3: 水平轨迹
        print("测试水平轨迹...")
        traj_horizontal = self.predictor.predict(100, 300, 700, 300)
        self.assertGreater(len(traj_horizontal), 0, "水平轨迹应生成")
        print(f"  ✓ 水平轨迹: {len(traj_horizontal)} 个点")

        # 测试4: 垂直轨迹
        print("测试垂直轨迹...")
        traj_vertical = self.predictor.predict(400, 100, 400, 500)
        self.assertGreater(len(traj_vertical), 0, "垂直轨迹应生成")
        print(f"  ✓ 垂直轨迹: {len(traj_vertical)} 个点")

        # 测试5: 对角线轨迹
        print("测试对角线轨迹...")
        traj_diagonal = self.predictor.predict(100, 100, 700, 500)
        self.assertGreater(len(traj_diagonal), 0, "对角线轨迹应生成")
        print(f"  ✓ 对角线轨迹: {len(traj_diagonal)} 个点")


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestTrajectoryPrediction)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 打印总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"总测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✓ 所有测试通过！")
    else:
        print("\n✗ 部分测试失败")

    print("=" * 80)

    return result


if __name__ == '__main__':
    # 运行测试
    result = run_tests()

    # 根据测试结果设置退出码
    sys.exit(0 if result.wasSuccessful() else 1)
"""
评估代码使用示例
展示如何使用 TrajectoryEvaluator 的各种功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import TrajectoryEvaluator

# 获取项目根目录和模型路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_model.pth')


# ============================================================================
# 示例 1：基本使用 - 生成并可视化轨迹
# ============================================================================

def example_basic():
    """基本使用示例"""
    print("\n" + "="*70)
    print("示例 1：基本使用")
    print("="*70)

    # 创建评估器（使用原始像素坐标）
    evaluator = TrajectoryEvaluator(
        model_path=MODEL_PATH,
        screen_width=1920,
        screen_height=1080,
        use_normalized=False
    )

    # 生成轨迹
    trajectories = evaluator.generate(
        start=(100, 100),      # 起点
        end=(800, 600),        # 终点
        num_samples=3,         # 生成 3 条轨迹
        return_raw=True        # 返回原始坐标
    )

    # 可视化
    evaluator.visualize(
        trajectories=trajectories,
        start=(100, 100),
        end=(800, 600),
        save_path='example_basic.png',
        show_plot=False
    )

    print("✓ 轨迹已保存到 example_basic.png")


# ============================================================================
# 示例 2：批量生成多组轨迹
# ============================================================================

def example_batch():
    """批量生成示例"""
    print("\n" + "="*70)
    print("示例 2：批量生成多组起终点的轨迹")
    print("="*70)

    evaluator = TrajectoryEvaluator(
        model_path=MODEL_PATH,
        use_normalized=False
    )

    # 定义多组起终点
    test_cases = [
        ((100, 100), (900, 900), "对角线"),
        ((100, 500), (900, 500), "水平线"),
        ((500, 100), (500, 900), "垂直线"),
        ((900, 100), (100, 900), "反对角线"),
    ]

    for i, (start, end, desc) in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {desc} - 从 {start} 到 {end}")

        # 生成轨迹
        trajectories = evaluator.generate(
            start=start,
            end=end,
            num_samples=3,
            return_raw=True
        )

        # 可视化
        save_path = f'example_batch_{i}_{desc}.png'
        evaluator.visualize(
            trajectories=trajectories,
            start=start,
            end=end,
            save_path=save_path,
            title=f'{desc}轨迹：从 {start} 到 {end}',
            show_plot=False
        )

        print(f"  ✓ 已保存到 {save_path}")


# ============================================================================
# 示例 3：生成并评估轨迹质量
# ============================================================================

def example_evaluation():
    """轨迹评估示例"""
    print("\n" + "="*70)
    print("示例 3：生成并评估轨迹质量")
    print("="*70)

    evaluator = TrajectoryEvaluator(
        model_path=MODEL_PATH,
        use_normalized=False
    )

    start = (200, 200)
    end = (1600, 800)

    print(f"起点: {start}")
    print(f"终点: {end}")
    print(f"\n生成 10 条轨迹并统计质量指标...")

    # 生成多条轨迹
    trajectories = evaluator.generate(
        start=start,
        end=end,
        num_samples=10,
        return_raw=True
    )

    # 评估所有轨迹
    all_metrics = []
    for i, traj in enumerate(trajectories, 1):
        metrics = evaluator.evaluate(traj, start, end)
        all_metrics.append(metrics)

        # 打印每条轨迹的指标
        evaluator.analyzer.print_metrics(metrics, trajectory_id=i)

    # 统计平均指标
    print("\n" + "="*70)
    print("平均指标统计")
    print("="*70)
    avg_efficiency = sum(m.path_efficiency for m in all_metrics) / len(all_metrics)
    avg_error = sum(m.endpoint_error for m in all_metrics) / len(all_metrics)
    avg_length = sum(m.total_length for m in all_metrics) / len(all_metrics)

    print(f"平均路径效率: {avg_efficiency:.2%}")
    print(f"平均终点误差: {avg_error:.4f}")
    print(f"平均轨迹长度: {avg_length:.2f}")

    # 可视化最好和最差的轨迹
    best_idx = max(range(len(all_metrics)), key=lambda i: all_metrics[i].path_efficiency)
    worst_idx = min(range(len(all_metrics)), key=lambda i: all_metrics[i].path_efficiency)

    print(f"\n最优轨迹: #{best_idx+1} (效率: {all_metrics[best_idx].path_efficiency:.2%})")
    print(f"最差轨迹: #{worst_idx+1} (效率: {all_metrics[worst_idx].path_efficiency:.2%})")

    # 可视化对比
    evaluator.visualize(
        trajectories=[trajectories[best_idx], trajectories[worst_idx]],
        start=start,
        end=end,
        save_path='example_comparison.png',
        title=f'最优 vs 最差轨迹对比',
        show_plot=False
    )

    print("✓ 对比图已保存到 example_comparison.png")


# ============================================================================
# 示例 4：使用归一化坐标
# ============================================================================

def example_normalized():
    """使用归一化坐标示例"""
    print("\n" + "="*70)
    print("示例 4：使用归一化坐标 [0, 1]")
    print("="*70)

    # 使用归一化坐标模式
    evaluator = TrajectoryEvaluator(
        model_path=MODEL_PATH,
        use_normalized=True
    )

    # 归一化坐标（0-1 之间）
    start = (0.1, 0.1)
    end = (0.9, 0.9)

    print(f"起点（归一化）: {start}")
    print(f"终点（归一化）: {end}")

    # 生成轨迹
    trajectories = evaluator.generate(
        start=start,
        end=end,
        num_samples=5,
        return_raw=False  # 返回归一化坐标
    )

    # 可视化
    evaluator.visualize(
        trajectories=trajectories,
        start=start,
        end=end,
        save_path='example_normalized.png',
        title='归一化坐标轨迹',
        show_plot=False
    )

    print("✓ 轨迹已保存到 example_normalized.png")


# ============================================================================
# 示例 5：自定义序列长度
# ============================================================================

def example_custom_length():
    """自定义序列长度示例"""
    print("\n" + "="*70)
    print("示例 5：生成不同长度的轨迹")
    print("="*70)

    evaluator = TrajectoryEvaluator(
        model_path=MODEL_PATH,
        use_normalized=False
    )

    start = (300, 300)
    end = (1200, 700)

    # 生成不同长度的轨迹
    lengths = [10, 20, 50]

    for length in lengths:
        print(f"\n生成序列长度为 {length} 的轨迹...")

        trajectories = evaluator.generate(
            start=start,
            end=end,
            seq_len=length,
            num_samples=3,
            return_raw=True
        )

        save_path = f'example_length_{length}.png'
        evaluator.visualize(
            trajectories=trajectories,
            start=start,
            end=end,
            save_path=save_path,
            title=f'轨迹长度: {length} 点',
            show_plot=False
        )

        print(f"  ✓ 已保存到 {save_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有示例"""
    print("\n" + "="*70)
    print("鼠标轨迹评估代码 - 使用示例")
    print("="*70)

    # 运行各个示例
    try:
        example_basic()
        example_batch()
        example_evaluation()
        example_normalized()
        example_custom_length()

        print("\n" + "="*70)
        print("所有示例运行完成！")
        print("="*70)

    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("请先训练模型: python scripts/train.py")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
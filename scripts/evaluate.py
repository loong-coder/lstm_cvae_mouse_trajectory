"""
轨迹评估与生成模块
提供清晰的 API 用于生成和可视化鼠标轨迹

使用示例:
    evaluator = TrajectoryEvaluator(model_path='models/best_model.pth')
    trajectories = evaluator.generate(start=(100, 100), end=(800, 600), num_samples=5)
    evaluator.visualize(trajectories, save_path='output.png')
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.lstm_cvae import TrajectoryCVAE
from config.config import Config

# 配置 matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class TrajectoryMetrics:
    """轨迹评估指标"""
    total_length: float          # 轨迹总长度
    direct_distance: float       # 起终点直线距离
    path_efficiency: float       # 路径效率 (直线距离/轨迹长度)
    endpoint_error: float        # 终点误差
    avg_velocity: float          # 平均速度
    avg_acceleration: float      # 平均加速度
    num_points: int              # 轨迹点数


# ============================================================================
# 第 1 层：数据预处理层
# ============================================================================

class CoordinateNormalizer:
    """坐标归一化器 - 处理原始坐标与归一化坐标的转换"""

    def __init__(self, screen_width: float = 1920, screen_height: float = 1080):
        """
        Args:
            screen_width: 屏幕宽度（像素）
            screen_height: 屏幕高度（像素）
        """
        self.screen_width = screen_width
        self.screen_height = screen_height

    def normalize(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """将原始坐标归一化到 [0, 1] 范围

        Args:
            point: (x, y) 原始坐标

        Returns:
            (x_norm, y_norm) 归一化坐标
        """
        x, y = point
        x_norm = x / self.screen_width
        y_norm = y / self.screen_height
        return (x_norm, y_norm)

    def denormalize(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """将归一化坐标还原为原始坐标

        Args:
            point: (x_norm, y_norm) 归一化坐标

        Returns:
            (x, y) 原始坐标
        """
        x_norm, y_norm = point
        x = x_norm * self.screen_width
        y = y_norm * self.screen_height
        return (x, y)

    def denormalize_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """批量反归一化轨迹坐标

        Args:
            trajectory: (N, 2) 归一化轨迹坐标

        Returns:
            (N, 2) 原始轨迹坐标
        """
        denorm = trajectory.copy()
        denorm[:, 0] *= self.screen_width
        denorm[:, 1] *= self.screen_height
        return denorm


# ============================================================================
# 第 2 层：模型加载与推理层
# ============================================================================

class ModelInference:
    """模型推理器 - 负责加载模型和生成轨迹"""

    def __init__(self, model_path: str):
        """
        Args:
            model_path: 模型文件路径
        """
        self.model_path = model_path
        self.model = None
        self.config = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._load_model()

    def _load_model(self):
        """加载训练好的模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"模型文件不存在: {self.model_path}\n"
                f"请先运行训练脚本: python scripts/train.py"
            )

        print(f"正在加载模型: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        self.config = checkpoint['config']
        self.model = TrajectoryCVAE(
            feat_dim=self.config.FEAT_DIM,
            cond_dim=self.config.COND_DIM,
            latent_dim=self.config.LATENT_DIM,
            hidden_dim=self.config.HIDDEN_DIM
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"✓ 模型加载成功")
        print(f"  训练轮数: {checkpoint['epoch'] + 1}")
        print(f"  验证损失: {checkpoint['best_val_loss']:.4f}")

    def generate_trajectories(
        self,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        seq_len: Optional[int] = None,
        num_samples: int = 1,
        auto_calculate: bool = False,
        use_model_prediction: bool = False,
        pixels_per_step: float = 30.0
    ) -> List[np.ndarray]:
        """生成轨迹（归一化坐标）

        Args:
            start_point: (x, y) 起点坐标（归一化）
            end_point: (x, y) 终点坐标（归一化）
            seq_len: 序列长度，如果为None则根据其他参数决定
            num_samples: 生成的轨迹数量
            auto_calculate: 是否使用公式根据距离自动计算点数
            use_model_prediction: 是否使用模型预测的长度（优先级最高）
            pixels_per_step: 每步移动的平均像素数（用于公式计算）

        Returns:
            List[(seq_len, 2)] 生成的轨迹坐标列表（归一化）
        """
        # 构建条件向量 [start_x, start_y, end_x, end_y]
        condition = torch.tensor(
            [[start_point[0], start_point[1], end_point[0], end_point[1]]],
            dtype=torch.float32,
            device=self.device
        )  # [1, 4]

        trajectories = []

        with torch.no_grad():
            for _ in range(num_samples):
                # 决定序列长度的优先级：
                # 1. 使用模型预测（如果启用）
                # 2. 使用指定的 seq_len
                # 3. 使用公式自动计算（如果启用）
                # 4. 使用默认值

                if use_model_prediction:
                    # 使用模型预测长度
                    generated = self.model.inference(condition, use_predicted_len=True)
                    actual_seq_len = generated.shape[1]
                    print(f"  模型预测点数: {actual_seq_len} 个点")
                elif seq_len is not None:
                    # 使用指定长度
                    generated = self.model.inference(condition, seq_len=seq_len, use_predicted_len=False)
                    actual_seq_len = seq_len
                elif auto_calculate:
                    # 使用公式计算长度
                    calculated_len = TrajectoryCVAE.calculate_seq_len(
                        start=start_point,
                        end=end_point,
                        pixels_per_step=pixels_per_step
                    )
                    generated = self.model.inference(condition, seq_len=calculated_len, use_predicted_len=False)
                    actual_seq_len = calculated_len
                    print(f"  公式计算点数: {actual_seq_len} 个点")
                else:
                    # 使用默认长度
                    default_len = getattr(self.config, 'SEQ_LEN', 20)  # 兼容旧配置
                    if not hasattr(self.config, 'SEQ_LEN'):
                        # 如果使用新配置，尝试使用 MAX_SEQ_LEN 的一半作为默认值
                        default_len = getattr(self.config, 'MAX_SEQ_LEN', 50) // 2
                    generated = self.model.inference(condition, seq_len=default_len, use_predicted_len=False)
                    actual_seq_len = default_len

                # 提取坐标 (current_x, current_y)
                coords = generated[0, :, 0:2].cpu().numpy()  # (actual_seq_len, 2)
                trajectories.append(coords)

        return trajectories


# ============================================================================
# 第 3 层：轨迹评估层
# ============================================================================

class TrajectoryAnalyzer:
    """轨迹分析器 - 计算轨迹质量指标"""

    @staticmethod
    def calculate_metrics(
        trajectory: np.ndarray,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float]
    ) -> TrajectoryMetrics:
        """计算轨迹的各项评估指标

        Args:
            trajectory: (N, 2) 轨迹坐标
            start_point: (x, y) 起点坐标
            end_point: (x, y) 终点坐标

        Returns:
            TrajectoryMetrics 对象
        """
        # 计算轨迹总长度
        displacements = np.diff(trajectory, axis=0)  # (N-1, 2)
        distances = np.sqrt(np.sum(displacements ** 2, axis=1))  # (N-1,)
        total_length = np.sum(distances)

        # 计算起终点直线距离
        direct_distance = np.sqrt(
            (end_point[0] - start_point[0]) ** 2 +
            (end_point[1] - start_point[1]) ** 2
        )

        # 路径效率
        path_efficiency = direct_distance / total_length if total_length > 0 else 0

        # 终点误差
        final_point = trajectory[-1]
        endpoint_error = np.sqrt(
            (final_point[0] - end_point[0]) ** 2 +
            (final_point[1] - end_point[1]) ** 2
        )

        # 平均速度（假设时间间隔均匀）
        avg_velocity = np.mean(distances) if len(distances) > 0 else 0

        # 平均加速度
        velocity_changes = np.diff(distances)
        avg_acceleration = np.mean(np.abs(velocity_changes)) if len(velocity_changes) > 0 else 0

        return TrajectoryMetrics(
            total_length=total_length,
            direct_distance=direct_distance,
            path_efficiency=path_efficiency,
            endpoint_error=endpoint_error,
            avg_velocity=avg_velocity,
            avg_acceleration=avg_acceleration,
            num_points=len(trajectory)
        )

    @staticmethod
    def print_metrics(metrics: TrajectoryMetrics, trajectory_id: int = 1):
        """打印轨迹评估指标"""
        print(f"\n【轨迹 {trajectory_id} 评估指标】")
        print(f"  轨迹点数: {metrics.num_points}")
        print(f"  轨迹长度: {metrics.total_length:.2f}")
        print(f"  直线距离: {metrics.direct_distance:.2f}")
        print(f"  路径效率: {metrics.path_efficiency:.2%}")
        print(f"  终点误差: {metrics.endpoint_error:.4f}")
        print(f"  平均速度: {metrics.avg_velocity:.4f}")
        print(f"  平均加速度: {metrics.avg_acceleration:.4f}")


# ============================================================================
# 第 4 层：可视化层
# ============================================================================

class TrajectoryVisualizer:
    """轨迹可视化器 - 绘制轨迹图"""

    @staticmethod
    def plot(
        trajectories: List[np.ndarray],
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        save_path: str = 'trajectory.png',
        title: str = '生成的鼠标轨迹',
        show_plot: bool = True,
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 150
    ):
        """绘制轨迹图

        Args:
            trajectories: List[(N, 2)] 轨迹坐标列表
            start_point: (x, y) 起点坐标
            end_point: (x, y) 终点坐标
            save_path: 图片保存路径
            title: 图表标题
            show_plot: 是否显示图表
            figsize: 图表大小
            dpi: 图片分辨率
        """
        plt.figure(figsize=figsize)

        # 绘制每条轨迹
        colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))
        for i, (trajectory, color) in enumerate(zip(trajectories, colors)):
            # 获取轨迹点数
            num_points = len(trajectory)

            # 绘制轨迹线
            plt.plot(
                trajectory[:, 0], trajectory[:, 1],
                linewidth=2, alpha=0.7, color=color,
                label=f'轨迹 {i+1} ({num_points}点)'
            )
            # 绘制轨迹点
            plt.scatter(
                trajectory[:, 0], trajectory[:, 1],
                s=15, alpha=0.4, color=color
            )

        # 绘制起点（绿色大圆）
        plt.scatter(
            start_point[0], start_point[1],
            c='green', s=200, marker='o',
            edgecolors='black', linewidths=2,
            label='起点', zorder=10
        )

        # 绘制终点（红色大圆）
        plt.scatter(
            end_point[0], end_point[1],
            c='red', s=200, marker='o',
            edgecolors='black', linewidths=2,
            label='终点', zorder=10
        )

        # 绘制直线路径参考
        plt.plot(
            [start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            'k--', alpha=0.3, linewidth=1.5,
            label='直线路径'
        )

        # 设置图表属性
        plt.xlabel('X 坐标', fontsize=12)
        plt.ylabel('Y 坐标', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.axis('equal')  # 保持纵横比

        plt.tight_layout()

        # 保存图片
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ 轨迹图已保存: {save_path}")

        # 显示图片
        if show_plot:
            plt.show()
        else:
            plt.close()


# ============================================================================
# 第 5 层：综合评估器（对外 API）
# ============================================================================

class TrajectoryEvaluator:
    """轨迹评估器 - 提供统一的评估接口"""

    def __init__(
        self,
        model_path: str = 'models/best_model.pth',
        screen_width: float = 1920,
        screen_height: float = 1080,
        use_normalized: bool = False
    ):
        """
        Args:
            model_path: 模型文件路径
            screen_width: 屏幕宽度（用于坐标转换）
            screen_height: 屏幕高度（用于坐标转换）
            use_normalized: 是否直接使用归一化坐标（True）还是原始坐标（False）
        """
        self.model_inference = ModelInference(model_path)
        self.normalizer = CoordinateNormalizer(screen_width, screen_height)
        self.analyzer = TrajectoryAnalyzer()
        self.visualizer = TrajectoryVisualizer()
        self.use_normalized = use_normalized

    def generate(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        seq_len: Optional[int] = None,
        num_samples: int = 1,
        return_raw: bool = False,
        auto_seq_len: bool = False,
        use_model_prediction: bool = True,
        pixels_per_step: float = 30.0
    ) -> List[np.ndarray]:
        """生成轨迹

        Args:
            start: (x, y) 起点坐标
            end: (x, y) 终点坐标
            seq_len: 序列长度（如果指定则优先使用）
            num_samples: 生成的轨迹数量
            return_raw: 是否返回原始坐标（True）还是归一化坐标（False）
            use_model_prediction: 是否使用模型预测的长度（默认True，推荐）
            auto_seq_len: 是否使用公式根据起终点距离自动计算点数（默认False）
            pixels_per_step: 每步移动的平均像素数（用于公式计算）

        Returns:
            List[(N, 2)] 轨迹坐标列表

        注意：
            序列长度决定优先级：
            1. seq_len（如果指定）
            2. use_model_prediction（如果启用且seq_len未指定）
            3. auto_seq_len（如果启用且前两者都未使用）
            4. 默认值
        """
        # 坐标归一化（如果需要）
        if not self.use_normalized:
            start_norm = self.normalizer.normalize(start)
            end_norm = self.normalizer.normalize(end)
            # 自动计算时使用原始坐标计算距离
            start_for_calc = start
            end_for_calc = end
        else:
            start_norm = start
            end_norm = end
            # 归一化坐标需要转换为像素坐标计算距离
            start_for_calc = self.normalizer.denormalize(start)
            end_for_calc = self.normalizer.denormalize(end)

        # 决定使用哪种长度计算方式
        # 优先级：指定长度 > 模型预测 > 公式计算 > 默认值
        use_model_pred = use_model_prediction and (seq_len is None)
        auto_calculate = auto_seq_len and (seq_len is None) and (not use_model_pred)

        # 生成轨迹
        trajectories = self.model_inference.generate_trajectories(
            start_point=start_norm,
            end_point=end_norm,
            seq_len=seq_len,
            num_samples=num_samples,
            auto_calculate=auto_calculate,
            use_model_prediction=use_model_pred,
            pixels_per_step=pixels_per_step
        )

        # 坐标反归一化（如果需要）
        if not self.use_normalized and return_raw:
            trajectories = [
                self.normalizer.denormalize_trajectory(traj)
                for traj in trajectories
            ]

        return trajectories

    def evaluate(
        self,
        trajectory: np.ndarray,
        start: Tuple[float, float],
        end: Tuple[float, float]
    ) -> TrajectoryMetrics:
        """评估单条轨迹

        Args:
            trajectory: (N, 2) 轨迹坐标
            start: (x, y) 起点坐标
            end: (x, y) 终点坐标

        Returns:
            TrajectoryMetrics 评估指标
        """
        return self.analyzer.calculate_metrics(trajectory, start, end)

    def visualize(
        self,
        trajectories: List[np.ndarray],
        start: Tuple[float, float],
        end: Tuple[float, float],
        save_path: str = 'trajectory.png',
        **kwargs
    ):
        """可视化轨迹

        Args:
            trajectories: List[(N, 2)] 轨迹坐标列表
            start: (x, y) 起点坐标
            end: (x, y) 终点坐标
            save_path: 保存路径
            **kwargs: 其他可视化参数
        """
        self.visualizer.plot(trajectories, start, end, save_path, **kwargs)


# ============================================================================
# 第 6 层：主函数（使用示例）
# ============================================================================

def main():
    """主函数 - 演示如何使用评估器"""
    print("=" * 70)
    print("鼠标轨迹生成与评估系统")
    print("=" * 70)

    # 1. 初始化评估器
    # 使用项目根目录的绝对路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, 'models', 'best_model.pth')

    evaluator = TrajectoryEvaluator(
        model_path=model_path,
        screen_width=1920,
        screen_height=1080,
        use_normalized=False  # 使用原始像素坐标
    )

    # 2. 设置起点和终点（像素坐标）
    start_point = (200, 200)
    end_point = (1600, 800)

    print(f"\n起点: {start_point}")
    print(f"终点: {end_point}")

    # 3. 生成轨迹
    print(f"\n正在生成 5 条轨迹...")
    trajectories = evaluator.generate(
        start=start_point,
        end=end_point,
        num_samples=5,
        auto_seq_len=True,
        return_raw=True  # 返回原始坐标
    )

    # 4. 评估每条轨迹
    print("\n" + "=" * 70)
    print("轨迹评估结果")
    print("=" * 70)

    for i, trajectory in enumerate(trajectories, 1):
        metrics = evaluator.evaluate(trajectory, start_point, end_point)
        evaluator.analyzer.print_metrics(metrics, trajectory_id=i)

    # 5. 可视化轨迹
    print("\n" + "=" * 70)
    print("正在生成可视化图表...")
    print("=" * 70)

    evaluator.visualize(
        trajectories=trajectories,
        start=start_point,
        end=end_point,
        save_path='generated_trajectories.png',
        title=f'鼠标轨迹：从 {start_point} 到 {end_point}',
        show_plot=True
    )

    print("\n✓ 评估完成！")


if __name__ == '__main__':
    main()
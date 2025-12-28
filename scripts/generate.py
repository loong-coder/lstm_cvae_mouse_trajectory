"""
简单的轨迹生成脚本
使用方法：python scripts/generate.py
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.lstm_cvae import TrajectoryCVAE
from config.config import Config

# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False


def load_model(model_path='models/best_model.pth'):
    """加载训练好的模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}\n请先训练模型！")

    print(f"加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    config = checkpoint['config']
    model = TrajectoryCVAE(
        feat_dim=config.FEAT_DIM,
        cond_dim=config.COND_DIM,
        latent_dim=config.LATENT_DIM,
        hidden_dim=config.HIDDEN_DIM
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ 模型加载成功")
    print(f"  训练轮数: {checkpoint['epoch'] + 1}")
    print(f"  最佳验证损失: {checkpoint['best_val_loss']:.4f}\n")

    return model, config


def generate_trajectory(model, start_point, end_point, seq_len=20, num_samples=1):
    """
    生成轨迹

    Args:
        model: 训练好的模型
        start_point: [x, y] 起点坐标（归一化后）
        end_point: [x, y] 终点坐标（归一化后）
        seq_len: 序列长度
        num_samples: 生成的轨迹数量

    Returns:
        trajectories: List[(seq_len, 2)] 生成的轨迹坐标（归一化后）
    """
    start_tensor = torch.tensor([start_point], dtype=torch.float32)
    end_tensor = torch.tensor([end_point], dtype=torch.float32)

    # 构建条件向量 [start_x, start_y, end_x, end_y]
    c = torch.cat([start_tensor, end_tensor], dim=1)  # [1, 4]

    trajectories = []

    with torch.no_grad():
        for i in range(num_samples):
            # 生成轨迹
            generated = model.inference(c, seq_len=seq_len)  # (1, seq_len, 5)

            # 提取坐标 (current_x, current_y)
            coords = generated[0, :, 0:2].numpy()  # (seq_len, 2)
            trajectories.append(coords)

    return trajectories


def plot_trajectories(trajectories, start_point, end_point, save_path='generated_trajectory.png'):
    """
    绘制轨迹图

    Args:
        trajectories: List[(seq_len, 2)] 轨迹坐标列表（归一化后）
        start_point: [x, y] 起点坐标（归一化后）
        end_point: [x, y] 终点坐标（归一化后）
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))

    # 绘制轨迹
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))
    for i, (trajectory, color) in enumerate(zip(trajectories, colors)):
        plt.plot(trajectory[:, 0], trajectory[:, 1],
                linewidth=2, alpha=0.7, color=color, label=f'轨迹 {i+1}')
        plt.scatter(trajectory[:, 0], trajectory[:, 1],
                   s=10, alpha=0.3, color=color)

    # 绘制起点（绿色）
    plt.scatter(start_point[0], start_point[1],
               c='green', s=200, marker='o',
               edgecolors='black', linewidths=2,
               label='起点', zorder=5)

    # 绘制终点（红色）
    plt.scatter(end_point[0], end_point[1],
               c='red', s=200, marker='o',
               edgecolors='black', linewidths=2,
               label='终点', zorder=5)

    # 绘制直线参考
    plt.plot([start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            'k--', alpha=0.3, linewidth=1, label='直线路径')

    # 设置图形
    plt.xlabel('X 坐标 (归一化)', fontsize=12)
    plt.ylabel('Y 坐标 (归一化)', fontsize=12)
    plt.title('生成的鼠标移动轨迹', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 轨迹图已保存: {save_path}")

    # 显示图片
    plt.show()


def main():
    """主函数"""
    print("="*60)
    print("鼠标轨迹生成器")
    print("="*60)

    # 加载模型
    try:
        model, config = load_model()
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return

    # 设置起点和终点（归一化坐标，范围 0-1）
    print("\n请输入轨迹的起点和终点坐标（归一化坐标，范围0-1）：")
    print("示例：起点 (0.1, 0.1)  终点 (0.8, 0.8)")
    print()

    try:
        start_input = input("起点坐标 (格式: x,y，如 0.1,0.1，按Enter使用默认): ").strip()
        if not start_input:
            start_point = [0.1, 0.1]
            print(f"使用默认起点: {start_point}")
        else:
            start_x, start_y = map(float, start_input.split(','))
            start_point = [start_x, start_y]

        end_input = input("终点坐标 (格式: x,y，如 0.8,0.8，按Enter使用默认): ").strip()
        if not end_input:
            end_point = [0.8, 0.8]
            print(f"使用默认终点: {end_point}")
        else:
            end_x, end_y = map(float, end_input.split(','))
            end_point = [end_x, end_y]

        # 生成数量
        num_samples_input = input("生成轨迹数量 (默认3条): ").strip()
        if not num_samples_input:
            num_samples = 3
        else:
            num_samples = int(num_samples_input)

        # 序列长度
        seq_len_input = input(f"序列长度 (默认{config.SEQ_LEN}): ").strip()
        if not seq_len_input:
            seq_len = config.SEQ_LEN
        else:
            seq_len = int(seq_len_input)

    except ValueError as e:
        print(f"输入格式错误: {e}")
        print("使用默认值进行演示...")
        start_point = [0.1, 0.1]
        end_point = [0.8, 0.8]
        num_samples = 3
        seq_len = config.SEQ_LEN

    print(f"\n生成轨迹中...")
    print(f"  起点: ({start_point[0]}, {start_point[1]})")
    print(f"  终点: ({end_point[0]}, {end_point[1]})")
    print(f"  数量: {num_samples} 条")
    print(f"  序列长度: {seq_len}")
    print()

    # 生成轨迹
    trajectories = generate_trajectory(
        model, start_point, end_point,
        seq_len=seq_len,
        num_samples=num_samples
    )

    # 输出轨迹信息
    for i, traj in enumerate(trajectories):
        print(f"轨迹 {i+1}: {len(traj)} 个点")
        # 计算轨迹长度
        total_distance = np.sum(np.sqrt(np.sum(np.diff(traj, axis=0)**2, axis=1)))
        direct_distance = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
        efficiency = direct_distance / total_distance if total_distance > 0 else 0
        print(f"  路径效率: {efficiency:.2%}")
        # 最终位置与终点的距离
        final_pos = traj[-1]
        endpoint_error = np.sqrt((final_pos[0] - end_point[0])**2 + (final_pos[1] - end_point[1])**2)
        print(f"  终点误差: {endpoint_error:.4f}")

    print()

    # 绘制轨迹
    plot_trajectories(trajectories, start_point, end_point, save_path='generated_trajectory.png')

    print("\n完成！")


if __name__ == '__main__':
    main()
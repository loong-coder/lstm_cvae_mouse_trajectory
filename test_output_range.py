"""
测试模型输出范围约束

验证添加激活函数后，模型输出是否在预期范围内：
- X, Y: [-1, 1] (相对坐标归一化值)
- Time: [0, +∞) (非负)
- Direction: [-1, 1] (对应 [-π, π])
- Speed: [0, 1] (归一化速度)
"""

import torch
import numpy as np
from cvae_model import TrajectoryPredictorCVAE
from trajectory_dataset import MAX_TRAJECTORY_LENGTH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}\n")

# 创建模型
model = TrajectoryPredictorCVAE(max_len=MAX_TRAJECTORY_LENGTH).to(device)
model.eval()

print("=" * 80)
print("测试模型输出范围约束")
print("=" * 80)

# 准备测试输入
batch_size = 10
task_context = torch.randn(batch_size, 5).to(device)  # 随机任务上下文

print(f"\n测试配置:")
print(f"  - Batch size: {batch_size}")
print(f"  - 最大轨迹长度: {MAX_TRAJECTORY_LENGTH}")
print(f"  - 输出特征维度: 5 [X, Y, Time, Direction, Speed]")

# 进行推理
with torch.no_grad():
    reconstruction, mu, logvar, predicted_length_ratio = model(task_context)

# 提取各个特征
x_values = reconstruction[:, :, 0].cpu().numpy()  # (batch, max_len)
y_values = reconstruction[:, :, 1].cpu().numpy()
time_values = reconstruction[:, :, 2].cpu().numpy()
direction_values = reconstruction[:, :, 3].cpu().numpy()
speed_values = reconstruction[:, :, 4].cpu().numpy()

print("\n" + "=" * 80)
print("输出范围统计")
print("=" * 80)

# X 坐标
print(f"\n【X 坐标】(相对坐标，应在 [-1, 1])")
print(f"  - 最小值: {x_values.min():.6f}")
print(f"  - 最大值: {x_values.max():.6f}")
print(f"  - 平均值: {x_values.mean():.6f}")
print(f"  - 标准差: {x_values.std():.6f}")
if x_values.min() >= -1.0 and x_values.max() <= 1.0:
    print(f"  ✓ 范围正确")
else:
    print(f"  ✗ 范围超出限制！")

# Y 坐标
print(f"\n【Y 坐标】(相对坐标，应在 [-1, 1])")
print(f"  - 最小值: {y_values.min():.6f}")
print(f"  - 最大值: {y_values.max():.6f}")
print(f"  - 平均值: {y_values.mean():.6f}")
print(f"  - 标准差: {y_values.std():.6f}")
if y_values.min() >= -1.0 and y_values.max() <= 1.0:
    print(f"  ✓ 范围正确")
else:
    print(f"  ✗ 范围超出限制！")

# Time
print(f"\n【Time】(时间，应为非负)")
print(f"  - 最小值: {time_values.min():.6f}")
print(f"  - 最大值: {time_values.max():.6f}")
print(f"  - 平均值: {time_values.mean():.6f}")
print(f"  - 标准差: {time_values.std():.6f}")
if time_values.min() >= 0.0:
    print(f"  ✓ 范围正确 (非负)")
else:
    print(f"  ✗ 范围错误 (出现负值)！")

# Direction
print(f"\n【Direction】(方向，应在 [-1, 1]，对应 [-π, π])")
print(f"  - 最小值: {direction_values.min():.6f}")
print(f"  - 最大值: {direction_values.max():.6f}")
print(f"  - 平均值: {direction_values.mean():.6f}")
print(f"  - 标准差: {direction_values.std():.6f}")
if direction_values.min() >= -1.0 and direction_values.max() <= 1.0:
    print(f"  ✓ 范围正确")
else:
    print(f"  ✗ 范围超出限制！")

# Speed
print(f"\n【Speed】(速度，应在 [0, 1])")
print(f"  - 最小值: {speed_values.min():.6f}")
print(f"  - 最大值: {speed_values.max():.6f}")
print(f"  - 平均值: {speed_values.mean():.6f}")
print(f"  - 标准差: {speed_values.std():.6f}")
if speed_values.min() >= 0.0 and speed_values.max() <= 1.0:
    print(f"  ✓ 范围正确")
else:
    print(f"  ✗ 范围超出限制！")

# 测试反归一化后的绝对坐标范围
print("\n" + "=" * 80)
print("反归一化后的绝对坐标测试")
print("=" * 80)

CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600

# 模拟不同的起点
test_start_points = [
    (100, 100),
    (400, 300),
    (700, 500),
    (50, 50),
    (750, 550)
]

print(f"\n画布尺寸: {CANVAS_WIDTH} x {CANVAS_HEIGHT}")
print(f"测试起点数量: {len(test_start_points)}\n")

all_abs_x = []
all_abs_y = []

for start_x, start_y in test_start_points:
    # 转换为绝对坐标
    # 相对坐标 = 归一化值 * 画布尺寸
    # 绝对坐标 = 相对坐标 + 起点
    for i in range(batch_size):
        rel_x = x_values[i] * CANVAS_WIDTH  # 反归一化
        rel_y = y_values[i] * CANVAS_HEIGHT

        abs_x = rel_x + start_x
        abs_y = rel_y + start_y

        all_abs_x.extend(abs_x)
        all_abs_y.extend(abs_y)

all_abs_x = np.array(all_abs_x)
all_abs_y = np.array(all_abs_y)

print(f"【绝对 X 坐标】")
print(f"  - 最小值: {all_abs_x.min():.2f}")
print(f"  - 最大值: {all_abs_x.max():.2f}")
print(f"  - 超出范围 [0, {CANVAS_WIDTH}] 的点数: {np.sum((all_abs_x < 0) | (all_abs_x > CANVAS_WIDTH))}")

print(f"\n【绝对 Y 坐标】")
print(f"  - 最小值: {all_abs_y.min():.2f}")
print(f"  - 最大值: {all_abs_y.max():.2f}")
print(f"  - 超出范围 [0, {CANVAS_HEIGHT}] 的点数: {np.sum((all_abs_y < 0) | (all_abs_y > CANVAS_HEIGHT))}")

# 统计超出范围的比例
total_points = len(all_abs_x)
out_of_bound_x = np.sum((all_abs_x < 0) | (all_abs_x > CANVAS_WIDTH))
out_of_bound_y = np.sum((all_abs_y < 0) | (all_abs_y > CANVAS_HEIGHT))
out_of_bound_total = np.sum((all_abs_x < 0) | (all_abs_x > CANVAS_WIDTH) |
                             (all_abs_y < 0) | (all_abs_y > CANVAS_HEIGHT))

print(f"\n【超出范围统计】")
print(f"  - 总点数: {total_points}")
print(f"  - X 超出: {out_of_bound_x} ({out_of_bound_x/total_points*100:.2f}%)")
print(f"  - Y 超出: {out_of_bound_y} ({out_of_bound_y/total_points*100:.2f}%)")
print(f"  - 总超出: {out_of_bound_total} ({out_of_bound_total/total_points*100:.2f}%)")

print("\n" + "=" * 80)
print("总结")
print("=" * 80)

# 检查所有约束
all_constraints_ok = (
    x_values.min() >= -1.0 and x_values.max() <= 1.0 and
    y_values.min() >= -1.0 and y_values.max() <= 1.0 and
    time_values.min() >= 0.0 and
    direction_values.min() >= -1.0 and direction_values.max() <= 1.0 and
    speed_values.min() >= 0.0 and speed_values.max() <= 1.0
)

if all_constraints_ok:
    print("\n✓ 所有激活函数约束都正常工作！")
    print("✓ 模型输出在预期范围内")
else:
    print("\n✗ 某些约束失败！请检查模型实现")

print(f"\n说明:")
print(f"  - 归一化相对坐标在 [-1, 1] 范围内")
print(f"  - 反归一化后可能超出画布范围（取决于起点位置）")
print(f"  - 这是正常的，会在推理时通过边界检查过滤")
print(f"  - 或通过仿射变换对齐到正确的终点")

print("\n" + "=" * 80)
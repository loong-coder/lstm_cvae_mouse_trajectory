# 鼠标轨迹预测系统 - LSTM + CVAE

> 基于深度学习的鼠标轨迹预测与生成系统，使用 LSTM 和 CVAE（条件变分自编码器）组合架构

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 目录

- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [功能特点](#功能特点)
- [安装指南](#安装指南)
- [使用教程](#使用教程)
- [模块参考](#模块参考)
- [API 文档](#api-文档)
- [配置说明](#配置说明)
- [开发指南](#开发指南)
- [常见问题](#常见问题)

---

## 🚀 快速开始

### 三步上手

```bash
# 1. 安装项目
pip install -e .

# 2. 收集数据
trajectory-collect

# 3. 训练模型
trajectory-train

# 4. 评估模型
trajectory-evaluate
```

### 或使用脚本方式

```bash
# 收集数据
python scripts/collect_data.py

# 训练模型
python scripts/train.py

# 评估模型
python scripts/evaluate.py
```

---

## 📁 项目结构

```
lstm_cvae_mouse_trajectory/
│
├── 📦 src/                          # 源代码包
│   ├── models/                      # 模型定义
│   │   └── lstm_cvae.py            # LSTM-CVAE 主模型 + 长度预测器
│   ├── data/                        # 数据处理
│   │   └── dataset.py              # 数据集类和加载器
│   ├── training/                    # 训练模块
│   │   └── trainer.py              # 训练器类（含 TensorBoard）
│   ├── utils/                       # 工具函数
│   │   └── trajectory_utils.py     # 轨迹提取和分析工具
│   └── gui/                         # GUI 应用
│       ├── collector.py            # 数据收集界面
│       └── evaluator.py            # 评估界面（人类 vs AI）
│
├── ⚙️ config/                        # 配置模块
│   └── config.py                   # 超参数配置
│
├── 🎬 scripts/                       # 可执行脚本
│   ├── train.py                    # 训练入口
│   ├── collect_data.py             # 数据收集入口
│   └── evaluate.py                 # 评估入口
│
├── 💾 models/                        # 模型存储
│   └── best_model.pth              # 训练后生成
│
├── 📊 runs/                          # TensorBoard 日志
├── 📄 mouse_trajectories.csv        # 训练数据（1.1M）
├── 📖 requirements.txt              # 依赖列表
└── 📦 setup.py                      # 安装配置
```

---

## ✨ 功能特点

### 🎯 核心功能

#### 1. **模块化架构**
- **模型模块** (`src/models`): LSTM-CVAE 主模型、长度预测器、损失函数
- **数据模块** (`src/data`): 数据集、数据加载器、归一化处理
- **训练模块** (`src/training`): 训练器、TensorBoard 集成、模型管理
- **工具模块** (`src/utils`): 轨迹提取器、比较器、统计分析
- **GUI 模块** (`src/gui`): 数据收集、可视化评估

#### 2. **智能模型架构**
```
输入特征 (10维)
    ↓
LSTM 编码器 → 提取时序特征
    ↓
CVAE 编码器 → 学习潜在表示
    ↓
CVAE 解码器 → 生成轨迹点
    ↓
输出轨迹
```

- **LSTM**: 处理序列特征（位置、速度、加速度、方向等）
- **CVAE**: 学习轨迹的潜在表示，支持多样化生成
- **长度预测器**: 根据起点终点自动预测轨迹长度

#### 3. **丰富的特征工程**

模型输入特征（10 维）：
| 特征 | 说明 | 维度 |
|-----|------|------|
| `start_x, start_y` | 上一个位置坐标 | 2 |
| `end_x, end_y` | 目标终点坐标 | 2 |
| `current_x, current_y` | 当前位置坐标 | 2 |
| `velocity` | 速度（像素/秒） | 1 |
| `acceleration` | 加速度（像素/秒²） | 1 |
| `direction` | 运动方向（0-360°） | 1 |
| `distance` | 移动距离 | 1 |

#### 4. **完整的工具链**
- ✅ 数据收集 GUI
- ✅ 自动数据预处理和归一化
- ✅ 训练进度可视化（TensorBoard）
- ✅ 模型评估和对比
- ✅ 轨迹分析工具

---

## 📦 安装指南

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA (可选，用于 GPU 加速)

### 方式 1: 使用 pip 安装（推荐）

```bash
# 克隆项目
git clone https://github.com/your-repo/lstm_cvae_mouse_trajectory.git
cd lstm_cvae_mouse_trajectory

# 开发模式安装
pip install -e .
```

**优点**：
- 自动安装所有依赖
- 可以直接使用命令行工具
- 支持在任何位置导入包

### 方式 2: 手动安装依赖

```bash
# 安装依赖
pip install -r requirements.txt

# 运行脚本
python scripts/train.py
```

### 验证安装

```python
# 测试导入
from src.models import LSTMCVAE
from src.data import MouseTrajectoryDataset
from config import Config

print("安装成功！")
```

---

## 📖 使用教程

### 步骤 1: 收集训练数据

```bash
# 使用命令行工具
trajectory-collect

# 或使用脚本
python scripts/collect_data.py
```

**操作说明**：
1. 程序启动后显示一个**绿色圆点**（起点）
2. 点击绿色点，会出现**红色圆点**（终点）
3. 移动鼠标到红色点（系统自动记录轨迹）
4. 点击红色点完成一组数据
5. 重复步骤 1-4 收集多组数据（建议 **100-200 组**）
6. 按 **ESC** 键退出

**数据保存**：所有数据保存在 `mouse_trajectories.csv`

### 步骤 2: 配置超参数（可选）

编辑 `config/config.py`：

```python
class Config:
    # 模型参数
    LSTM_HIDDEN_DIM = 128        # LSTM 隐藏层维度
    LATENT_DIM = 32              # CVAE 潜在空间维度

    # 训练参数
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100

    # 生成多样性控制
    KL_WEIGHT = 0.001            # 越大越随机，越小越确定
```

### 步骤 3: 训练模型

```bash
# 使用命令行工具
trajectory-train

# 或使用脚本
python scripts/train.py
```

**训练输出**：
- 实时进度条显示损失
- 自动保存最佳模型到 `models/best_model.pth`
- TensorBoard 日志保存到 `runs/`

**查看训练曲线**：
```bash
tensorboard --logdir=runs
# 打开浏览器访问 http://localhost:6006
```

### 步骤 4: 评估模型

```bash
# 使用命令行工具
trajectory-evaluate

# 或使用脚本
python scripts/evaluate.py
```

**评估界面**：
- 点击绿色起点，移动到红色终点（绘制人类轨迹）
- AI 自动生成轨迹
- **蓝色线条** = 人类轨迹
- **红色线条** = AI 生成轨迹
- 直观对比模型性能

---

## 🔧 模块参考

### src/models - 模型模块

**文件**: `lstm_cvae.py`

**导出类和函数**：
```python
from src.models import (
    LSTMCVAE,                 # 主模型
    TrajectoryLengthPredictor,  # 长度预测器
    compute_loss              # 损失函数
)
```

**核心类**：
- `LSTMCVAE`: LSTM + CVAE 组合模型
  - `forward()`: 训练模式（重建轨迹）
  - `generate()`: 生成模式（生成新轨迹）

- `TrajectoryLengthPredictor`: 预测轨迹点数量
  - `forward()`: 根据起点终点预测长度

- `CVAEEncoder`: CVAE 编码器
- `CVAEDecoder`: CVAE 解码器

### src/data - 数据模块

**文件**: `dataset.py`

**导出类和函数**：
```python
from src.data import (
    MouseTrajectoryDataset,   # 数据集类
    create_data_loaders,      # 创建加载器
    collate_fn                # 批处理函数
)
```

**核心类**：
- `MouseTrajectoryDataset`: 轨迹数据集
  - 自动归一化
  - 特征工程
  - 返回格式: `{'features', 'start_point', 'end_point', 'length'}`

### src/training - 训练模块

**文件**: `trainer.py`

**导出类**：
```python
from src.training import Trainer
```

**核心方法**：
- `train()`: 完整训练流程
- `train_epoch()`: 训练一个 epoch
- `validate()`: 验证模型
- `save_checkpoint()`: 保存模型
- `load_checkpoint()`: 加载模型

### src/utils - 工具模块

**文件**: `trajectory_utils.py`

**导出类**：
```python
from src.utils import (
    TrajectoryExtractor,      # 轨迹提取器
    TrajectoryComparator      # 轨迹比较器
)
```

**TrajectoryExtractor** 方法：
- `extract_trajectory_points()`: 提取完整轨迹信息
- `extract_coordinates_only()`: 只提取坐标
- `calculate_trajectory_metrics()`: 计算统计指标
- `save_trajectory_to_csv()`: 保存到 CSV
- `interpolate_trajectory()`: 轨迹插值

**TrajectoryComparator** 方法：
- `compute_dtw_distance()`: DTW 距离
- `compute_frechet_distance()`: Fréchet 距离

### src/gui - GUI 模块

**文件**: `collector.py`, `evaluator.py`

**导出类**：
```python
from src.gui import (
    MouseTrajectoryCollector,    # 数据收集界面
    TrajectoryEvaluationGUI      # 评估界面
)
```

---

## 💻 API 文档

### 1. 训练自定义模型

```python
from config import Config
from src.training import Trainer

# 创建配置
config = Config()
config.NUM_EPOCHS = 50
config.BATCH_SIZE = 64

# 训练
trainer = Trainer(config)
trainer.train()
```

### 2. 加载和使用模型

```python
import torch
from src.models import LSTMCVAE, TrajectoryLengthPredictor
from config import Config

# 加载配置
config = Config()

# 加载模型
checkpoint = torch.load('models/best_model.pth')
model = LSTMCVAE(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

length_predictor = TrajectoryLengthPredictor()
length_predictor.load_state_dict(checkpoint['length_predictor_state_dict'])
length_predictor.eval()

# 生成轨迹
with torch.no_grad():
    start = torch.tensor([[100.0, 100.0]])
    end = torch.tensor([[500.0, 500.0]])

    # 预测长度
    length = int(length_predictor(start, end).item())

    # 生成轨迹
    trajectory = model.generate(start, end, length)
    print(f"生成了 {length} 个轨迹点")
```

### 3. 使用轨迹提取器

```python
from src.utils import TrajectoryExtractor

# 创建提取器（需要归一化统计信息）
extractor = TrajectoryExtractor(norm_stats)

# 提取完整轨迹信息
points = extractor.extract_trajectory_points(model_output)

# 遍历轨迹点
for point in points:
    print(f"位置: ({point['current_x']}, {point['current_y']})")
    print(f"速度: {point['velocity']:.2f} px/s")
    print(f"加速度: {point['acceleration']:.2f} px/s²")
    print(f"方向: {point['direction']:.1f}°")

# 只提取坐标（用于绘图）
coords = extractor.extract_coordinates_only(model_output)

# 计算轨迹统计指标
metrics = extractor.calculate_trajectory_metrics(points)
print(f"总距离: {metrics['total_distance']:.2f}")
print(f"平均速度: {metrics['avg_velocity']:.2f}")
print(f"路径效率: {metrics['path_efficiency']:.2%}")

# 保存到 CSV
extractor.save_trajectory_to_csv(points, 'output_trajectory.csv')
```

### 4. 自定义数据加载

```python
from src.data import MouseTrajectoryDataset, create_data_loaders

# 方式 1: 使用便捷函数
train_loader, val_loader, stats = create_data_loaders(
    'mouse_trajectories.csv',
    batch_size=32,
    train_split=0.8
)

# 方式 2: 手动创建
dataset = MouseTrajectoryDataset('mouse_trajectories.csv', normalize=True)
print(f"数据集大小: {len(dataset)}")

# 获取一个样本
sample = dataset[0]
print(f"特征形状: {sample['features'].shape}")
print(f"起点: {sample['start_point']}")
print(f"终点: {sample['end_point']}")
print(f"长度: {sample['length']}")
```

---

## ⚙️ 配置说明

### config/config.py

```python
class Config:
    # ===== 数据配置 =====
    DATA_FILE = 'mouse_trajectories.csv'
    NORMALIZE_COORDS = True

    # ===== 模型配置 =====
    # LSTM 参数
    LSTM_HIDDEN_DIM = 128         # 隐藏层维度
    LSTM_NUM_LAYERS = 2           # LSTM 层数
    LSTM_DROPOUT = 0.2            # Dropout 率

    # CVAE 参数
    LATENT_DIM = 32               # 潜在空间维度（影响生成能力）
    ENCODER_HIDDEN_DIM = 128      # 编码器隐藏层
    DECODER_HIDDEN_DIM = 128      # 解码器隐藏层
    KL_WEIGHT = 0.001             # KL 散度权重（控制多样性）

    # 长度预测器
    LENGTH_PREDICTOR_HIDDEN_DIM = 64
    MAX_TRAJECTORY_LENGTH = 500

    # ===== 训练配置 =====
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    TRAIN_SPLIT = 0.8

    # ===== 设备配置 =====
    DEVICE = 'cuda'  # 'cuda' or 'cpu'

    # ===== 路径配置 =====
    MODEL_SAVE_PATH = 'models/'
    BEST_MODEL_PATH = 'models/best_model.pth'
```

### 关键参数调优指南

| 参数 | 作用 | 调优建议 |
|-----|------|---------|
| `LSTM_HIDDEN_DIM` | 模型容量 | 过拟合 → 减小；欠拟合 → 增大 |
| `LATENT_DIM` | 潜在表示维度 | 增大 → 更强表达能力 |
| `KL_WEIGHT` | 生成多样性 | 增大 → 更随机；减小 → 更确定 |
| `LEARNING_RATE` | 学习速率 | 损失不降 → 减小；收敛慢 → 增大 |
| `BATCH_SIZE` | 批大小 | GPU 内存允许尽量大 |

---

## 🛠️ 开发指南

### 添加新特征

1. **更新配置**（`config/config.py`）：
```python
INPUT_DIM = 12  # 从 10 增加到 12
```

2. **修改数据集**（`src/data/dataset.py`）：
```python
def _process_trajectory(self, group_data):
    # 添加新特征计算
    curvature = self.calculate_curvature(...)
    jerk = self.calculate_jerk(...)

    feature_vector = [
        start_x, start_y, end_x, end_y,
        current_x, current_y,
        velocity, acceleration, direction, distance,
        curvature, jerk  # 新特征
    ]
```

3. **测试**：
```bash
python scripts/train.py
```

### 扩展模型

```python
from src.models import LSTMCVAE
import torch.nn as nn

class CustomLSTMCVAE(LSTMCVAE):
    def __init__(self, config):
        super().__init__(config)
        # 添加注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=config.LSTM_HIDDEN_DIM,
            num_heads=4
        )

    def forward(self, x, start_point, end_point):
        # LSTM 编码
        lstm_out, _ = self.lstm(x)

        # 添加注意力
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # CVAE 处理
        mu, logvar = self.encoder(attn_out)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z, attn_out)

        return reconstructed, mu, logvar
```

### 添加新工具

在 `src/utils/` 创建新文件：

```python
# src/utils/visualization.py
import matplotlib.pyplot as plt

class TrajectoryVisualizer:
    def plot_trajectory(self, coords):
        plt.figure(figsize=(10, 8))
        plt.plot(coords[:, 0], coords[:, 1], 'b-')
        plt.scatter(coords[0, 0], coords[0, 1], c='g', s=100)
        plt.scatter(coords[-1, 0], coords[-1, 1], c='r', s=100)
        plt.show()
```

更新 `src/utils/__init__.py`：
```python
from .visualization import TrajectoryVisualizer
__all__ = ['TrajectoryExtractor', 'TrajectoryComparator', 'TrajectoryVisualizer']
```

---

## ❓ 常见问题

### Q1: 训练损失不下降？

**可能原因**：
- 数据量不足（< 50 组）
- 学习率过大
- 模型过于复杂

**解决方案**：
```python
# 1. 检查数据量
import pandas as pd
df = pd.read_csv('mouse_trajectories.csv')
print(f"轨迹组数: {df['group_id'].nunique()}")

# 2. 降低学习率
config.LEARNING_RATE = 0.0001

# 3. 简化模型
config.LSTM_HIDDEN_DIM = 64
config.LSTM_NUM_LAYERS = 1
```

### Q2: 生成的轨迹不自然？

**原因**: `KL_WEIGHT` 参数不合适

**解决方案**：
```python
# 轨迹太随机、波动大
config.KL_WEIGHT = 0.0001  # 减小权重

# 轨迹太死板、缺乏变化
config.KL_WEIGHT = 0.01    # 增大权重
```

### Q3: 导入错误 `ModuleNotFoundError`？

**解决方案**：
```bash
# 确保使用开发模式安装
pip install -e .

# 或在脚本开头添加路径
import sys
sys.path.insert(0, '/path/to/project')
```

### Q4: CUDA out of memory？

**解决方案**：
```python
# 减小批大小
config.BATCH_SIZE = 16  # 或 8

# 或使用 CPU
config.DEVICE = 'cpu'
```

### Q5: 如何继续训练？

```python
from src.training import Trainer
from config import Config

config = Config()
trainer = Trainer(config)

# 加载检查点
start_epoch = trainer.load_checkpoint('models/best_model.pth')

# 继续训练
trainer.train()
```

---

## 📊 性能优化建议

### 数据层面
- ✅ 收集至少 **100-200 组**轨迹数据
- ✅ 确保数据多样性（不同起点、终点组合）
- ✅ 数据质量检查（无异常值）

### 训练层面
- ⚡ 使用 **GPU** 加速（10-30x 速度提升）
- ⚡ 调整 `BATCH_SIZE`（GPU 内存允许尽量大）
- ⚡ 使用学习率调度器（自动集成）

### 模型层面
- 🎯 起始参数：`LSTM_HIDDEN_DIM=128`, `LATENT_DIM=32`
- 🎯 过拟合：增加 `DROPOUT` 或减小模型
- 🎯 欠拟合：增大模型容量

---

## 📚 参考文献

本项目基于以下研究：
- **LSTM**: Hochreiter & Schmidhuber (1997) - Long Short-Term Memory
- **CVAE**: Sohn et al. (2015) - Learning Structured Output Representation using Deep Conditional Generative Models
- **Human-Computer Interaction**: 人机交互行为建模

---

## 📄 License

MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 👨‍💻 作者

AI Algorithm Engineer @ Google (示例项目)

---

## 🙏 致谢

感谢所有贡献者和使用者的支持！

**如有问题或建议，欢迎提 Issue！**

---

**⭐ 如果这个项目对你有帮助，请给个 Star 支持一下！**
# 项目结构说明

## 目录结构

```
lstm_cvae_mouse_trajectory/
│
├── src/                                # 源代码包
│   ├── __init__.py                    # 包初始化
│   │
│   ├── models/                         # 模型定义模块
│   │   ├── __init__.py                # 导出: LSTMCVAE, TrajectoryLengthPredictor, compute_loss
│   │   └── lstm_cvae.py               # LSTM-CVAE模型和长度预测器
│   │
│   ├── data/                           # 数据处理模块
│   │   ├── __init__.py                # 导出: MouseTrajectoryDataset, create_data_loaders, collate_fn
│   │   └── dataset.py                 # 数据集类和加载器
│   │
│   ├── training/                       # 训练模块
│   │   ├── __init__.py                # 导出: Trainer
│   │   └── trainer.py                 # 训练器类
│   │
│   ├── utils/                          # 工具模块
│   │   ├── __init__.py                # 导出: TrajectoryExtractor, TrajectoryComparator
│   │   └── trajectory_utils.py        # 轨迹提取和分析工具
│   │
│   └── gui/                            # GUI应用模块
│       ├── __init__.py                # 导出: MouseTrajectoryCollector, TrajectoryEvaluationGUI
│       ├── collector.py               # 数据收集界面
│       └── evaluator.py               # 评估界面
│
├── config/                             # 配置模块
│   ├── __init__.py                    # 导出: Config
│   └── config.py                      # 超参数配置类
│
├── scripts/                            # 可执行脚本
│   ├── train.py                       # 训练脚本
│   ├── collect_data.py                # 数据收集脚本
│   └── evaluate.py                    # 评估脚本
│
├── models/                             # 模型文件存储
│   ├── .gitkeep
│   ├── best_model.pth                 # 最佳模型（训练后生成）
│   └── checkpoint_epoch_*.pth         # 检查点文件（训练后生成）
│
├── runs/                               # TensorBoard日志（训练后生成）
│   └── lstm_cvae_trajectory/
│
├── mouse_trajectories.csv              # 轨迹数据文件
├── requirements.txt                    # Python依赖列表
├── setup.py                           # 安装配置
├── README.md                          # 项目文档
└── STRUCTURE.md                       # 本文件 - 结构说明
```

## 模块说明

### 1. src/models/ - 模型定义
**文件**: `lstm_cvae.py`

**类和函数**:
- `TrajectoryLengthPredictor`: 预测轨迹点数量的辅助网络
- `CVAEEncoder`: CVAE编码器
- `CVAEDecoder`: CVAE解码器
- `LSTMCVAE`: 主模型（LSTM + CVAE组合）
- `compute_loss()`: 损失函数计算

**导入示例**:
```python
from src.models import LSTMCVAE, TrajectoryLengthPredictor, compute_loss
```

### 2. src/data/ - 数据处理
**文件**: `dataset.py`

**类和函数**:
- `MouseTrajectoryDataset`: 轨迹数据集类
- `collate_fn()`: 批处理函数（处理变长序列）
- `create_data_loaders()`: 创建训练/验证数据加载器

**导入示例**:
```python
from src.data import MouseTrajectoryDataset, create_data_loaders
```

### 3. src/training/ - 训练模块
**文件**: `trainer.py`

**类和方法**:
- `Trainer`: 训练器类
  - `train_epoch()`: 训练一个epoch
  - `validate()`: 验证
  - `train()`: 完整训练流程
  - `save_checkpoint()`: 保存模型
  - `load_checkpoint()`: 加载模型

**导入示例**:
```python
from src.training import Trainer
```

### 4. src/utils/ - 工具模块
**文件**: `trajectory_utils.py`

**类**:
- `TrajectoryExtractor`: 轨迹提取器
  - `extract_trajectory_points()`: 提取完整轨迹点信息
  - `extract_coordinates_only()`: 只提取坐标
  - `calculate_trajectory_metrics()`: 计算统计指标
  - `save_trajectory_to_csv()`: 保存到CSV
  - `interpolate_trajectory()`: 轨迹插值

- `TrajectoryComparator`: 轨迹比较器
  - `compute_dtw_distance()`: DTW距离
  - `compute_frechet_distance()`: Fréchet距离

**导入示例**:
```python
from src.utils import TrajectoryExtractor, TrajectoryComparator
```

### 5. src/gui/ - GUI应用
**文件**:
- `collector.py`: 数据收集界面
- `evaluator.py`: 评估界面

**类**:
- `MouseTrajectoryCollector`: 数据收集GUI
- `TrajectoryEvaluationGUI`: 评估GUI（人类 vs AI对比）

**导入示例**:
```python
from src.gui import MouseTrajectoryCollector, TrajectoryEvaluationGUI
```

### 6. config/ - 配置模块
**文件**: `config.py`

**类**:
- `Config`: 配置类（包含所有超参数）

**导入示例**:
```python
from config import Config
```

## 可执行脚本

### scripts/train.py
训练模型的入口脚本

```bash
python scripts/train.py
```

### scripts/collect_data.py
数据收集的入口脚本

```bash
python scripts/collect_data.py
```

### scripts/evaluate.py
模型评估的入口脚本

```bash
python scripts/evaluate.py
```

## 导入规范

### 从外部导入包
```python
# 导入配置
from config.config import Config

# 导入模型
from src.models.lstm_cvae import LSTMCVAE, TrajectoryLengthPredictor

# 导入数据
from src.data.dataset import MouseTrajectoryDataset, create_data_loaders

# 导入训练器
from src.training.trainer import Trainer

# 导入工具
from src.utils.trajectory_utils import TrajectoryExtractor
```

### 包内相对导入
在 `src/` 包内部，使用相对导入：

```python
# 在 src/training/trainer.py 中
from ..models.lstm_cvae import LSTMCVAE
from ..data.dataset import create_data_loaders
```

## 数据流

```
1. 数据收集
   GUI (collector.py) → mouse_trajectories.csv

2. 数据加载
   CSV → MouseTrajectoryDataset → DataLoader → Trainer

3. 模型训练
   Trainer → LSTMCVAE → models/best_model.pth

4. 模型评估
   best_model.pth → TrajectoryEvaluationGUI → 可视化对比
```

## 配置流

```
config/config.py
    ↓
所有模块读取配置
    ├── src/models/lstm_cvae.py (网络结构参数)
    ├── src/data/dataset.py (数据处理参数)
    ├── src/training/trainer.py (训练参数)
    └── scripts/*.py (运行参数)
```

## 安装方式

### 开发安装（推荐）
```bash
pip install -e .
```
安装后可以：
- 使用命令行工具: `trajectory-train`, `trajectory-collect`, `trajectory-evaluate`
- 在任何地方导入包: `from src.models import LSTMCVAE`

### 手动运行
不安装包，直接运行脚本：
```bash
python scripts/train.py
```
脚本会自动添加项目路径到 `sys.path`

## 文件职责

| 文件 | 职责 | 依赖 |
|-----|-----|-----|
| src/models/lstm_cvae.py | 定义神经网络模型 | config.config |
| src/data/dataset.py | 数据加载和预处理 | config.config |
| src/training/trainer.py | 训练循环和模型管理 | models, data, config |
| src/utils/trajectory_utils.py | 轨迹分析工具 | - |
| src/gui/collector.py | 数据收集界面 | config |
| src/gui/evaluator.py | 评估界面 | models, utils, config |
| config/config.py | 配置管理 | - |
| scripts/*.py | 入口脚本 | 所有模块 |

## 扩展指南

### 添加新模型
1. 在 `src/models/` 创建新文件
2. 在 `src/models/__init__.py` 导出新模型
3. 在配置中添加模型参数

### 添加新数据源
1. 在 `src/data/` 创建新的Dataset类
2. 在 `src/data/__init__.py` 导出
3. 修改 `create_data_loaders()` 支持新数据源

### 添加新工具
1. 在 `src/utils/` 添加工具类/函数
2. 在 `src/utils/__init__.py` 导出
3. 在 README 中添加使用示例

## 版本控制建议

**.gitignore** 应包含：
```
# 模型文件
models/*.pth
!models/.gitkeep

# TensorBoard日志
runs/

# Python缓存
__pycache__/
*.pyc
*.pyo

# IDE
.idea/
.vscode/

# 数据（可选）
mouse_trajectories.csv
```

---

**维护者**: AI Algorithm Engineer
**最后更新**: 2025-12-12
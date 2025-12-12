# 鼠标轨迹预测系统 - LSTM + CVAE

这是一个基于深度学习的鼠标轨迹预测系统，使用LSTM和CVAE（条件变分自编码器）结合的架构来预测和生成鼠标移动轨迹。

## 项目结构

```
lstm_cvae_mouse_trajectory/
├── src/                            # 源代码包
│   ├── models/                     # 模型定义
│   │   ├── __init__.py
│   │   └── lstm_cvae.py           # LSTM-CVAE模型和长度预测器
│   ├── data/                       # 数据处理
│   │   ├── __init__.py
│   │   └── dataset.py             # 数据集类和加载器
│   ├── training/                   # 训练模块
│   │   ├── __init__.py
│   │   └── trainer.py             # 训练器类
│   ├── utils/                      # 工具函数
│   │   ├── __init__.py
│   │   └── trajectory_utils.py    # 轨迹提取和分析工具
│   └── gui/                        # GUI应用
│       ├── __init__.py
│       ├── collector.py           # 数据收集界面
│       └── evaluator.py           # 评估界面
├── config/                         # 配置文件
│   ├── __init__.py
│   └── config.py                  # 超参数配置
├── scripts/                        # 可执行脚本
│   ├── train.py                   # 训练脚本
│   ├── collect_data.py            # 数据收集脚本
│   └── evaluate.py                # 评估脚本
├── models/                         # 保存的模型文件
├── runs/                           # TensorBoard日志
├── mouse_trajectories.csv          # 轨迹数据
├── requirements.txt               # Python依赖
├── setup.py                       # 安装脚本
└── README.md                      # 项目文档
```

## 功能特点

### 1. 模块化设计
- **src/models**: 模型定义模块
  - LSTM-CVAE主模型
  - 轨迹长度预测器
  - 损失函数

- **src/data**: 数据处理模块
  - 数据集类
  - 数据加载器
  - 归一化处理

- **src/training**: 训练模块
  - 训练器类
  - TensorBoard集成
  - 模型保存/加载

- **src/utils**: 工具模块
  - 轨迹提取器
  - 轨迹比较器
  - 统计分析工具

- **src/gui**: GUI应用
  - 数据收集界面
  - 可视化评估界面

### 2. 模型架构
- **LSTM**: 处理序列特征（上一个点、目标点、当前点、速度、加速度、距离等）
- **CVAE编码器**: 学习轨迹的潜在表示
- **CVAE解码器**: 从潜在变量生成轨迹点
- **长度预测器**: 根据起点和终点预测轨迹需要多少个点

### 3. 特征工程
模型输入特征（10维）：
1. `start_x, start_y`: 上一个位置
2. `end_x, end_y`: 目标终点
3. `current_x, current_y`: 当前位置
4. `velocity`: 速度（像素/秒）
5. `acceleration`: 加速度（像素/秒²）
6. `direction`: 运动方向（0-360度）
7. `distance`: 相对上一次移动的距离

## 安装

### 方式1: 使用pip安装（推荐）

```bash
# 克隆项目
git clone <repository-url>
cd lstm_cvae_mouse_trajectory

# 安装包和依赖
pip install -e .
```

安装后可以直接使用命令行工具：
```bash
trajectory-train      # 训练模型
trajectory-collect    # 收集数据
trajectory-evaluate   # 评估模型
```

### 方式2: 手动安装依赖

```bash
pip install -r requirements.txt
```

## 使用步骤

### 步骤1: 收集数据

```bash
# 使用安装的命令
trajectory-collect

# 或直接运行脚本
python scripts/collect_data.py
```

操作步骤：
- 点击绿色点开始
- 移动鼠标到红色点并点击完成一组数据
- 重复多次收集足够的数据（建议至少100组）
- 按ESC退出

数据保存在 `mouse_trajectories.csv`

### 步骤2: 配置参数

编辑 `config/config.py` 根据需要调整超参数：

```python
# 关键参数
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
LSTM_HIDDEN_DIM = 128
LATENT_DIM = 32
KL_WEIGHT = 0.001
```

### 步骤3: 训练模型

```bash
# 使用安装的命令
trajectory-train

# 或直接运行脚本
python scripts/train.py
```

训练过程中：
- 进度条显示实时损失
- 模型自动保存到 `models/best_model.pth`
- TensorBoard日志保存到 `runs/`

查看训练曲线：
```bash
tensorboard --logdir=runs
```

### 步骤4: 评估模型

```bash
# 使用安装的命令
trajectory-evaluate

# 或直接运行脚本
python scripts/evaluate.py
```

在GUI中：
- 点击绿色起点
- 移动鼠标到红色终点（绘制人类轨迹）
- 点击红色终点，AI自动生成轨迹
- 对比蓝色（人类）和红色（AI）轨迹

## API使用示例

### 训练自定义模型

```python
from config.config import Config
from src.training.trainer import Trainer

# 创建配置
config = Config()
config.NUM_EPOCHS = 50
config.BATCH_SIZE = 64

# 训练
trainer = Trainer(config)
trainer.train()
```

### 使用轨迹工具

```python
from src.utils.trajectory_utils import TrajectoryExtractor
import torch

# 创建提取器（需要归一化统计信息）
extractor = TrajectoryExtractor(norm_stats)

# 从模型输出提取轨迹信息
trajectory_points = extractor.extract_trajectory_points(model_output)

# 每个点包含完整信息
for point in trajectory_points:
    print(f"位置: ({point['current_x']}, {point['current_y']})")
    print(f"速度: {point['velocity']}, 加速度: {point['acceleration']}")
    print(f"方向: {point['direction']}°")

# 只提取坐标（用于绘图）
coords = extractor.extract_coordinates_only(model_output)

# 计算轨迹统计指标
metrics = extractor.calculate_trajectory_metrics(trajectory_points)
print(f"总距离: {metrics['total_distance']}")
print(f"平均速度: {metrics['avg_velocity']}")
print(f"路径效率: {metrics['path_efficiency']}")

# 保存到CSV
extractor.save_trajectory_to_csv(trajectory_points, 'output_trajectory.csv')
```

### 加载和使用模型

```python
import torch
from config.config import Config
from src.models.lstm_cvae import LSTMCVAE, TrajectoryLengthPredictor

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
    # 预测长度
    start = torch.tensor([[100.0, 100.0]])
    end = torch.tensor([[500.0, 500.0]])

    length = int(length_predictor(start, end).item())

    # 生成轨迹
    trajectory = model.generate(start, end, length)
    print(f"生成了 {length} 个轨迹点")
```

## 高级配置

### 修改网络结构

在 `config/config.py` 中：

```python
# LSTM参数
LSTM_HIDDEN_DIM = 128      # LSTM隐藏层维度
LSTM_NUM_LAYERS = 2        # LSTM层数
LSTM_DROPOUT = 0.2         # Dropout率

# CVAE参数
LATENT_DIM = 32            # 潜在空间维度
ENCODER_HIDDEN_DIM = 128   # 编码器隐藏层维度
DECODER_HIDDEN_DIM = 128   # 解码器隐藏层维度

# KL散度权重
KL_WEIGHT = 0.001          # 控制生成多样性
```

### 调整训练参数

```python
# 训练参数
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
TRAIN_SPLIT = 0.8

# 设备选择
DEVICE = 'cuda'  # 'cuda' or 'cpu'
```

## 性能优化建议

1. **数据量**: 收集至少100-200组轨迹数据
2. **训练时间**:
   - CPU: 10-30分钟
   - GPU: 2-5分钟
3. **超参数调整**:
   - 轨迹太随机 → 降低 `KL_WEIGHT`
   - 缺乏多样性 → 增加 `KL_WEIGHT`
   - 过拟合 → 增加 `LSTM_DROPOUT` 或减少 `LSTM_HIDDEN_DIM`

## 开发指南

### 添加新特征

1. 在 `config/config.py` 中更新 `INPUT_DIM`
2. 在 `src/data/dataset.py` 的 `_process_trajectory` 中添加特征提取
3. 更新模型输入维度

### 扩展模型

继承基础模型类：

```python
from src.models.lstm_cvae import LSTMCVAE

class CustomLSTMCVAE(LSTMCVAE):
    def __init__(self, config):
        super().__init__(config)
        # 添加自定义层
        self.custom_layer = nn.Linear(...)

    def forward(self, x, start_point, end_point):
        # 自定义前向传播
        ...
```

## 常见问题

### Q: 训练损失不下降？
A:
- 检查数据是否足够（至少50组以上）
- 降低学习率 `LEARNING_RATE`
- 增加训练轮数 `NUM_EPOCHS`

### Q: 生成的轨迹不自然？
A:
- 调整 `KL_WEIGHT` 参数
- 增加 `LATENT_DIM` 提高表达能力
- 收集更多样化的数据

### Q: 导入错误？
A:
- 确保使用 `pip install -e .` 安装包
- 或在脚本中添加项目路径到 `sys.path`

## 论文参考

本项目实现基于以下研究领域：
- LSTM: Long Short-Term Memory Networks
- CVAE: Conditional Variational Autoencoder
- 时间序列预测
- 人机交互行为建模

## License

MIT License

## 作者

_AI算法工程师_ @ Google (示例项目)

---

**祝训练顺利！如有问题欢迎提Issue。**
# 鼠标轨迹评估代码使用说明

## 概述

`evaluate.py` 提供了一个清晰、模块化的轨迹生成和评估系统，用于基于训练好的 LSTM-CVAE 模型生成鼠标移动轨迹并进行可视化分析。

## 代码架构

代码采用清晰的**六层架构设计**，每层职责明确：

```
┌─────────────────────────────────────────────────────┐
│  第 6 层：主函数 (main)                              │
│  - 提供使用示例                                      │
└─────────────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│  第 5 层：综合评估器 (TrajectoryEvaluator)          │
│  - 统一的对外 API                                   │
│  - 整合各层功能                                      │
└─────────────────────────────────────────────────────┘
                        ▼
┌──────────────┬──────────────┬──────────────────────┐
│ 第 1 层：     │ 第 2 层：     │ 第 3 层：             │
│ 数据预处理    │ 模型推理     │ 轨迹分析             │
│──────────────│──────────────│──────────────────────│
│CoordinateNorm│ModelInference│TrajectoryAnalyzer    │
│alizer        │              │                      │
│- 归一化      │- 加载模型     │- 计算评估指标         │
│- 反归一化    │- 生成轨迹     │- 质量分析            │
└──────────────┴──────────────┴──────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│  第 4 层：可视化 (TrajectoryVisualizer)             │
│  - 绘制轨迹图                                       │
│  - 保存图片                                         │
└─────────────────────────────────────────────────────┘
```

### 各层详细说明

#### 第 1 层：CoordinateNormalizer (数据预处理层)
- **功能**: 处理原始坐标与归一化坐标的转换
- **方法**:
  - `normalize()`: 将像素坐标转换为 [0, 1] 范围
  - `denormalize()`: 将归一化坐标还原为像素坐标
  - `denormalize_trajectory()`: 批量转换轨迹坐标

#### 第 2 层：ModelInference (模型推理层)
- **功能**: 加载模型并生成轨迹
- **方法**:
  - `_load_model()`: 加载训练好的模型
  - `generate_trajectories()`: 根据起终点生成轨迹

#### 第 3 层：TrajectoryAnalyzer (轨迹分析层)
- **功能**: 计算轨迹质量指标
- **方法**:
  - `calculate_metrics()`: 计算轨迹评估指标
  - `print_metrics()`: 打印指标信息
- **评估指标**:
  - 轨迹总长度
  - 直线距离
  - 路径效率（直线距离 / 轨迹长度）
  - 终点误差
  - 平均速度
  - 平均加速度

#### 第 4 层：TrajectoryVisualizer (可视化层)
- **功能**: 绘制和保存轨迹图
- **方法**:
  - `plot()`: 绘制轨迹，标注起终点，保存图片

#### 第 5 层：TrajectoryEvaluator (综合评估器)
- **功能**: 提供统一的对外 API，整合上述所有功能
- **核心方法**:
  - `generate()`: 生成轨迹
  - `evaluate()`: 评估轨迹质量
  - `visualize()`: 可视化轨迹

## 快速开始

### 基本使用

```python
from evaluate import TrajectoryEvaluator

# 1. 创建评估器
evaluator = TrajectoryEvaluator(
    model_path='models/best_model.pth',
    screen_width=1920,
    screen_height=1080,
    use_normalized=False  # 使用原始像素坐标
)

# 2. 生成轨迹
trajectories = evaluator.generate(
    start=(200, 200),      # 起点坐标
    end=(1600, 800),       # 终点坐标
    num_samples=5,         # 生成 5 条轨迹
    return_raw=True        # 返回原始坐标
)

# 3. 可视化
evaluator.visualize(
    trajectories=trajectories,
    start=(200, 200),
    end=(1600, 800),
    save_path='output.png'
)
```

### 评估轨迹质量

```python
# 评估单条轨迹
metrics = evaluator.evaluate(
    trajectory=trajectories[0],
    start=(200, 200),
    end=(1600, 800)
)

# 打印评估指标
evaluator.analyzer.print_metrics(metrics, trajectory_id=1)
```

## 主要功能

### 1. 坐标模式选择

支持两种坐标模式：

#### 原始像素坐标模式（默认）
```python
evaluator = TrajectoryEvaluator(
    use_normalized=False,
    screen_width=1920,
    screen_height=1080
)

trajectories = evaluator.generate(
    start=(100, 100),  # 像素坐标
    end=(800, 600),
    return_raw=True
)
```

#### 归一化坐标模式 [0, 1]
```python
evaluator = TrajectoryEvaluator(
    use_normalized=True
)

trajectories = evaluator.generate(
    start=(0.1, 0.1),  # 归一化坐标
    end=(0.9, 0.9),
    return_raw=False
)
```

### 2. 自定义序列长度

```python
trajectories = evaluator.generate(
    start=(100, 100),
    end=(800, 600),
    seq_len=50,  # 生成 50 个点的轨迹
    num_samples=3
)
```

### 3. 批量生成

```python
# 定义多组起终点
test_cases = [
    ((100, 100), (900, 900)),
    ((100, 500), (900, 500)),
    ((500, 100), (500, 900)),
]

for start, end in test_cases:
    trajectories = evaluator.generate(
        start=start,
        end=end,
        num_samples=5
    )
    evaluator.visualize(
        trajectories=trajectories,
        start=start,
        end=end,
        save_path=f'traj_{start}_{end}.png'
    )
```

### 4. 可视化自定义

```python
evaluator.visualize(
    trajectories=trajectories,
    start=(100, 100),
    end=(800, 600),
    save_path='custom.png',
    title='自定义标题',
    figsize=(12, 10),
    dpi=200,
    show_plot=False  # 不显示，只保存
)
```

## 运行示例

### 运行主评估程序
```bash
python scripts/evaluate.py
```

### 运行所有示例
```bash
python scripts/evaluate_example.py
```

示例包括：
1. **基本使用** - 生成并可视化轨迹
2. **批量生成** - 多组起终点轨迹
3. **质量评估** - 生成并评估轨迹质量
4. **归一化坐标** - 使用 [0, 1] 坐标
5. **自定义长度** - 不同序列长度的轨迹

## API 参考

### TrajectoryEvaluator 类

#### 构造函数
```python
TrajectoryEvaluator(
    model_path: str = 'models/best_model.pth',
    screen_width: float = 1920,
    screen_height: float = 1080,
    use_normalized: bool = False
)
```

#### generate() 方法
```python
generate(
    start: Tuple[float, float],      # 起点坐标
    end: Tuple[float, float],        # 终点坐标
    seq_len: Optional[int] = None,   # 序列长度（默认使用配置）
    num_samples: int = 1,            # 生成数量
    return_raw: bool = False         # 返回原始坐标
) -> List[np.ndarray]
```

#### evaluate() 方法
```python
evaluate(
    trajectory: np.ndarray,          # 轨迹坐标 (N, 2)
    start: Tuple[float, float],      # 起点
    end: Tuple[float, float]         # 终点
) -> TrajectoryMetrics
```

#### visualize() 方法
```python
visualize(
    trajectories: List[np.ndarray],  # 轨迹列表
    start: Tuple[float, float],      # 起点
    end: Tuple[float, float],        # 终点
    save_path: str = 'trajectory.png',
    title: str = '生成的鼠标轨迹',
    show_plot: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 150
)
```

## 评估指标说明

### TrajectoryMetrics 数据类

| 字段 | 类型 | 说明 |
|------|------|------|
| `total_length` | float | 轨迹总长度（各点间距离之和） |
| `direct_distance` | float | 起终点直线距离 |
| `path_efficiency` | float | 路径效率 = 直线距离 / 轨迹长度 |
| `endpoint_error` | float | 终点误差（生成终点与目标终点的距离） |
| `avg_velocity` | float | 平均速度 |
| `avg_acceleration` | float | 平均加速度 |
| `num_points` | int | 轨迹点数 |

### 指标解读

- **路径效率**: 越接近 1 表示轨迹越接近直线
- **终点误差**: 越小表示生成的轨迹越准确到达目标终点
- **平均速度**: 反映鼠标移动的整体速度
- **平均加速度**: 反映鼠标移动的平滑程度

## 注意事项

1. **模型文件**: 运行前确保已训练模型并保存在 `models/best_model.pth`
2. **坐标系统**: 注意区分归一化坐标和原始像素坐标
3. **序列长度**: 序列长度会影响轨迹的平滑程度，建议使用训练时的长度
4. **随机性**: 每次生成的轨迹略有不同（VAE 的随机采样特性）

## 故障排除

### 模型文件不存在
```
FileNotFoundError: 模型文件不存在: models/best_model.pth
```
**解决**: 先运行训练脚本 `python scripts/train.py`

### CUDA 内存不足
**解决**: 在 `config.py` 中设置 `DEVICE = 'cpu'`

### 中文字体显示问题
**解决**: 修改 `evaluate.py` 第 18 行的字体列表，添加你系统中的中文字体

## 扩展建议

1. **添加更多评估指标**: 如曲率、抖动程度等
2. **支持批量导出**: 将轨迹保存为 CSV 或 JSON 格式
3. **实时可视化**: 使用动画展示轨迹生成过程
4. **对比分析**: 与真实轨迹进行对比评估

## 许可与贡献

欢迎提出改进建议和 Bug 报告！
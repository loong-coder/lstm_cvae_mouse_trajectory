# 人类化鼠标移动系统

这个系统使用CVAE（条件变分自编码器）模型来预测包含完整速度和时间信息的鼠标轨迹，并通过pyautogui来精确模拟人类的鼠标移动模式。

## 主要特性

- **完整的速度信息**: 每个轨迹点都包含预测的移动速度（像素/秒）
- **精确的时间控制**: 每个点之间的时间间隔由模型预测，模拟真实的人类移动节奏
- **方向信息**: 保留移动方向，确保轨迹的连贯性
- **可调速度**: 支持速度倍数调整（加速/减速）
- **PyAutoGUI集成**: 直接控制系统鼠标进行真实移动

## 文件说明

### 核心模块

1. **trajectory_point.py** - 轨迹点数据结构
   - `TrajectoryPoint`: 单个轨迹点，包含坐标、时间、速度、方向
   - `EnhancedTrajectory`: 增强轨迹类，提供轨迹分析和操作方法

2. **predict_trajectory.py** - 轨迹预测（已增强）
   - 原有的 `predict()` 方法：返回坐标列表（向后兼容）
   - 新增的 `predict_enhanced()` 方法：返回包含完整信息的 `EnhancedTrajectory`
   - 自动提取模型输出的速度和时间信息

3. **mouse_controller.py** - 鼠标控制器
   - `HumanMouseController`: 人类化鼠标控制，使用增强轨迹
   - `SimpleMouseController`: 简化控制器（向后兼容）
   - 支持移动、点击、拖动等操作

4. **demo_humanlike_mouse.py** - 完整演示
   - 5个交互式演示，展示各种使用场景
   - 包含轨迹分析功能

## 快速开始

### 安装依赖

```bash
pip install pyautogui torch numpy matplotlib
```

### 基本使用示例

```python
from predict_trajectory import TrajectoryPredictor
from mouse_controller import HumanMouseController

# 初始化预测器和控制器
predictor = TrajectoryPredictor(model_path='cvae_trajectory_predictor.pth')
controller = HumanMouseController()

# 预测轨迹（包含完整的速度和时间信息）
trajectory = predictor.predict_enhanced(
    start_x=100, start_y=100,
    end_x=700, end_y=500
)

# 查看轨迹信息
print(trajectory.summary())

# 移动鼠标
elapsed_time = controller.move_to_target(700, 500, trajectory)

print(f"移动完成！耗时: {elapsed_time:.3f}秒")
```

### 运行交互式演示

```bash
python demo_humanlike_mouse.py
```

## 详细使用指南

### 1. 轨迹预测

#### 方法1: 预测增强轨迹（推荐）

```python
predictor = TrajectoryPredictor(model_path='cvae_trajectory_predictor.pth')

# 预测包含完整速度和时间信息的轨迹
trajectory = predictor.predict_enhanced(
    start_x=100,
    start_y=100,
    end_x=700,
    end_y=500,
    smooth=True,          # 是否平滑
    downsample=True,      # 是否降采样
    align_endpoints=True  # 是否精确对齐终点
)

# 轨迹信息
print(f"轨迹点数: {len(trajectory)}")
print(f"总时长: {trajectory.total_duration:.3f}秒")
print(f"总距离: {trajectory.total_distance:.1f}像素")
print(f"平均速度: {trajectory.average_speed:.1f}px/s")

# 访问单个点
for point in trajectory:
    print(f"位置: ({point.x:.1f}, {point.y:.1f})")
    print(f"速度: {point.speed:.1f} px/s")
    print(f"时间: {point.timestamp:.3f}s")
    print(f"方向: {math.degrees(point.direction):.1f}°")
```

#### 方法2: 预测坐标列表（向后兼容）

```python
# 返回简单的坐标列表
coordinates = predictor.predict(100, 100, 700, 500)
# [(100, 100), (102.3, 101.5), ..., (700, 500)]
```

### 2. 鼠标控制

#### 基本移动

```python
controller = HumanMouseController()

# 移动到目标位置
elapsed_time = controller.move_to_target(
    target_x=700,
    target_y=500,
    trajectory=trajectory,
    speed_multiplier=1.0  # 速度倍数
)
```

#### 调整移动速度

```python
# 2倍速移动（更快）
controller.move_to_target(700, 500, trajectory, speed_multiplier=2.0)

# 半速移动（更慢，更像新手）
controller.move_to_target(700, 500, trajectory, speed_multiplier=0.5)
```

#### 移动并点击

```python
# 移动到目标并左键单击
controller.move_and_click(
    target_x=500,
    target_y=400,
    trajectory=trajectory,
    button='left',      # 'left', 'right', 'middle'
    clicks=1,           # 点击次数
    speed_multiplier=1.0,
    click_delay=0.1     # 到达后等待时间（模拟反应时间）
)

# 移动并双击
controller.move_and_click(
    target_x=500,
    target_y=400,
    trajectory=trajectory,
    clicks=2
)
```

#### 拖动操作

```python
# 按住鼠标并拖动
controller.drag_along_trajectory(
    trajectory=trajectory,
    button='left',
    speed_multiplier=1.0
)
```

### 3. 轨迹分析

```python
trajectory = predictor.predict_enhanced(100, 100, 700, 500)

# 基本信息
print(trajectory.summary())

# 速度分析
speeds = trajectory.get_speeds()
print(f"最小速度: {min(speeds):.1f} px/s")
print(f"最大速度: {max(speeds):.1f} px/s")

# 时间分析
timestamps = trajectory.get_timestamps()
durations = trajectory.get_durations()

# 访问起点和终点
start = trajectory.start_point
end = trajectory.end_point

# 计算距离
total_distance = trajectory.total_distance
```

### 4. 高级功能

#### 从当前位置自动调整轨迹

```python
# 自动从当前鼠标位置开始
current_x, current_y = controller.get_current_position()

trajectory = predictor.predict_enhanced(
    current_x, current_y,
    700, 500
)

controller.move_to_target(700, 500, trajectory)
```

#### 生成多样化轨迹（展示CVAE的随机性）

```python
# 生成多条不同的轨迹
for i in range(5):
    trajectory = predictor.predict_enhanced(100, 100, 700, 500)
    print(f"轨迹 {i+1}: {len(trajectory)}点, {trajectory.total_duration:.3f}秒")
```

## 数据结构详解

### TrajectoryPoint

```python
@dataclass
class TrajectoryPoint:
    x: float           # X坐标（像素）
    y: float           # Y坐标（像素）
    timestamp: float   # 相对时间戳（秒）
    speed: float       # 移动速度（像素/秒）
    direction: float   # 移动方向（弧度）
    duration: float    # 到达此点的时间增量（秒）
```

### EnhancedTrajectory

主要属性：
- `points`: 轨迹点列表
- `start_point`: 起点
- `end_point`: 终点
- `total_duration`: 总时长
- `total_distance`: 总距离
- `average_speed`: 平均速度

主要方法：
- `get_coordinates()`: 获取坐标列表
- `get_speeds()`: 获取速度列表
- `get_timestamps()`: 获取时间戳列表
- `summary()`: 获取摘要信息

## 速度和时间信息的来源

模型输出格式（每个点5个特征）：
1. **相对X坐标**（归一化）
2. **相对Y坐标**（归一化）
3. **时间戳**（秒）- 从模型直接输出
4. **方向**（归一化到[-1,1]）
5. **速度**（归一化到[0,1]）- 从模型直接输出

`predict_enhanced()` 方法会：
1. 从模型输出提取归一化的速度和时间
2. 反归一化到实际单位（px/s 和 秒）
3. 计算相邻点之间的时间增量（duration）
4. 构建完整的 `EnhancedTrajectory` 对象

## PyAutoGUI安全提示

1. **Failsafe功能**: 将鼠标移动到屏幕角落可以中断程序
2. **测试建议**: 先在测试环境中运行，确认行为符合预期
3. **延迟设置**: 可以通过 `pyautogui.PAUSE` 设置操作间的延迟
4. **坐标验证**: 确保坐标在屏幕范围内

```python
# 启用failsafe（默认启用）
controller = HumanMouseController(failsafe=True)

# 获取屏幕尺寸
print(f"屏幕尺寸: {controller.screen_width} x {controller.screen_height}")

# 验证坐标
if 0 <= target_x <= controller.screen_width and \
   0 <= target_y <= controller.screen_height:
    controller.move_to_target(target_x, target_y, trajectory)
```

## 性能优化建议

1. **复用预测器**: 避免重复加载模型
   ```python
   # 好的做法
   predictor = TrajectoryPredictor()
   for target in targets:
       trajectory = predictor.predict_enhanced(*target)

   # 不好的做法
   for target in targets:
       predictor = TrajectoryPredictor()  # 每次都重新加载
       trajectory = predictor.predict_enhanced(*target)
   ```

2. **调整降采样**: 减少轨迹点数量可以提高性能
   ```python
   trajectory = predictor.predict_enhanced(
       ...,
       downsample=True  # 启用降采样
   )
   ```

3. **速度倍数**: 使用更高的速度倍数可以缩短执行时间
   ```python
   controller.move_to_target(..., speed_multiplier=2.0)
   ```

## 常见问题

### Q1: 如何让移动更快/更慢？

使用 `speed_multiplier` 参数：
```python
controller.move_to_target(..., speed_multiplier=2.0)  # 2倍速
controller.move_to_target(..., speed_multiplier=0.5)  # 半速
```

### Q2: 如何确保移动看起来更自然？

1. 使用 `smooth=True` 进行平滑处理
2. 保持 `speed_multiplier` 在 0.7-1.5 之间
3. 添加随机性（生成多次，随机选择一条）

### Q3: 如何处理多显示器？

PyAutoGUI支持多显示器，坐标系统会自动扩展：
```python
# 获取所有屏幕的总尺寸
width, height = pyautogui.size()
```

### Q4: 如何避免误操作？

1. 使用 `failsafe=True`（默认启用）
2. 添加确认延迟
3. 在测试模式下先不点击，只移动

```python
# 测试模式：只移动不点击
controller.move_to_target(x, y, trajectory)

# 确认后再点击
if input("确认点击? (y/n): ") == 'y':
    controller.click_at(x, y)
```

## 示例应用场景

1. **自动化测试**: 模拟真实用户操作
2. **游戏脚本**: 生成人类化的鼠标移动避免检测
3. **UI演示**: 录制自然的交互演示
4. **辅助工具**: 帮助行动不便的用户
5. **研究目的**: 研究人类鼠标行为模式

## 许可和免责声明

此代码仅供学习和研究使用。使用此代码进行任何自动化操作时，请确保：
1. 遵守相关软件的服务条款
2. 不用于任何非法目的
3. 尊重他人的隐私和权益

## 更新日志

- **v1.0** (2024-12)
  - 添加 `TrajectoryPoint` 和 `EnhancedTrajectory` 数据结构
  - 增强 `predict_trajectory.py`，新增 `predict_enhanced()` 方法
  - 创建 `mouse_controller.py`，支持基于速度的鼠标控制
  - 添加完整的演示程序 `demo_humanlike_mouse.py`

## 联系和贡献

如有问题或建议，欢迎提出Issue或Pull Request。
# 快速开始指南

## 新增功能概述

现在模型输出不仅包含坐标点，还包含**完整的速度信息和时间信息**，方便使用pyautogui进行人类化的鼠标移动。

## 核心改进

### 之前（仅坐标）
```python
# 旧方法：只返回坐标列表
trajectory = predictor.predict(100, 100, 700, 500)
# 返回: [(100, 100), (102.3, 101.5), ..., (700, 500)]
```

### 现在（完整信息）
```python
# 新方法：返回包含速度、时间的增强轨迹
trajectory = predictor.predict_enhanced(100, 100, 700, 500)

# 每个点包含：
# - x, y 坐标
# - timestamp: 时间戳（秒）
# - speed: 移动速度（像素/秒）
# - direction: 移动方向（弧度）
# - duration: 时间增量（秒）

# 使用pyautogui控制鼠标
controller = HumanMouseController()
controller.move_to_target(700, 500, trajectory)
```

## 30秒快速体验

### 1. 测试数据结构（不控制鼠标）

```bash
python test_enhanced_trajectory.py
```

这将运行5个测试，验证：
- 轨迹数据结构
- 速度和时间信息
- 数据完整性
- 方法功能

### 2. 交互式演示（会控制鼠标）

```bash
python demo_humanlike_mouse.py
```

选择演示项目：
- **演示4**: 轨迹分析（不控制鼠标，安全）
- **演示5**: 多样化轨迹（不控制鼠标，安全）
- **演示1-3**: 实际控制鼠标移动

## 最简单的使用示例

```python
from predict_trajectory import TrajectoryPredictor
from mouse_controller import HumanMouseController

# 1. 加载模型
predictor = TrajectoryPredictor(model_path='cvae_trajectory_predictor.pth')

# 2. 预测轨迹（包含速度和时间）
trajectory = predictor.predict_enhanced(
    start_x=100, start_y=100,
    end_x=700, end_y=500
)

# 3. 查看信息
print(trajectory.summary())
# 输出：
# Trajectory Summary:
#   Points: 47
#   Start: (100.0, 100.0)
#   End: (700.0, 500.0)
#   Duration: 1.234s
#   Distance: 721.1px
#   Avg Speed: 584.2px/s
#   Speed Range: [0.0, 847.3]px/s

# 4. 控制鼠标移动（可选）
controller = HumanMouseController()
controller.move_to_target(700, 500, trajectory)
```

## 核心数据结构

### TrajectoryPoint（轨迹点）

```python
point = trajectory[5]  # 获取第5个点

print(f"坐标: ({point.x}, {point.y})")
print(f"时间: {point.timestamp}秒")
print(f"速度: {point.speed}像素/秒")
print(f"方向: {point.direction}弧度")
print(f"时间增量: {point.duration}秒")
```

### EnhancedTrajectory（增强轨迹）

```python
trajectory = predictor.predict_enhanced(100, 100, 700, 500)

# 访问属性
len(trajectory)              # 点数
trajectory.start_point       # 起点
trajectory.end_point         # 终点
trajectory.total_duration    # 总时长（秒）
trajectory.total_distance    # 总距离（像素）
trajectory.average_speed     # 平均速度（px/s）

# 获取列表
trajectory.get_coordinates() # [(x1,y1), (x2,y2), ...]
trajectory.get_speeds()      # [speed1, speed2, ...]
trajectory.get_timestamps()  # [t1, t2, ...]
trajectory.get_durations()   # [Δt1, Δt2, ...]

# 迭代
for point in trajectory:
    print(point.x, point.y, point.speed)
```

## PyAutoGUI控制示例

### 基本移动

```python
from mouse_controller import HumanMouseController

controller = HumanMouseController()

# 移动到目标（使用预测的速度）
controller.move_to_target(700, 500, trajectory)
```

### 调整速度

```python
# 2倍速（更快）
controller.move_to_target(700, 500, trajectory, speed_multiplier=2.0)

# 半速（更慢，更像新手）
controller.move_to_target(700, 500, trajectory, speed_multiplier=0.5)
```

### 移动并点击

```python
# 移动到目标并单击
controller.move_and_click(
    target_x=500,
    target_y=400,
    trajectory=trajectory,
    button='left',
    clicks=1
)

# 移动并双击
controller.move_and_click(500, 400, trajectory, clicks=2)

# 移动并右键
controller.move_and_click(500, 400, trajectory, button='right')
```

## 实际应用场景

### 1. 自动化脚本

```python
predictor = TrajectoryPredictor()
controller = HumanMouseController()

# 点击一系列按钮
buttons = [(300, 200), (500, 300), (700, 400)]

for button_pos in buttons:
    current_pos = controller.get_current_position()

    # 预测轨迹
    trajectory = predictor.predict_enhanced(
        *current_pos, *button_pos
    )

    # 移动并点击
    controller.move_and_click(*button_pos, trajectory)

    time.sleep(1)  # 等待1秒
```

### 2. 拖放操作

```python
# 从起点拖到终点
start = (200, 200)
end = (600, 400)

trajectory = predictor.predict_enhanced(*start, *end)

# 执行拖动
controller.drag_along_trajectory(trajectory, button='left')
```

### 3. 速度变化演示

```python
# 模拟不同熟练度的用户
speeds = {
    "新手": 0.5,
    "普通": 1.0,
    "熟练": 1.5,
    "专家": 2.0
}

for skill_level, speed in speeds.items():
    print(f"模拟{skill_level}用户...")
    controller.move_to_target(
        700, 500,
        trajectory,
        speed_multiplier=speed
    )
    time.sleep(2)
```

## 安全提示

1. **Failsafe功能**（默认启用）
   - 将鼠标快速移到屏幕四个角落之一可中断程序

2. **测试建议**
   ```python
   # 先测试不控制鼠标
   python test_enhanced_trajectory.py

   # 再运行交互式演示，选择安全项目
   python demo_humanlike_mouse.py
   # 选择: 4 - 轨迹分析（不控制鼠标）
   ```

3. **坐标验证**
   ```python
   # 确保坐标在屏幕范围内
   if 0 <= x <= controller.screen_width and \
      0 <= y <= controller.screen_height:
       controller.move_to_target(x, y, trajectory)
   ```

## 文件说明

| 文件 | 说明 | 是否控制鼠标 |
|------|------|------------|
| `trajectory_point.py` | 数据结构定义 | ❌ |
| `predict_trajectory.py` | 轨迹预测（已增强） | ❌ |
| `mouse_controller.py` | PyAutoGUI控制器 | ✅ |
| `test_enhanced_trajectory.py` | 测试脚本 | ❌ |
| `demo_humanlike_mouse.py` | 交互式演示 | ✅（可选） |

## 常见问题

### Q: 旧代码还能用吗？

A: 完全兼容！旧的 `predict()` 方法仍然可用：
```python
# 旧方法仍然有效
coords = predictor.predict(100, 100, 700, 500)

# 新方法提供更多信息
trajectory = predictor.predict_enhanced(100, 100, 700, 500)
```

### Q: 如何只获取坐标不要其他信息？

A: 使用 `get_coordinates()` 方法：
```python
trajectory = predictor.predict_enhanced(100, 100, 700, 500)
coords = trajectory.get_coordinates()  # 返回 [(x1,y1), ...]
```

### Q: 速度信息是怎么来的？

A: 直接从CVAE模型输出提取：
- 模型输出5个特征：[X, Y, Time, Direction, Speed]
- `predict_enhanced()` 会反归一化这些值
- Speed从 [0,1] 映射到实际单位（像素/秒）

### Q: 移动太快/太慢怎么办？

A: 使用 `speed_multiplier` 参数：
```python
# 更快
controller.move_to_target(..., speed_multiplier=2.0)

# 更慢
controller.move_to_target(..., speed_multiplier=0.5)
```

## 下一步

- 阅读完整文档: `HUMANLIKE_MOUSE_README.md`
- 运行测试: `python test_enhanced_trajectory.py`
- 体验演示: `python demo_humanlike_mouse.py`
- 查看源码: `trajectory_point.py`, `mouse_controller.py`

## 总结

✅ **新增功能**: 模型输出现在包含速度和时间信息
✅ **PyAutoGUI集成**: 可直接控制鼠标进行人类化移动
✅ **向后兼容**: 旧代码无需修改仍可使用
✅ **安全可靠**: 内置Failsafe，提供测试模式
✅ **灵活调节**: 支持速度倍数调整

开始使用吧！🚀
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
elapsed_time = controller.move_to_target(700, 500, trajectory, speed_multiplier=4)

print(f"移动完成！耗时: {elapsed_time:.3f}秒")
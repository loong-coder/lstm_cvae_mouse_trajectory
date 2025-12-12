"""
配置文件 - 所有超参数和路径配置
"""
import os

class Config:
    # 项目根目录（相对于此配置文件）
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 数据相关
    DATA_FILE = os.path.join(PROJECT_ROOT, 'mouse_trajectories.csv')

    # 特征维度
    # 输入特征: start_x, start_y, end_x, end_y, current_x, current_y, velocity, acceleration, sin_direction, cos_direction, distance, remaining_points
    INPUT_DIM = 12  # 改为12维（新增remaining_points特征）

    # LSTM参数
    LSTM_HIDDEN_DIM = 128
    LSTM_NUM_LAYERS = 2
    LSTM_DROPOUT = 0.2

    # CVAE参数
    LATENT_DIM = 32
    ENCODER_HIDDEN_DIM = 128
    DECODER_HIDDEN_DIM = 128

    # 轨迹点数量预测网络参数
    LENGTH_PREDICTOR_HIDDEN_DIM = 64
    MAX_TRAJECTORY_LENGTH = 500  # 最大轨迹长度

    # 训练参数
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 200  # 最大训练轮数
    TRAIN_SPLIT = 0.8  # 训练集比例

    # Early Stopping 参数
    EARLY_STOPPING_PATIENCE = 20  # 连续20次不改善则停止训练
    MIN_DELTA = 1e-4  # 最小改善阈值

    # KL散度权重（CVAE损失）
    KL_WEIGHT = 0.001

    # 损失函数权重
    ENDPOINT_WEIGHT = 1.0  # 终点损失权重
    SMOOTHNESS_WEIGHT = 0.1  # 平滑度损失权重
    LENGTH_CONSISTENCY_WEIGHT = 5.0  # 长度一致性损失权重（主模型训练时，使模型生成的轨迹点数接近预测的点数）
    LENGTH_PREDICTOR_WEIGHT = 2.0  # 长度预测器损失权重（训练长度预测器时使用的权重）

    # 模型保存路径
    MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models/')
    BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models/best_model.pth')

    # 设备
    DEVICE = 'cuda'  # 'cuda' or 'cpu'

    # 数据归一化参数（会在训练时计算）
    NORMALIZE_COORDS = True
    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 1080
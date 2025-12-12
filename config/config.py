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
    # 输入特征: start_x, start_y, end_x, end_y, current_x, current_y, velocity, acceleration, sin_direction, cos_direction, distance
    INPUT_DIM = 11  # 改为11维（direction拆分为sin和cos）

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

    # 模型保存路径
    MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models/')
    BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models/best_model.pth')

    # 设备
    DEVICE = 'cuda'  # 'cuda' or 'cpu'

    # 数据归一化参数（会在训练时计算）
    NORMALIZE_COORDS = True
    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 1080
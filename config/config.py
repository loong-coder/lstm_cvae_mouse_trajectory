"""
配置文件 - 所有超参数和路径配置
"""

class Config:
    # 数据相关
    DATA_FILE = 'mouse_trajectories.csv'

    # 特征维度
    # 输入特征: start_x, start_y, end_x, end_y, current_x, current_y, velocity, acceleration, direction, distance
    INPUT_DIM = 10

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
    NUM_EPOCHS = 100
    TRAIN_SPLIT = 0.8  # 训练集比例

    # KL散度权重（CVAE损失）
    KL_WEIGHT = 0.001

    # 模型保存路径
    MODEL_SAVE_PATH = 'models/'
    BEST_MODEL_PATH = 'models/best_model.pth'
    LENGTH_PREDICTOR_PATH = 'models/length_predictor.pth'

    # 设备
    DEVICE = 'cuda'  # 'cuda' or 'cpu'

    # 数据归一化参数（会在训练时计算）
    NORMALIZE_COORDS = True
    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 1080
"""
配置文件 - 所有超参数和路径配置
"""
import os

class Config:
    # 项目根目录（相对于此配置文件）
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 数据相关
    DATA_FILE = os.path.join(PROJECT_ROOT, 'mouse_trajectories.csv')

    # 模型参数
    FEAT_DIM = 5  # 特征维度：current_x, current_y, velocity, acceleration, direction
    COND_DIM = 4  # 条件维度：start_x, start_y, end_x, end_y
    LATENT_DIM = 16  # 潜在空间维度
    HIDDEN_DIM = 128  # LSTM 隐藏层维度

    # 数据参数
    SEQ_LEN = 20  # 序列长度

    # 训练参数
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    TRAIN_SPLIT = 0.8  # 训练集比例

    # 损失函数权重
    KLD_WEIGHT = 0.01  # KL 散度权重

    # Teacher Forcing
    TEACHER_FORCING_RATIO = 0.5  # 训练时使用真值的概率

    # 模型保存路径
    MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models/')
    BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models/best_model.pth')

    # 设备
    DEVICE = 'cuda'  # 'cuda' or 'cpu'
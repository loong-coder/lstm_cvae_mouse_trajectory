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
    FEAT_DIM = 6  # 特征维度：current_x, current_y, velocity, acceleration, direction, position_encoding
    COND_DIM = 4  # 条件维度：start_x, start_y, end_x, end_y
    LATENT_DIM = 32  # 潜在空间维度（从16增加到32以提高表达能力）
    HIDDEN_DIM = 128  # LSTM 隐藏层维度

    # 数据参数
    MAX_SEQ_LEN = 50  # 最大序列长度
    MIN_SEQ_LEN = 5   # 最小序列长度
    PIXELS_PER_STEP = 30  # 每步平均像素数，用于计算目标序列长度

    # 训练参数
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0005  # 学习率（从0.001降低到0.0005，提高训练稳定性）
    NUM_EPOCHS = 100
    TRAIN_SPLIT = 0.8  # 训练集比例

    # 早停参数
    EARLY_STOPPING_PATIENCE = 15  # 验证损失多少个epoch不下降就停止训练
    MIN_DELTA = 1e-4  # 最小改善阈值

    # 损失函数权重 - KLD Annealing
    KLD_WEIGHT_START = 0.0  # KLD权重起始值
    KLD_WEIGHT_END = 0.1  # KLD权重最终值
    KLD_ANNEALING_EPOCHS = 30  # 在前30个epoch逐步增加KLD权重

    # 物理约束损失权重
    VELOCITY_PROFILE_WEIGHT = 0.05  # 速度曲线约束权重（"慢-快-慢"模式）
    MINIMUM_JERK_WEIGHT = 0.01  # 最小加加速度约束权重
    SEQ_LEN_PRED_WEIGHT = 0.1  # 序列长度预测损失权重

    # Teacher Forcing
    TEACHER_FORCING_RATIO = 0.5  # 训练时使用真值的概率（已被动态衰减替代）

    # 模型保存路径
    MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models/')
    BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models/best_model.pth')

    # 设备
    DEVICE = 'cuda'  # 'cuda' or 'cpu'
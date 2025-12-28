import torch
import torch.nn as nn
import random

class TrajectoryCVAE(nn.Module):
    def __init__(self, feat_dim=5, cond_dim=4, latent_dim=16, hidden_dim=128):
        super(TrajectoryCVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # --- 【Encoder 部分】 ---
        # 作用：学习轨迹的整体风格。输入是 [特征 + 条件]
        self.enc_lstm = nn.LSTM(feat_dim + cond_dim, hidden_dim, batch_first=True)
        # 将 LSTM 最后时刻的输出映射到潜在空间的 均值(mu) 和 方差(logvar)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # --- 【Decoder 部分】 ---
        # 作用：根据风格 z 和条件 c，逐步生成轨迹。
        # 输入：[当前时刻特征 + z + 条件]
        self.dec_input_dim = feat_dim + latent_dim + cond_dim
        self.dec_lstm = nn.LSTM(self.dec_input_dim, hidden_dim, batch_first=True)
        # 将隐藏状态映射回特征空间（6维：x, y, v, a, d, pos）
        self.fc_out = nn.Linear(hidden_dim, feat_dim)

        # --- 【序列长度预测器】 ---
        # 作用：根据起点终点（条件c）预测应该生成多少个点
        # 这样模型可以学习距离→点数的映射关系
        self.seq_len_predictor = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 输出一个数：预测的序列长度
        )

    @staticmethod
    def calculate_seq_len(start, end, pixels_per_step=30):
        """
        根据起点终点距离动态计算序列长度

        Args:
            start: (x, y) 起点坐标
            end: (x, y) 终点坐标
            pixels_per_step: 每步移动的平均像素数（默认30像素/步）

        Returns:
            seq_len: 合理的序列长度（最少5个点，最多50个点）
        """
        import math

        # 计算欧氏距离
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.sqrt(dx**2 + dy**2)

        # 根据距离计算点数：distance / pixels_per_step
        # 距离越远，点数越多
        seq_len = int(distance / pixels_per_step) + 1

        # 限制在合理范围：最少5个点（避免轨迹过于简单），最多50个点（避免计算开销）
        seq_len = max(5, min(50, seq_len))

        return seq_len

    def reparameterize(self, mu, logvar):
        """重参数化技巧：使随机采样过程可导"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c, teacher_forcing_ratio=0.5):
        """
        x: 真实轨迹 [Batch, Seq, 6]
        c: 起终点条件 [Batch, 4]
        teacher_forcing_ratio: 训练时使用真值作为下一步输入的概率

        Returns:
            recon_x: 重构的轨迹 [Batch, Seq, 6]
            mu: 潜在空间均值 [Batch, latent_dim]
            logvar: 潜在空间方差 [Batch, latent_dim]
            pred_seq_len: 预测的序列长度 [Batch, 1]
        """
        batch_size, seq_len, _ = x.size()

        # 0. 预测序列长度（辅助任务）
        # 模型学习从起点终点预测应该有多少个点
        pred_seq_len = self.seq_len_predictor(c)  # [Batch, 1]

        # 1. Encoder：提取轨迹特征
        # c_expand 形状：[Batch, Seq, 4]，为了与 x 拼接
        c_expand = c.unsqueeze(1).expand(-1, seq_len, -1)
        enc_in = torch.cat([x, c_expand], dim=-1) # [Batch, Seq, 10]
        _, (h_n, c_n) = self.enc_lstm(enc_in)

        mu = self.fc_mu(h_n[-1])       # [Batch, latent_dim]
        logvar = self.fc_logvar(h_n[-1])
        z = self.reparameterize(mu, logvar) # [Batch, latent_dim]

        # 2. Decoder：逐步生成轨迹
        # 准备 z 和 c 的拼接体，它们在每个时间步都是固定的输入
        z_c = torch.cat([z, c], dim=-1).unsqueeze(1) # [Batch, 1, latent+cond]

        outputs = []
        # 第一步的输入通常是真实轨迹的起始点
        curr_input = x[:, 0, :].unsqueeze(1) # [Batch, 1, 6]
        # ⚠️ 关键改动：Decoder从空状态开始，不使用Encoder状态
        # 这样训练和推理完全一致，强制z包含所有必要信息
        h_d, c_d = None, None

        for t in range(seq_len):
            # 将 [当前点, z, c] 拼接
            dec_in = torch.cat([curr_input, z_c], dim=-1) # [Batch, 1, 6+32+4]
            # 第一次迭代时h_d和c_d是None，LSTM会自动初始化为0
            if h_d is None:
                out, (h_d, c_d) = self.dec_lstm(dec_in)
            else:
                out, (h_d, c_d) = self.dec_lstm(dec_in, (h_d, c_d))
            pred = self.fc_out(out) # 预测当前步或下一步的值
            outputs.append(pred)

            # Teacher Forcing：决定下一步是用预测值还是真值
            is_teacher = random.random() < teacher_forcing_ratio
            if is_teacher and t < seq_len - 1:
                curr_input = x[:, t+1, :].unsqueeze(1) # 下一步用真实点
            else:
                curr_input = pred # 下一步用刚才预测的点

        return torch.cat(outputs, dim=1), mu, logvar, pred_seq_len

    def predict_seq_len(self, c, add_noise=False, noise_std=0.1):
        """使用模型预测序列长度

        Args:
            c: 条件向量 [batch, 4] (start_x, start_y, end_x, end_y)
            add_noise: 是否添加随机噪声以增加多样性（推理时使用）
            noise_std: 噪声标准差（相对于预测值的比例）

        Returns:
            预测的序列长度（整数），限制在[5, 50]范围内
        """
        with torch.no_grad():
            pred_len = self.seq_len_predictor(c)  # [batch, 1]
            pred_len = pred_len.squeeze(-1)  # [batch]

            # 添加随机噪声以增加多样性
            if add_noise and self.training == False:  # 只在推理时添加
                noise = torch.randn_like(pred_len.float()) * noise_std * pred_len.float()
                pred_len = pred_len.float() + noise

            # 四舍五入并限制范围
            pred_len = torch.round(pred_len).clamp(5, 50).int()
        return pred_len

    def inference(self, c, seq_len=None, use_predicted_len=False, add_length_noise=True):
        """推理模式：完全不需要真实轨迹，只需起点终点

        Args:
            c: 条件向量 [batch, 4]
            seq_len: 序列长度（如果为None且use_predicted_len=False，使用默认值20）
            use_predicted_len: 是否使用模型预测的序列长度
            add_length_noise: 是否为预测长度添加随机噪声以增加多样性
        """
        batch_size = c.size(0)
        device = c.device

        # 决定使用的序列长度
        if use_predicted_len:
            pred_lens = self.predict_seq_len(c, add_noise=add_length_noise)  # [batch]
            # 如果batch size = 1，直接使用预测值
            if batch_size == 1:
                seq_len = pred_lens[0].item()
            else:
                # batch size > 1，使用平均值（或者可以逐个生成）
                seq_len = int(pred_lens.float().mean().item())
        elif seq_len is None:
            seq_len = 20  # 默认值

        # 从标准正态分布中随机采样"风格"
        z = torch.randn(batch_size, self.latent_dim).to(device)
        z_c = torch.cat([z, c], dim=-1).unsqueeze(1)

        # 从条件中提取起点坐标 (start_x, start_y)，速度、加速度、方向初始化为0，位置编码=0.0
        start_xy = c[:, 0:2]  # [batch, 2]
        curr_input = torch.cat([
            start_xy,
            torch.zeros(batch_size, 3).to(device),  # velocity, acceleration, direction
            torch.zeros(batch_size, 1).to(device)   # position_encoding (初始为0)
        ], dim=1).unsqueeze(1)  # [batch, 1, 6]

        outputs = []
        h_d, c_d = None, None
        for t in range(seq_len):
            dec_in = torch.cat([curr_input, z_c], dim=-1)
            # 第一次迭代时h_d和c_d是None，LSTM会自动初始化为0
            if h_d is None:
                out, (h_d, c_d) = self.dec_lstm(dec_in)
            else:
                out, (h_d, c_d) = self.dec_lstm(dec_in, (h_d, c_d))
            pred = self.fc_out(out)  # [batch, 1, 6]

            # 手动设置位置编码为当前进度 (t / seq_len)
            # 这样模型在推理时也能知道当前在轨迹的什么阶段
            pred[:, :, 5] = t / (seq_len - 1) if seq_len > 1 else 0  # 位置编码：0, 1/19, 2/19, ..., 1

            outputs.append(pred)
            curr_input = pred # 推理时完全自回归

        return torch.cat(outputs, dim=1)


def elbo_loss(recon_x, x, mu, logvar, kld_weight=0.01):
    """
    ELBO 损失函数：MSE（重构度） + KLD（分布正规化）
    """
    # 1. 重构损失：生成的点要尽可能接近真实点
    mse = nn.MSELoss()(recon_x, x)

    # 2. KL 散度：强制 Encoder 输出的 mu/logvar 接近标准正态分布 N(0,1)
    # 这样推理时从 N(0,1) 采样 z 才有意义
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # 这里的 kld 做了均值化处理，防止数值过大
    kld = kld / x.size(0)

    return mse + kld * kld_weight


def velocity_profile_loss(trajectory):
    """
    速度曲线约束损失 - 强制执行"慢-快-慢"模式（Fitts's Law）

    人类运动遵循贝尔曲线速度分布：
    - 开始阶段（0-30%）：慢速启动，加速
    - 中间阶段（30-70%）：高速移动
    - 结束阶段（70-100%）：减速，精确定位

    Args:
        trajectory: [batch, seq, feat_dim] 生成的轨迹，feat_dim包含velocity（索引2）

    Returns:
        loss: 速度曲线与理想贝尔曲线的差异
    """
    # 自动获取序列长度
    seq_len = trajectory.size(1)

    # 提取速度特征（索引2）
    velocity = trajectory[:, :, 2]  # [batch, seq]

    # 创建理想的贝尔曲线速度模板（使用高斯分布）
    # 峰值在中间，两端接近0
    t = torch.linspace(0, 1, seq_len).to(trajectory.device)  # [0, 1/(seq_len-1), ..., 1]
    # 高斯分布：exp(-((t-0.5)/sigma)^2)，sigma=0.2使曲线不会太宽或太窄
    ideal_velocity_profile = torch.exp(-((t - 0.5) / 0.2) ** 2)  # [seq]

    # 将每个batch的速度归一化到[0,1]，然后与理想曲线对比
    # 这样可以关注速度变化模式，而不是绝对速度值
    batch_size = velocity.size(0)
    loss = 0.0
    for i in range(batch_size):
        v = velocity[i]  # [seq]
        # 归一化：(v - min) / (max - min + eps)
        v_min = v.min()
        v_max = v.max()
        v_normalized = (v - v_min) / (v_max - v_min + 1e-8)

        # 计算与理想曲线的MSE
        loss += nn.MSELoss()(v_normalized, ideal_velocity_profile)

    return loss / batch_size


def minimum_jerk_loss(trajectory):
    """
    最小加加速度约束（Minimum Jerk Principle）

    人类运动倾向于最小化"jerk"（加速度的导数，即加加速度）
    这使得运动更加平滑、自然、节省能量

    数学上：jerk = d³x/dt³ = Δ(acceleration)

    Args:
        trajectory: [batch, seq, feat_dim] 生成的轨迹

    Returns:
        loss: 加加速度的平方和（越小越平滑）
    """
    # 提取加速度特征（索引3）
    acceleration = trajectory[:, :, 3]  # [batch, seq]

    # 计算加速度的变化率（加加速度）
    # jerk[t] = acceleration[t+1] - acceleration[t]
    jerk = acceleration[:, 1:] - acceleration[:, :-1]  # [batch, seq-1]

    # 最小化jerk的平方和（平滑性约束）
    # 使用平方和而不是MSE，因为我们希望所有jerk都接近0
    jerk_squared = jerk ** 2
    loss = jerk_squared.mean()

    return loss


def seq_len_prediction_loss(pred_seq_len, true_seq_len):
    """
    序列长度预测损失

    让模型学习从起点终点预测应该生成多少个点
    这样模型可以自动适应不同距离的轨迹

    Args:
        pred_seq_len: 预测的序列长度 [batch, 1]
        true_seq_len: 真实的序列长度 [batch] 或标量

    Returns:
        loss: MSE损失
    """
    # 确保 true_seq_len 是 tensor
    if not isinstance(true_seq_len, torch.Tensor):
        true_seq_len = torch.tensor([true_seq_len], dtype=torch.float32, device=pred_seq_len.device)

    # 如果是标量，扩展到batch维度
    if true_seq_len.dim() == 0:
        true_seq_len = true_seq_len.unsqueeze(0).expand(pred_seq_len.size(0))

    # 确保形状匹配
    if true_seq_len.dim() == 1:
        true_seq_len = true_seq_len.unsqueeze(-1)  # [batch, 1]

    # MSE损失
    return nn.MSELoss()(pred_seq_len, true_seq_len.float())
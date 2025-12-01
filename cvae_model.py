import torch
import torch.nn as nn
import math

from trajectory_dataset import MAX_TRAJECTORY_LENGTH


class TrajectoryPredictorCVAE(nn.Module):
    """
    条件变分自编码器 (Conditional VAE) 用于轨迹预测

    优势：
    1. 捕捉轨迹生成的随机性和多样性
    2. 并行生成整条轨迹（比自回归快）
    3. 潜在空间连续，生成轨迹更平滑
    4. 可以通过采样不同的z生成多种可能的轨迹
    """

    def __init__(self, task_context_size=5, latent_dim=64, hidden_size=256,
                 output_size=5, max_len=MAX_TRAJECTORY_LENGTH):
        """
        Args:
            task_context_size: 任务上下文维度 (5: start_x, start_y, end_x, end_y, distance)
            latent_dim: 潜在变量 z 的维度
            hidden_size: 隐藏层维度
            output_size: 每个点的特征维度 (5: X, Y, Time, Direction, Speed)
            max_len: 最大轨迹长度
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_len = max_len

        # ========== 编码器 (Encoder): q(z|x,c) ==========
        # 输入: 目标轨迹 + 任务上下文
        # 输出: 潜在变量 z 的均值和方差

        # 轨迹编码器（使用LSTM压缩整条轨迹）
        self.trajectory_encoder_lstm = nn.LSTM(
            input_size=output_size,
            hidden_size=hidden_size // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True  # 双向LSTM，更好地捕捉全局信息
        )

        # 编码器MLP：将LSTM输出 + 任务上下文 映射到潜在空间
        self.encoder_fc = nn.Sequential(
            nn.Linear(hidden_size + task_context_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 潜在变量的均值和对数方差
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

        # ========== 解码器 (Decoder): p(x|z,c) ==========
        # 输入: 潜在变量 z + 任务上下文
        # 输出: 完整轨迹序列

        # 解码器MLP：将 z + 上下文 映射到LSTM初始状态
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim + task_context_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size * 2)  # h0 和 c0
        )

        # 解码器LSTM：生成轨迹序列（自回归方式）
        self.decoder_lstm = nn.LSTM(
            input_size=output_size + hidden_size,  # 上一步输出 + 条件向量
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )

        # 输出层：将LSTM输出映射到轨迹点特征
        # 分别处理不同特征，以便应用不同的激活函数
        self.output_layer = nn.Linear(hidden_size, output_size)

        # 注意：我们将在 decode() 中对输出应用特定的激活函数
        # - X, Y (相对坐标): 使用 Tanh 限制在 [-1, 1] 范围
        # - Time: 使用 Softplus 确保非负
        # - Direction: 使用 Tanh 限制在 [-1, 1] (对应 [-π, π])
        # - Speed: 使用 Sigmoid 限制在 [0, 1]

        # 位置编码（正弦余弦编码）
        self.position_encoding = self._create_position_encoding(max_len, hidden_size)

        # ========== 辅助网络 ==========

        # 序列长度预测器
        self.length_predictor = nn.Sequential(
            nn.Linear(task_context_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # 输出0-1之间的值，表示序列长度占max_len的比例
        )

        # 先验网络：p(z|c) - 仅基于任务上下文预测 z（推理时使用）
        self.prior_net = nn.Sequential(
            nn.Linear(task_context_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.prior_mu = nn.Linear(hidden_size, latent_dim)
        self.prior_logvar = nn.Linear(hidden_size, latent_dim)

        # 权重初始化
        self._initialize_weights()

    def _create_position_encoding(self, max_len, hidden_size):
        """创建正弦余弦位置编码"""
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))

        pe = torch.zeros(max_len, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return nn.Parameter(pe, requires_grad=False)

    def _initialize_weights(self):
        """初始化模型权重以提高数值稳定性"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

        # 特别初始化 logvar 层，使其初始输出接近0（对应方差接近1）
        # 使用小的权重而不是0，以避免CUDA上的数值问题
        nn.init.normal_(self.fc_logvar.weight, mean=0, std=0.001)
        nn.init.constant_(self.fc_logvar.bias, -1.0)  # log(0.36) ≈ -1，初始方差约0.36
        nn.init.normal_(self.prior_logvar.weight, mean=0, std=0.001)
        nn.init.constant_(self.prior_logvar.bias, -1.0)

    def encode(self, trajectory, task_context):
        """
        编码器：从轨迹和任务上下文推断潜在变量 z

        Args:
            trajectory: (Batch, max_len, output_size) - 目标轨迹
            task_context: (Batch, task_context_size) - 任务上下文

        Returns:
            mu: (Batch, latent_dim) - 潜在变量均值
            logvar: (Batch, latent_dim) - 潜在变量对数方差
        """
        batch_size = trajectory.size(0)

        # 1. 使用双向LSTM编码整条轨迹
        lstm_out, (h_n, c_n) = self.trajectory_encoder_lstm(trajectory)

        # 2. 取最后一个时间步的隐藏状态（双向拼接）
        # h_n: (num_layers * 2, batch, hidden_size // 2)
        trajectory_embedding = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (Batch, hidden_size)

        # 3. 拼接轨迹编码和任务上下文
        combined = torch.cat([trajectory_embedding, task_context], dim=1)

        # 4. 通过MLP得到潜在变量的分布参数
        h = self.encoder_fc(combined)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def prior(self, task_context):
        """
        先验网络：仅基于任务上下文预测 z 的分布
        推理时使用（因为没有目标轨迹）

        Args:
            task_context: (Batch, task_context_size)

        Returns:
            mu: (Batch, latent_dim)
            logvar: (Batch, latent_dim)
        """
        h = self.prior_net(task_context)
        mu = self.prior_mu(h)
        logvar = self.prior_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧：z = mu + sigma * epsilon

        Args:
            mu: (Batch, latent_dim)
            logvar: (Batch, latent_dim)

        Returns:
            z: (Batch, latent_dim)
        """
        # 限制logvar范围，防止exp溢出
        logvar = torch.clamp(logvar, min=-10, max=10)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z, task_context, target_trajectory=None):
        """
        解码器：从潜在变量 z 和任务上下文生成轨迹（自回归方式）

        Args:
            z: (Batch, latent_dim) - 潜在变量
            task_context: (Batch, task_context_size) - 任务上下文
            target_trajectory: (Batch, max_len, output_size) - 目标轨迹（训练时用于 teacher forcing）

        Returns:
            trajectory: (Batch, max_len, output_size) - 生成的轨迹
        """
        batch_size = z.size(0)
        device = z.device

        # 1. 拼接 z 和任务上下文，创建条件向量
        combined = torch.cat([z, task_context], dim=1)

        # 2. 通过MLP得到LSTM初始状态
        decoder_input = self.decoder_fc(combined)
        h0, c0 = torch.chunk(decoder_input, 2, dim=1)
        h0 = h0.unsqueeze(0).repeat(2, 1, 1).contiguous()  # (2, Batch, hidden_size)
        c0 = c0.unsqueeze(0).repeat(2, 1, 1).contiguous()

        # 3. 创建条件向量（z 编码的信息），在每个时间步都使用
        condition_vector = self.decoder_fc[0](combined)  # (Batch, hidden_size)

        # 4. 自回归生成序列
        predictions = []
        hidden_state = (h0, c0)

        # 初始输入为零向量
        current_input = torch.zeros(batch_size, self.output_size, device=device)

        use_teacher_forcing = self.training and target_trajectory is not None

        for t in range(self.max_len):
            # 4.1 拼接上一步的输出和条件向量
            decoder_input_t = torch.cat([current_input, condition_vector], dim=1).unsqueeze(1)  # (Batch, 1, output_size + hidden_size)

            # 4.2 LSTM 前向传播一步
            lstm_out, hidden_state = self.decoder_lstm(decoder_input_t, hidden_state)

            # 4.3 解码得到预测
            raw_output = self.output_layer(lstm_out.squeeze(1))  # (Batch, output_size)

            # 4.4 应用激活函数约束不同特征到合理范围
            # output_size = 5: [X, Y, Time, Direction, Speed]
            prediction = torch.cat([
                torch.tanh(raw_output[:, 0:1]),      # X: 相对坐标，限制在 [-1, 1]
                torch.tanh(raw_output[:, 1:2]),      # Y: 相对坐标，限制在 [-1, 1]
                torch.nn.functional.softplus(raw_output[:, 2:3]),  # Time: 非负
                torch.tanh(raw_output[:, 3:4]),      # Direction: 限制在 [-1, 1]
                torch.sigmoid(raw_output[:, 4:5])    # Speed: 限制在 [0, 1]
            ], dim=1)

            predictions.append(prediction)

            # 4.4 准备下一步的输入
            if use_teacher_forcing:
                # 训练时使用 teacher forcing
                current_input = target_trajectory[:, t, :]
            else:
                # 推理时使用预测值
                current_input = prediction

        # 5. 堆叠所有预测
        trajectory = torch.stack(predictions, dim=1)  # (Batch, max_len, output_size)

        return trajectory

    def forward(self, task_context, target_trajectory=None):
        """
        前向传播

        Args:
            task_context: (Batch, task_context_size) - 任务上下文
            target_trajectory: (Batch, max_len, output_size) - 目标轨迹（训练时提供）

        Returns:
            reconstruction: (Batch, max_len, output_size) - 重建/生成的轨迹
            mu: (Batch, latent_dim) - 潜在变量均值
            logvar: (Batch, latent_dim) - 潜在变量对数方差
            predicted_length_ratio: (Batch, 1) - 预测的序列长度比例
        """
        # 预测序列长度
        predicted_length_ratio = self.length_predictor(task_context)

        if self.training and target_trajectory is not None:
            # ===== 训练模式：使用编码器 =====
            # 1. 从目标轨迹编码得到潜在变量
            mu, logvar = self.encode(target_trajectory, task_context)

            # 2. 重参数化采样
            z = self.reparameterize(mu, logvar)
        else:
            # ===== 推理模式：使用先验网络 =====
            # 1. 从任务上下文预测潜在变量分布
            mu, logvar = self.prior(task_context)

            # 2. 从先验分布采样
            z = self.reparameterize(mu, logvar)

        # 3. 解码生成轨迹（训练时传入 target_trajectory 用于 teacher forcing）
        reconstruction = self.decode(z, task_context, target_trajectory)

        return reconstruction, mu, logvar, predicted_length_ratio

    def sample(self, task_context, num_samples=1):
        """
        从先验分布采样生成多条可能的轨迹

        Args:
            task_context: (Batch, task_context_size)
            num_samples: 每个任务生成多少条轨迹

        Returns:
            trajectories: (Batch, num_samples, max_len, output_size)
        """
        self.eval()
        with torch.no_grad():
            batch_size = task_context.size(0)
            device = task_context.device

            # 扩展task_context以生成多个样本
            task_context_expanded = task_context.unsqueeze(1).repeat(1, num_samples, 1)
            task_context_expanded = task_context_expanded.view(batch_size * num_samples, -1)

            # 从先验分布采样
            mu, logvar = self.prior(task_context_expanded)
            z = self.reparameterize(mu, logvar)

            # 解码生成轨迹
            trajectories = self.decode(z, task_context_expanded)

            # 重塑为 (Batch, num_samples, max_len, output_size)
            trajectories = trajectories.view(batch_size, num_samples, self.max_len, self.output_size)

        return trajectories
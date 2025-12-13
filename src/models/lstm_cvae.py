"""
模型定义 - LSTM+CVAE组合模型（集成轨迹长度预测）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import Config


class CVAEEncoder(nn.Module):
    """
    CVAE编码器：将轨迹序列编码为潜在空间的分布（均值和方差）
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(CVAEEncoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # 输出均值和对数方差
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.relu = nn.ReLU()

        # 初始化logvar为小值，提高数值稳定性
        nn.init.constant_(self.fc_logvar.bias, -1.0)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) 输入序列
        Returns:
            mu: (batch, latent_dim) 潜在变量的均值
            logvar: (batch, latent_dim) 潜在变量的对数方差
        """
        # 对序列进行平均池化
        x = torch.mean(x, dim=1)  # (batch, input_dim)

        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class PriorNetwork(nn.Module):
    """
    先验网络：p(z|c) - 仅基于任务上下文预测潜在变量分布
    推理时使用，使生成的轨迹与任务相关
    """
    def __init__(self, context_dim, hidden_dim, latent_dim):
        super(PriorNetwork, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 初始化logvar为-1.0（对应方差约0.36），提高数值稳定性
        nn.init.constant_(self.fc_logvar.bias, -1.0)

    def forward(self, start_point, end_point):
        """
        Args:
            start_point: (batch, 2) 起点坐标
            end_point: (batch, 2) 终点坐标
        Returns:
            mu: (batch, latent_dim) 先验分布均值
            logvar: (batch, latent_dim) 先验分布对数方差
        """
        # 拼接起点和终点作为任务上下文
        context = torch.cat([start_point, end_point], dim=1)  # (batch, 4)
        h = self.fc(context)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class CVAEDecoder(nn.Module):
    """
    CVAE解码器：从潜在变量和LSTM隐藏状态生成轨迹点
    """
    def __init__(self, latent_dim, lstm_hidden_dim, output_dim):
        super(CVAEDecoder, self).__init__()

        # 将潜在变量投影到与LSTM输出相同的维度
        self.fc1 = nn.Linear(latent_dim + lstm_hidden_dim, lstm_hidden_dim)
        self.fc2 = nn.Linear(lstm_hidden_dim, output_dim)

        self.relu = nn.ReLU()

    def forward(self, z, lstm_hidden):
        """
        Args:
            z: (batch, latent_dim) 潜在变量
            lstm_hidden: (batch, seq_len, lstm_hidden_dim) LSTM隐藏状态
        Returns:
            output: (batch, seq_len, output_dim) 生成的轨迹点
        """
        batch_size, seq_len, _ = lstm_hidden.shape

        # 将z扩展到每个时间步
        z_expanded = z.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, latent_dim)

        # 拼接潜在变量和LSTM隐藏状态
        combined = torch.cat([z_expanded, lstm_hidden], dim=2)  # (batch, seq_len, latent_dim + lstm_hidden_dim)

        h = self.relu(self.fc1(combined))
        output = self.fc2(h)  # (batch, seq_len, output_dim)

        return output


class LSTMCVAE(nn.Module):
    """
    主模型：LSTM + CVAE 组合（改进版：添加先验网络）
    用于生成鼠标移动轨迹
    """
    def __init__(self, config):
        super(LSTMCVAE, self).__init__()

        self.config = config
        self.norm_stats = None  # 归一化统计信息

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=config.INPUT_DIM,
            hidden_size=config.LSTM_HIDDEN_DIM,
            num_layers=config.LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=config.LSTM_DROPOUT if config.LSTM_NUM_LAYERS > 1 else 0
        )

        # CVAE编码器（训练时使用）
        self.encoder = CVAEEncoder(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.ENCODER_HIDDEN_DIM,
            latent_dim=config.LATENT_DIM
        )

        # 先验网络（推理时使用）⭐新增⭐
        self.prior_net = PriorNetwork(
            context_dim=4,  # start_x, start_y, end_x, end_y
            hidden_dim=config.ENCODER_HIDDEN_DIM,
            latent_dim=config.LATENT_DIM
        )

        # CVAE解码器
        self.decoder = CVAEDecoder(
            latent_dim=config.LATENT_DIM,
            lstm_hidden_dim=config.LSTM_HIDDEN_DIM,
            output_dim=config.INPUT_DIM
        )

    def set_norm_stats(self, norm_stats):
        """设置归一化统计信息（用于generate方法）"""
        self.norm_stats = norm_stats

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧：从N(mu, var)中采样（改进：添加数值稳定性）
        """
        # 限制logvar范围，防止exp溢出或下溢
        logvar = torch.clamp(logvar, min=-10, max=10)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, start_point, end_point):
        """
        前向传播（改进：训练/推理分离）
        Args:
            x: (batch, seq_len, input_dim) 输入特征
            start_point: (batch, 2) 起点
            end_point: (batch, 2) 终点
        Returns:
            reconstructed: (batch, seq_len, input_dim) 重建的轨迹
            mu_posterior: (batch, latent_dim) 后验分布均值（训练时）
            logvar_posterior: (batch, latent_dim) 后验分布对数方差（训练时）
            mu_prior: (batch, latent_dim) 先验分布均值
            logvar_prior: (batch, latent_dim) 先验分布对数方差
        """
        # LSTM编码
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_hidden_dim)

        # 计算先验分布 p(z|c)
        mu_prior, logvar_prior = self.prior_net(start_point, end_point)

        if self.training:
            # 训练模式：从真实轨迹编码得到后验分布 q(z|x,c)
            mu_posterior, logvar_posterior = self.encoder(x)
            z = self.reparameterize(mu_posterior, logvar_posterior)
        else:
            # 推理模式：从先验分布采样
            z = self.reparameterize(mu_prior, logvar_prior)
            mu_posterior, logvar_posterior = None, None

        # CVAE解码
        reconstructed = self.decoder(z, lstm_out)

        return reconstructed, mu_posterior, logvar_posterior, mu_prior, logvar_prior

    def generate(self, start_point, end_point, trajectory_length, num_samples=1, endpoint_guidance_weight=0.3, temperature=0.8):
        """
        生成模式：给定起点和终点，生成轨迹（带持续终点逼近）
        Args:
            start_point: (batch, 2) 起点坐标（归一化）
            end_point: (batch, 2) 终点坐标（归一化）
            trajectory_length: int 轨迹长度（包括起点）
            num_samples: int 生成样本数量
            endpoint_guidance_weight: float 终点引导权重（0-1），越大越倾向于直接走向终点
            temperature: float 温度参数（0.3-1.0），控制随机性，越小越确定性
        Returns:
            generated_trajectory: (batch, trajectory_length, input_dim) 生成的轨迹
        """
        batch_size = start_point.shape[0]
        device = start_point.device

        # ⭐改进⭐：从先验网络预测的分布采样（而不是标准正态分布）
        mu_prior, logvar_prior = self.prior_net(start_point, end_point)
        logvar_prior = torch.clamp(logvar_prior, min=-10, max=10)
        std_prior = torch.exp(0.5 * logvar_prior) * temperature
        eps = torch.randn_like(std_prior)
        z = mu_prior + eps * std_prior

        # 创建初始输入序列（从起点开始）
        current_pos = start_point.clone()  # (batch, 2)

        generated_trajectory = []

        # 初始化LSTM隐藏状态
        h = torch.zeros(self.config.LSTM_NUM_LAYERS, batch_size, self.config.LSTM_HIDDEN_DIM).to(device)
        c = torch.zeros(self.config.LSTM_NUM_LAYERS, batch_size, self.config.LSTM_HIDDEN_DIM).to(device)

        # 第一个点：手动构造起点
        # 剩余点数：还需要生成 trajectory_length - 1 个点
        remaining_points_tensor = torch.full((batch_size, 1), trajectory_length - 1, dtype=torch.float32).to(device)
        first_point = torch.cat([
            start_point,  # start_x, start_y
            end_point,    # end_x, end_y
            start_point,  # current_x, current_y = start_x, start_y
            torch.zeros(batch_size, 5).to(device),  # velocity, acceleration, sin_dir, cos_dir, distance = 0
            remaining_points_tensor  # remaining_points = trajectory_length - 1
        ], dim=1).unsqueeze(1)  # (batch, 1, input_dim)
        generated_trajectory.append(first_point)

        # 计算总距离用于归一化步长
        total_distance_to_end = torch.sqrt(torch.sum((end_point - start_point)**2, dim=1, keepdim=True))

        # 从第2个点开始预测（第1个点已经是起点）
        for t in range(trajectory_length - 1):
            # 构建当前时间步的输入特征
            # [start_x, start_y, end_x, end_y, current_x, current_y, velocity, acceleration, sin_direction, cos_direction, distance, remaining_points]

            # 计算剩余点数（不包括当前点）
            remaining_steps = trajectory_length - 1 - t

            # 计算到起点的距离（与训练数据一致）
            dx_to_start = current_pos[:, 0:1] - start_point[:, 0:1]
            dy_to_start = current_pos[:, 1:2] - start_point[:, 1:2]
            distance = torch.sqrt(dx_to_start**2 + dy_to_start**2 + 1e-8)

            # 计算到终点的距离和方向（用于引导生成）
            dx_to_end = end_point[:, 0:1] - current_pos[:, 0:1]
            dy_to_end = end_point[:, 1:2] - current_pos[:, 1:2]
            distance_to_end = torch.sqrt(dx_to_end**2 + dy_to_end**2 + 1e-8)

            # 计算当前进度（已走过的距离占比）
            progress = 1.0 - (distance_to_end / (total_distance_to_end + 1e-8))
            progress = torch.clamp(progress, 0.0, 1.0)

            if t == 0:
                # 第一步（实际是第二个点）：使用从起点到终点的方向作为初始方向
                direction_rad = torch.atan2(dy_to_end, dx_to_end)
                sin_direction = torch.sin(direction_rad)
                cos_direction = torch.cos(direction_rad)
                velocity = torch.zeros(batch_size, 1).to(device)
                acceleration = torch.zeros(batch_size, 1).to(device)
            else:
                # 后续步骤：计算相对于上一步的移动方向和速度
                prev_pos = generated_trajectory[-1][:, 0, 4:6]  # 上一步的current位置
                dx = current_pos[:, 0:1] - prev_pos[:, 0:1]
                dy = current_pos[:, 1:2] - prev_pos[:, 1:2]
                step_distance = torch.sqrt(dx**2 + dy**2 + 1e-8)

                # 移动方向
                direction_rad = torch.atan2(dy, dx)
                sin_direction = torch.sin(direction_rad)
                cos_direction = torch.cos(direction_rad)

                # 速度（简化：假设固定时间间隔）
                velocity = step_distance * 100

                # 归一化速度（与训练数据一致）
                if self.norm_stats is not None:
                    velocity = (velocity - self.norm_stats['velocity_mean']) / (self.norm_stats['velocity_std'] + 1e-8)

                # 加速度（基于速度变化）
                prev_velocity = generated_trajectory[-1][:, 0, 6:7]  # 上一步的速度（已归一化）
                acceleration = (velocity - prev_velocity) * 100

                # 归一化加速度（与训练数据一致）
                if self.norm_stats is not None:
                    acceleration = (acceleration - self.norm_stats['acceleration_mean']) / (self.norm_stats['acceleration_std'] + 1e-8)

            # 构建输入向量
            remaining_points_current = torch.full((batch_size, 1), remaining_steps, dtype=torch.float32).to(device)
            x_t = torch.cat([
                start_point,  # start_x, start_y
                end_point,    # end_x, end_y
                current_pos,  # current_x, current_y
                velocity,
                acceleration,
                sin_direction,
                cos_direction,
                distance,  # 到起点的距离，与训练数据一致
                remaining_points_current  # 剩余点数
            ], dim=1).unsqueeze(1)  # (batch, 1, input_dim)

            # LSTM前向
            lstm_out, (h, c) = self.lstm(x_t, (h, c))  # (batch, 1, lstm_hidden_dim)

            # CVAE解码
            output = self.decoder(z, lstm_out)  # (batch, 1, input_dim)

            # 让模型自主预测下一个位置
            predicted_pos = output[:, 0, 4:6]  # 模型预测的下一个位置

            # 计算理想终点方向（用于引导，但权重较小，主要靠模型学习）
            direction_to_end = torch.cat([dx_to_end, dy_to_end], dim=1)  # (batch, 2)
            direction_to_end_normalized = direction_to_end / (distance_to_end + 1e-8)

            # 计算平均步长（仅用于基本约束，不强制模型遵循）
            if remaining_steps > 0:
                average_step_length = distance_to_end / remaining_steps
            else:
                average_step_length = distance_to_end

            # 理想下一个位置（朝向终点，均匀步长）
            ideal_next_pos = current_pos + direction_to_end_normalized * average_step_length

            # 宽松的步长约束：允许模型有较大自由度（最大3倍平均步长）
            pred_step = predicted_pos - current_pos
            pred_step_length = torch.sqrt(torch.sum(pred_step**2, dim=1, keepdim=True))
            max_step_length = average_step_length * 3.0

            # 只在预测步长过大时进行限制
            if pred_step_length > max_step_length:
                pred_step = pred_step / (pred_step_length + 1e-8) * max_step_length
                predicted_pos = current_pos + pred_step

            # 动态调整引导强度（确保到达终点）
            time_progress = (t + 1) / (trajectory_length - 1) if trajectory_length > 1 else 1.0

            # ⭐修复⭐：合理的引导权重，确保能到达终点
            # endpoint_guidance_weight 控制整体强度，但要确保最后能接近100%
            if time_progress < 0.5:
                # 前50%：逐渐增加引导（0 -> 15%）
                base_weight = time_progress * 0.3
            elif time_progress < 0.8:
                # 中间30%：稳定增加（15% -> 45%）
                progress_in_mid = (time_progress - 0.5) / 0.3
                base_weight = 0.15 + 0.3 * progress_in_mid
            else:
                # 最后20%：快速增加确保到达终点（45% -> 90%）
                progress_in_last = (time_progress - 0.8) / 0.2
                base_weight = 0.45 + 0.45 * progress_in_last

            # endpoint_guidance_weight 作为整体调节系数（0.3 -> 1.5）
            # 较小值（0.3-0.5）：给模型更多自由度，更自然但可能不精确到终点
            # 较大值（0.8-1.5）：强制引导到终点，更精确但可能不够自然
            dynamic_weight = min(base_weight * (1 + endpoint_guidance_weight), 0.95)

            # 混合模型预测和理想位置
            guided_pos = predicted_pos * (1 - dynamic_weight) + ideal_next_pos * dynamic_weight

            # ⭐移除强制跳到终点的逻辑⭐，让轨迹自然接近终点

            # 更新output中的current_x, current_y
            output[:, 0, 4:6] = guided_pos

            generated_trajectory.append(output)

            # 更新当前位置为引导后的位置
            current_pos = guided_pos

        # ⭐移除强制设置最后一个点为终点的逻辑⭐，让轨迹自然到达

        # 拼接所有时间步
        generated_trajectory = torch.cat(generated_trajectory, dim=1)  # (batch, trajectory_length, input_dim)

        return generated_trajectory


def compute_trajectory_distribution(trajectory, start_point, end_point, mask, num_bins=10):
    """
    计算轨迹点在起点到终点路径上的分布

    Args:
        trajectory: (batch, seq_len, input_dim) 轨迹数据，current_x, current_y在索引4:6
        start_point: (batch, 2) 起点坐标
        end_point: (batch, 2) 终点坐标
        mask: (batch, seq_len) 有效数据mask
        num_bins: int 区间数量

    Returns:
        x_distribution: (batch, num_bins) x方向的点分布概率
        y_distribution: (batch, num_bins) y方向的点分布概率
    """
    batch_size = trajectory.shape[0]
    device = trajectory.device

    # 提取轨迹坐标 (current_x, current_y)
    coords = trajectory[:, :, 4:6]  # (batch, seq_len, 2)

    # 初始化分布
    x_distribution = torch.zeros(batch_size, num_bins, device=device)
    y_distribution = torch.zeros(batch_size, num_bins, device=device)

    for b in range(batch_size):
        # 获取有效点
        valid_mask = mask[b] > 0
        valid_coords = coords[b, valid_mask]  # (num_valid, 2)

        if valid_coords.shape[0] == 0:
            # 如果没有有效点，使用均匀分布
            x_distribution[b] = 1.0 / num_bins
            y_distribution[b] = 1.0 / num_bins
            continue

        # 起点和终点
        start = start_point[b]  # (2,)
        end = end_point[b]  # (2,)

        # 计算x和y方向的范围
        x_min, x_max = start[0], end[0]
        y_min, y_max = start[1], end[1]

        # 确保min <= max
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min

        # 添加小的epsilon避免除零
        x_range = x_max - x_min + 1e-8
        y_range = y_max - y_min + 1e-8

        # 计算每个点所在的bin（x方向）
        x_coords = valid_coords[:, 0]
        x_bins = ((x_coords - x_min) / x_range * num_bins).long()
        x_bins = torch.clamp(x_bins, 0, num_bins - 1)

        # 计算每个点所在的bin（y方向）
        y_coords = valid_coords[:, 1]
        y_bins = ((y_coords - y_min) / y_range * num_bins).long()
        y_bins = torch.clamp(y_bins, 0, num_bins - 1)

        # 统计每个bin的点数
        for i in range(num_bins):
            x_distribution[b, i] = (x_bins == i).float().sum()
            y_distribution[b, i] = (y_bins == i).float().sum()

        # 归一化为概率分布
        x_distribution[b] = x_distribution[b] / (x_distribution[b].sum() + 1e-8)
        y_distribution[b] = y_distribution[b] / (y_distribution[b].sum() + 1e-8)

    return x_distribution, y_distribution


def compute_direction_loss(trajectory, start_point, end_point, mask):
    """
    方向一致性损失：确保轨迹中每一步都朝向终点方向移动（向量化版本）

    计算方法：
    1. 对每个点，计算移动方向向量 move_vec = current - prev
    2. 计算到终点的理想方向 ideal_vec = end - current
    3. 计算两个向量的夹角余弦值（内积/模长）
    4. 损失 = 1 - cos(angle)，越偏离方向损失越大
    """
    coords = trajectory[:, :, 4:6]  # (batch, seq_len, 2)

    # 向量化：一次性计算所有时间步
    # 移动方向：当前点 - 前一个点
    move_vec = coords[:, 1:, :] - coords[:, :-1, :]  # (batch, seq_len-1, 2)

    # 理想方向：终点 - 当前点
    ideal_vec = end_point.unsqueeze(1) - coords[:, 1:, :]  # (batch, seq_len-1, 2)

    # 归一化
    move_norm = torch.sqrt(torch.sum(move_vec**2, dim=2, keepdim=True)) + 1e-8  # (batch, seq_len-1, 1)
    ideal_norm = torch.sqrt(torch.sum(ideal_vec**2, dim=2, keepdim=True)) + 1e-8  # (batch, seq_len-1, 1)

    move_vec_normalized = move_vec / move_norm
    ideal_vec_normalized = ideal_vec / ideal_norm

    # 计算余弦相似度
    cos_similarity = torch.sum(move_vec_normalized * ideal_vec_normalized, dim=2)  # (batch, seq_len-1)

    # 损失：1 - cos，应用mask
    direction_loss = (1 - cos_similarity) * mask[:, 1:]  # (batch, seq_len-1)

    # 计算平均损失
    valid_count = torch.sum(mask[:, 1:]) + 1e-8
    return torch.sum(direction_loss) / valid_count


def compute_step_uniformity_loss(trajectory, mask):
    """
    步长均匀性损失：确保相邻点之间的距离相对均匀（向量化优化版本）

    计算方法：
    1. 计算所有相邻点之间的距离
    2. 计算这些距离的变异系数 (std/mean)
    3. 变异系数越小，分布越均匀
    """
    coords = trajectory[:, :, 4:6]  # (batch, seq_len, 2)

    # 向量化：一次性计算所有相邻点的距离
    distances = torch.sqrt(torch.sum((coords[:, 1:, :] - coords[:, :-1, :])**2, dim=2))  # (batch, seq_len-1)

    # 应用mask（排除padding的距离）
    valid_distances_mask = mask[:, 1:] * mask[:, :-1]  # 两个相邻点都有效时才计算
    distances_masked = distances * valid_distances_mask  # (batch, seq_len-1)

    # 对每个batch计算变异系数
    # 为了避免除零，只计算有效距离数>=2的batch
    valid_counts = torch.sum(valid_distances_mask, dim=1)  # (batch,)

    # 计算每个batch的均值和标准差（只考虑有效距离）
    mean_dists = torch.sum(distances_masked, dim=1) / (valid_counts + 1e-8)  # (batch,)

    # 计算标准差（向量化）
    # std = sqrt(E[(x - mean)^2])
    diff_squared = ((distances - mean_dists.unsqueeze(1)) ** 2) * valid_distances_mask  # (batch, seq_len-1)
    std_dists = torch.sqrt(torch.sum(diff_squared, dim=1) / (valid_counts + 1e-8))  # (batch,)

    # 变异系数 CV = std / mean
    cv = std_dists / (mean_dists + 1e-8)  # (batch,)

    # 只计算有效样本（至少有2个有效距离）的平均CV
    valid_samples = (valid_counts >= 2).float()  # (batch,)
    cv_masked = cv * valid_samples

    num_valid_samples = torch.sum(valid_samples) + 1e-8
    return torch.sum(cv_masked) / num_valid_samples


def compute_boundary_loss(trajectory, start_point, end_point, mask):
    """
    边界约束损失：防止轨迹点偏离起点-终点直线太远（向量化优化版本）

    计算方法：
    1. 对每个点，计算到起点-终点直线的垂直距离
    2. 距离越大，损失越大（允许小幅偏离）
    """
    coords = trajectory[:, :, 4:6]  # (batch, seq_len, 2)
    batch_size, seq_len, _ = coords.shape

    # 向量化：同时计算所有batch的所有点
    # 直线向量: end - start, shape (batch, 2)
    line_vec = end_point - start_point  # (batch, 2)
    line_length_sq = torch.sum(line_vec**2, dim=1, keepdim=True) + 1e-8  # (batch, 1)
    line_length = torch.sqrt(line_length_sq)  # (batch, 1)

    # 扩展维度以便广播计算
    # start_point: (batch, 2) -> (batch, 1, 2)
    # line_vec: (batch, 2) -> (batch, 1, 2)
    start_expanded = start_point.unsqueeze(1)  # (batch, 1, 2)
    line_vec_expanded = line_vec.unsqueeze(1)  # (batch, 1, 2)
    line_length_expanded = line_length.unsqueeze(1)  # (batch, 1, 1)

    # 计算所有点到直线的距离（向量化）
    # vec_to_point: coords - start, shape (batch, seq_len, 2)
    vec_to_point = coords - start_expanded  # (batch, seq_len, 2)

    # 投影长度: (P-A)·(B-A) / ||B-A||^2, shape (batch, seq_len)
    projection_length = torch.sum(vec_to_point * line_vec_expanded, dim=2, keepdim=True) / line_length_sq.unsqueeze(1)  # (batch, seq_len, 1)

    # 投影向量: projection_length * (B-A), shape (batch, seq_len, 2)
    projection = projection_length * line_vec_expanded  # (batch, seq_len, 2)

    # 垂直向量: (P-A) - projection, shape (batch, seq_len, 2)
    perpendicular_vec = vec_to_point - projection  # (batch, seq_len, 2)

    # 垂直距离: ||perpendicular_vec||, shape (batch, seq_len)
    distance = torch.sqrt(torch.sum(perpendicular_vec**2, dim=2))  # (batch, seq_len)

    # 归一化距离（相对于直线长度）, shape (batch, seq_len)
    normalized_distance = distance / line_length_expanded.squeeze(2)  # (batch, seq_len)

    # 损失：允许10%偏离，超过则惩罚
    boundary_loss_per_point = torch.relu(normalized_distance - 0.1) ** 2  # (batch, seq_len)

    # 应用mask，只计算有效点
    boundary_loss_masked = boundary_loss_per_point * mask  # (batch, seq_len)

    # 计算平均损失
    valid_count = torch.sum(mask) + 1e-8
    return torch.sum(boundary_loss_masked) / valid_count


def compute_loss(reconstructed, target, mu_posterior, logvar_posterior, mu_prior, logvar_prior, mask,
                 kl_weight=0.001,
                 endpoint_weight=1.0,
                 smoothness_weight=0.1,
                 distribution_weight=0.5,
                 direction_weight=1.5,          # 新增
                 step_uniformity_weight=1.0,    # 新增
                 boundary_weight=0.8,           # 新增
                 num_bins=10):
    """
    增强的CVAE损失：重建损失 + KL散度损失 + 终点损失 + 平滑度损失 + 分布损失 + 几何约束损失

    Args:
        reconstructed: (batch, seq_len, input_dim) 重建的轨迹
        target: (batch, seq_len, input_dim) 目标轨迹
        mu_posterior: (batch, latent_dim) 后验分布均值 q(z|x,c)
        logvar_posterior: (batch, latent_dim) 后验分布对数方差
        mu_prior: (batch, latent_dim) 先验分布均值 p(z|c)
        logvar_prior: (batch, latent_dim) 先验分布对数方差
        mask: (batch, seq_len) 有效数据的mask
        kl_weight: KL散度的权重
        endpoint_weight: 终点损失的权重
        smoothness_weight: 平滑度损失的权重
        distribution_weight: 分布损失的权重
        direction_weight: 方向一致性损失权重
        step_uniformity_weight: 步长均匀性损失权重
        boundary_weight: 边界约束损失权重
        num_bins: 区间数量
    Returns:
        total_loss: 总损失
        recon_loss: 重建损失
        kl_loss: KL散度损失
        endpoint_loss: 终点损失
        smoothness_loss: 平滑度损失
        distribution_loss: 分布损失
        direction_loss: 方向一致性损失
        step_uniformity_loss: 步长均匀性损失
        boundary_loss: 边界约束损失
    """
    # 1. 重建损失（MSE）
    recon_loss_raw = F.mse_loss(reconstructed, target, reduction='none')  # (batch, seq_len, input_dim)

    # 应用mask
    mask_expanded = mask.unsqueeze(-1).expand_as(recon_loss_raw)  # (batch, seq_len, input_dim)
    recon_loss = (recon_loss_raw * mask_expanded).sum() / (mask.sum() + 1e-8)

    # 2. KL散度损失：KL(q(z|x,c) || p(z|c)) ⭐改进⭐
    # 训练时：计算后验和先验之间的KL散度
    # 推理时：mu_posterior为None，不计算KL损失
    if mu_posterior is not None and logvar_posterior is not None:
        # KL(q || p) = 0.5 * sum(log(var_p/var_q) - 1 + var_q/var_p + (mu_q - mu_p)^2/var_p)
        # 使用对数方差形式：
        # = 0.5 * sum(logvar_p - logvar_q - 1 + exp(logvar_q)/exp(logvar_p) + (mu_q - mu_p)^2/exp(logvar_p))
        kl_loss = 0.5 * torch.sum(
            logvar_prior - logvar_posterior - 1
            + torch.exp(logvar_posterior) / (torch.exp(logvar_prior) + 1e-8)
            + (mu_posterior - mu_prior).pow(2) / (torch.exp(logvar_prior) + 1e-8)
        ) / mu_posterior.shape[0]
    else:
        # 推理模式：不计算KL损失
        kl_loss = torch.tensor(0.0, device=reconstructed.device)

    # 3. 终点损失：确保轨迹的最后一个点接近终点
    # 提取每个序列的最后一个有效点的坐标
    batch_size = reconstructed.shape[0]
    endpoint_loss = 0.0

    for b in range(batch_size):
        # 找到最后一个有效点的索引
        valid_indices = torch.where(mask[b] > 0)[0]
        if len(valid_indices) > 0:
            last_idx = valid_indices[-1]
            # 重建的最后位置 (current_x, current_y at indices 4, 5)
            reconstructed_endpoint = reconstructed[b, last_idx, 4:6]
            # 目标终点 (end_x, end_y at indices 2, 3)
            target_endpoint = target[b, last_idx, 2:4]
            # 计算终点误差
            endpoint_loss += F.mse_loss(reconstructed_endpoint, target_endpoint)

    endpoint_loss = endpoint_loss / batch_size

    # 4. 平滑度损失：鼓励相邻点之间的平滑过渡
    # 计算相邻坐标点之间的加速度变化
    smoothness_loss = 0.0

    if reconstructed.shape[1] > 2:  # 需要至少3个点来计算加速度
        # 提取坐标序列 (batch, seq_len, 2)
        coords = reconstructed[:, :, 4:6]  # current_x, current_y

        # 计算一阶差分（速度）
        velocity = coords[:, 1:, :] - coords[:, :-1, :]  # (batch, seq_len-1, 2)

        # 计算二阶差分（加速度）
        acceleration = velocity[:, 1:, :] - velocity[:, :-1, :]  # (batch, seq_len-2, 2)

        # 平滑度损失：加速度的平方和
        smoothness_loss = torch.mean(acceleration ** 2)

    # 5. 长度一致性损失：确保模型学习到轨迹长度信息
    length_loss = 0.0

    # 6. 分布损失：确保预测轨迹的点分布与真实轨迹一致
    distribution_loss = 0.0

    # 提取起点和终点
    # start_point和end_point在target中的索引0:2和2:4
    # 取第一个时间步的值（所有时间步的start_point和end_point应该是相同的）
    start_points = target[:, 0, 0:2]  # (batch, 2)
    end_points = target[:, 0, 2:4]    # (batch, 2)

    # 计算预测轨迹的分布
    pred_x_dist, pred_y_dist = compute_trajectory_distribution(
        reconstructed, start_points, end_points, mask, num_bins
    )

    # 计算目标轨迹的分布
    target_x_dist, target_y_dist = compute_trajectory_distribution(
        target, start_points, end_points, mask, num_bins
    )

    # 使用KL散度作为分布损失（交叉熵的变体）
    # KL(target || pred) = sum(target * log(target / pred))
    # 为了数值稳定性，添加小的epsilon
    epsilon = 1e-8
    pred_x_dist = pred_x_dist + epsilon
    pred_y_dist = pred_y_dist + epsilon
    target_x_dist = target_x_dist + epsilon
    target_y_dist = target_y_dist + epsilon

    # 归一化
    pred_x_dist = pred_x_dist / pred_x_dist.sum(dim=1, keepdim=True)
    pred_y_dist = pred_y_dist / pred_y_dist.sum(dim=1, keepdim=True)
    target_x_dist = target_x_dist / target_x_dist.sum(dim=1, keepdim=True)
    target_y_dist = target_y_dist / target_y_dist.sum(dim=1, keepdim=True)

    # 计算KL散度
    kl_x = torch.sum(target_x_dist * torch.log(target_x_dist / pred_x_dist), dim=1)
    kl_y = torch.sum(target_y_dist * torch.log(target_y_dist / pred_y_dist), dim=1)

    # 平均KL散度
    distribution_loss = torch.mean(kl_x + kl_y)

    # 7. 方向一致性损失（新增）
    direction_loss = compute_direction_loss(reconstructed, start_points, end_points, mask)

    # 8. 步长均匀性损失（新增）
    step_uniformity_loss = compute_step_uniformity_loss(reconstructed, mask)

    # 9. 边界约束损失（新增）
    boundary_loss = compute_boundary_loss(reconstructed, start_points, end_points, mask)

    # 总损失
    total_loss = (recon_loss +
                  kl_weight * kl_loss +
                  endpoint_weight * endpoint_loss +
                  smoothness_weight * smoothness_loss +
                  distribution_weight * distribution_loss +
                  direction_weight * direction_loss +               # 新增
                  step_uniformity_weight * step_uniformity_loss +   # 新增
                  boundary_weight * boundary_loss)                  # 新增

    return (total_loss, recon_loss, kl_loss, endpoint_loss, smoothness_loss, distribution_loss,
            direction_loss, step_uniformity_loss, boundary_loss)


if __name__ == '__main__':
    # 测试模型
    config = Config()
    model = LSTMCVAE(config)

    # 创建测试数据
    batch_size = 4
    seq_len = 50
    x = torch.randn(batch_size, seq_len, config.INPUT_DIM)
    start_point = torch.randn(batch_size, 2)
    end_point = torch.randn(batch_size, 2)

    # 测试前向传播
    reconstructed, mu, logvar = model(x, start_point, end_point)
    print(f"重建输出形状: {reconstructed.shape}")
    print(f"mu形状: {mu.shape}")
    print(f"logvar形状: {logvar.shape}")

    # 测试生成
    generated = model.generate(start_point, end_point, trajectory_length=30)
    print(f"生成轨迹形状: {generated.shape}")
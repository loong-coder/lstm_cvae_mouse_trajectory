"""
模型定义 - LSTM+CVAE组合模型和轨迹长度预测网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import Config


class TrajectoryLengthPredictor(nn.Module):
    """
    辅助网络：预测轨迹点的个数
    输入：起点和终点坐标
    输出：预测的轨迹长度

    优化版本：增加网络深度和宽度，添加距离特征
    """
    def __init__(self, input_dim=4, hidden_dim=128, max_length=500):
        super(TrajectoryLengthPredictor, self).__init__()

        # 增加隐藏层深度和宽度
        self.fc1 = nn.Linear(input_dim + 1, hidden_dim)  # +1 for distance feature
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, 1)  # 输出一个长度值

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.max_length = max_length

    def forward(self, start_point, end_point):
        """
        Args:
            start_point: (batch, 2) 起点坐标
            end_point: (batch, 2) 终点坐标
        Returns:
            predicted_length: (batch,) 预测的轨迹长度
        """
        # 计算起点到终点的欧氏距离作为额外特征
        distance = torch.sqrt(torch.sum((end_point - start_point)**2, dim=1, keepdim=True))

        # 拼接起点、终点和距离
        x = torch.cat([start_point, end_point, distance], dim=1)  # (batch, 5)

        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # (batch, 1)

        # 使用ReLU确保长度为正，并限制最大长度
        predicted_length = torch.clamp(x.squeeze(-1), min=1, max=self.max_length)

        return predicted_length


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
    主模型：LSTM + CVAE 组合
    用于生成鼠标移动轨迹
    """
    def __init__(self, config):
        super(LSTMCVAE, self).__init__()

        self.config = config

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=config.INPUT_DIM,
            hidden_size=config.LSTM_HIDDEN_DIM,
            num_layers=config.LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=config.LSTM_DROPOUT if config.LSTM_NUM_LAYERS > 1 else 0
        )

        # CVAE编码器
        self.encoder = CVAEEncoder(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.ENCODER_HIDDEN_DIM,
            latent_dim=config.LATENT_DIM
        )

        # CVAE解码器（输出11维：start_x, start_y, end_x, end_y, current_x, current_y, velocity, acceleration, sin_direction, cos_direction, distance）
        self.decoder = CVAEDecoder(
            latent_dim=config.LATENT_DIM,
            lstm_hidden_dim=config.LSTM_HIDDEN_DIM,
            output_dim=config.INPUT_DIM
        )

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧：从N(mu, var)中采样
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, start_point, end_point):
        """
        前向传播（训练模式）
        Args:
            x: (batch, seq_len, input_dim) 输入特征
            start_point: (batch, 2) 起点
            end_point: (batch, 2) 终点
        Returns:
            reconstructed: (batch, seq_len, input_dim) 重建的轨迹
            mu: (batch, latent_dim) 潜在变量均值
            logvar: (batch, latent_dim) 潜在变量对数方差
        """
        # LSTM编码
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_hidden_dim)

        # CVAE编码
        mu, logvar = self.encoder(x)  # (batch, latent_dim)

        # 重参数化采样
        z = self.reparameterize(mu, logvar)  # (batch, latent_dim)

        # CVAE解码
        reconstructed = self.decoder(z, lstm_out)  # (batch, seq_len, input_dim)

        return reconstructed, mu, logvar

    def generate(self, start_point, end_point, trajectory_length, num_samples=1):
        """
        生成模式：给定起点和终点，生成轨迹
        Args:
            start_point: (batch, 2) 起点坐标
            end_point: (batch, 2) 终点坐标
            trajectory_length: int 轨迹长度
            num_samples: int 生成样本数量
        Returns:
            generated_trajectory: (batch, trajectory_length, input_dim) 生成的轨迹
        """
        batch_size = start_point.shape[0]
        device = start_point.device

        # 从标准正态分布采样潜在变量
        z = torch.randn(batch_size, self.config.LATENT_DIM).to(device)

        # 创建初始输入序列（从起点开始）
        current_pos = start_point.clone()  # (batch, 2)

        generated_trajectory = []

        # 初始化LSTM隐藏状态
        h = torch.zeros(self.config.LSTM_NUM_LAYERS, batch_size, self.config.LSTM_HIDDEN_DIM).to(device)
        c = torch.zeros(self.config.LSTM_NUM_LAYERS, batch_size, self.config.LSTM_HIDDEN_DIM).to(device)

        for t in range(trajectory_length):
            # 构建当前时间步的输入特征
            # [start_x, start_y, end_x, end_y, current_x, current_y, velocity, acceleration, sin_direction, cos_direction, distance]

            # 计算到起点的距离（与训练数据一致）
            dx_to_start = current_pos[:, 0:1] - start_point[:, 0:1]
            dy_to_start = current_pos[:, 1:2] - start_point[:, 1:2]
            distance = torch.sqrt(dx_to_start**2 + dy_to_start**2 + 1e-8)

            # 计算到终点的距离和方向（用于引导生成）
            dx_to_end = end_point[:, 0:1] - current_pos[:, 0:1]
            dy_to_end = end_point[:, 1:2] - current_pos[:, 1:2]
            distance_to_end = torch.sqrt(dx_to_end**2 + dy_to_end**2 + 1e-8)

            if t == 0:
                # 第一步：使用从起点到终点的方向作为初始方向
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

                # 加速度（基于速度变化）
                prev_velocity = generated_trajectory[-1][:, 0, 6:7]  # 上一步的速度
                acceleration = (velocity - prev_velocity) * 100

            # 构建输入向量
            x_t = torch.cat([
                start_point,  # start_x, start_y
                end_point,    # end_x, end_y
                current_pos,  # current_x, current_y
                velocity,
                acceleration,
                sin_direction,
                cos_direction,
                distance  # 到起点的距离，与训练数据一致
            ], dim=1).unsqueeze(1)  # (batch, 1, input_dim)

            # LSTM前向
            lstm_out, (h, c) = self.lstm(x_t, (h, c))  # (batch, 1, lstm_hidden_dim)

            # CVAE解码
            output = self.decoder(z, lstm_out)  # (batch, 1, input_dim)

            generated_trajectory.append(output)

            # 更新当前位置为预测的下一个位置
            current_pos = output[:, 0, 4:6]  # 取current_x, current_y

        # 拼接所有时间步
        generated_trajectory = torch.cat(generated_trajectory, dim=1)  # (batch, trajectory_length, input_dim)

        return generated_trajectory


def compute_loss(reconstructed, target, mu, logvar, mask, kl_weight=0.001, endpoint_weight=1.0, smoothness_weight=0.1):
    """
    增强的CVAE损失：重建损失 + KL散度损失 + 终点损失 + 平滑度损失
    Args:
        reconstructed: (batch, seq_len, input_dim) 重建的轨迹
        target: (batch, seq_len, input_dim) 目标轨迹
        mu: (batch, latent_dim) 潜在变量均值
        logvar: (batch, latent_dim) 潜在变量对数方差
        mask: (batch, seq_len) 有效数据的mask
        kl_weight: KL散度的权重
        endpoint_weight: 终点损失的权重
        smoothness_weight: 平滑度损失的权重
    Returns:
        total_loss: 总损失
        recon_loss: 重建损失
        kl_loss: KL散度损失
        endpoint_loss: 终点损失
        smoothness_loss: 平滑度损失
    """
    # 1. 重建损失（MSE）
    recon_loss_raw = F.mse_loss(reconstructed, target, reduction='none')  # (batch, seq_len, input_dim)

    # 应用mask
    mask_expanded = mask.unsqueeze(-1).expand_as(recon_loss_raw)  # (batch, seq_len, input_dim)
    recon_loss = (recon_loss_raw * mask_expanded).sum() / (mask.sum() + 1e-8)

    # 2. KL散度损失
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]

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

    # 总损失
    total_loss = recon_loss + kl_weight * kl_loss + endpoint_weight * endpoint_loss + smoothness_weight * smoothness_loss

    return total_loss, recon_loss, kl_loss, endpoint_loss, smoothness_loss


if __name__ == '__main__':
    # 测试模型
    config = Config()
    model = LSTMCVAE(config)
    length_predictor = TrajectoryLengthPredictor()

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

    # 测试长度预测器
    predicted_length = length_predictor(start_point, end_point)
    print(f"预测长度: {predicted_length}")
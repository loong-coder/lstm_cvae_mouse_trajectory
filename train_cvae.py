import torch
from torch import nn
from torch.utils.data import DataLoader

from cvae_model import TrajectoryPredictorCVAE
from trajectory_dataset import TrajectoryDataset, MAX_TRAJECTORY_LENGTH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = TrajectoryDataset('mouse_trajectories_black_bg.csv', max_len=MAX_TRAJECTORY_LENGTH)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = TrajectoryPredictorCVAE(max_len=MAX_TRAJECTORY_LENGTH).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


class CVAELoss(nn.Module):
    """
    CVAE损失函数 = 重建损失 + KL散度损失 + 其他正则化损失

    重建损失：生成的轨迹与真实轨迹的差异
    KL散度：编码器分布 q(z|x,c) 与先验分布 p(z|c) 的差异
    """

    def __init__(self, position_weight=3.0, time_weight=0.5, direction_weight=3.0,
                 endpoint_weight=10.0, length_weight=2.0, speed_weight=1.5,
                 speed_deceleration_weight=2.0, direction_smoothness_weight=2.5,
                 trajectory_smoothness_weight=3.0, kl_weight=0.01):
        super().__init__()
        self.position_weight = position_weight
        self.time_weight = time_weight
        self.direction_weight = direction_weight
        self.endpoint_weight = endpoint_weight
        self.length_weight = length_weight
        self.speed_weight = speed_weight
        self.speed_deceleration_weight = speed_deceleration_weight
        self.direction_smoothness_weight = direction_smoothness_weight
        self.trajectory_smoothness_weight = trajectory_smoothness_weight
        self.kl_weight = kl_weight
        self.mse = nn.MSELoss(reduction='none')

    def kl_divergence(self, mu, logvar):
        """
        计算KL散度: KL(q(z|x,c) || p(z))
        假设先验 p(z) 是标准正态分布 N(0, I)

        KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        """
        # 限制logvar范围，防止exp溢出
        logvar = torch.clamp(logvar, min=-10, max=10)

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_loss.mean()

    def forward(self, reconstruction, target, mu, logvar, seq_lengths, predicted_length_ratio=None):
        """
        Args:
            reconstruction: (Batch, L_max, 5) - 重建的轨迹
            target: (Batch, L_max, 5) - 目标轨迹
            mu: (Batch, latent_dim) - 潜在变量均值
            logvar: (Batch, latent_dim) - 潜在变量对数方差
            seq_lengths: (Batch,) - 每个序列的实际长度
            predicted_length_ratio: (Batch, 1) - 预测的序列长度比例
        """
        batch_size = reconstruction.size(0)
        device = reconstruction.device
        max_len = reconstruction.size(1)

        # ========== 1. KL散度损失 ==========
        kl_loss = self.kl_divergence(mu, logvar)

        # ========== 2. 重建损失（与原来的损失函数类似）==========

        # 创建mask
        loss_mask = torch.zeros_like(reconstruction)
        for i, length in enumerate(seq_lengths):
            loss_mask[i, :length, :] = 1.0

        # 位置损失 (X, Y)
        position_loss = self.mse(reconstruction[:, :, :2], target[:, :, :2])
        position_loss = (position_loss * loss_mask[:, :, :2]).sum() / (loss_mask[:, :, :2].sum() + 1e-8)

        # 时间损失 (Time)
        time_loss = self.mse(reconstruction[:, :, 2], target[:, :, 2])
        time_loss = (time_loss * loss_mask[:, :, 2]).sum() / (loss_mask[:, :, 2].sum() + 1e-8)

        # 方向损失 (Direction)
        pred_direction = reconstruction[:, :, 3] * torch.pi
        true_direction = target[:, :, 3] * torch.pi
        direction_diff = pred_direction - true_direction
        direction_loss_raw = 1 - torch.cos(direction_diff)
        direction_loss = (direction_loss_raw * loss_mask[:, :, 3]).sum() / (loss_mask[:, :, 3].sum() + 1e-8)

        # 终点损失
        endpoint_loss = 0.0
        for i, length in enumerate(seq_lengths):
            if length > 0:
                pred_endpoint = reconstruction[i, length - 1, :2]
                true_endpoint = target[i, length - 1, :2]
                endpoint_loss += torch.mean((pred_endpoint - true_endpoint) ** 2)
        endpoint_loss = endpoint_loss / batch_size

        # 序列长度损失
        length_loss = 0.0
        if predicted_length_ratio is not None:
            for i, length in enumerate(seq_lengths):
                true_ratio = length / max_len
                pred_ratio = predicted_length_ratio[i, 0]
                length_loss += (true_ratio - pred_ratio) ** 2
            length_loss = length_loss / batch_size

        # 速度损失
        speed_loss = self.mse(reconstruction[:, :, 4], target[:, :, 4])
        speed_loss = (speed_loss * loss_mask[:, :, 4]).sum() / (loss_mask[:, :, 4].sum() + 1e-8)

        # 速度递减损失
        speed_deceleration_loss = 0.0
        for i, length in enumerate(seq_lengths):
            if length > 2:
                pred_speeds = reconstruction[i, :length, 4]
                speed_diff = pred_speeds[1:] - pred_speeds[:-1]
                deceleration_penalty = torch.relu(speed_diff).mean()
                start_speed_penalty = torch.relu(0.3 - pred_speeds[0])
                end_speed_penalty = torch.relu(pred_speeds[-1] - 0.2)
                speed_deceleration_loss += deceleration_penalty + 0.5 * (start_speed_penalty + end_speed_penalty)
        speed_deceleration_loss = speed_deceleration_loss / batch_size

        # 方向平滑度损失
        direction_smoothness_loss = 0.0
        for i, length in enumerate(seq_lengths):
            if length > 2:
                pred_directions = reconstruction[i, :length, 3] * torch.pi
                direction_changes = pred_directions[1:] - pred_directions[:-1]
                direction_changes = torch.atan2(torch.sin(direction_changes), torch.cos(direction_changes))
                smoothness_penalty = (direction_changes ** 2).mean()
                direction_smoothness_loss += smoothness_penalty
        direction_smoothness_loss = direction_smoothness_loss / batch_size

        # 轨迹平滑度损失（加速度约束）
        trajectory_smoothness_loss = 0.0
        for i, length in enumerate(seq_lengths):
            if length > 2:
                positions = reconstruction[i, :length, :2]
                velocities = positions[1:] - positions[:-1]
                accelerations = velocities[1:] - velocities[:-1]
                acceleration_magnitude = torch.sqrt((accelerations ** 2).sum(dim=1) + 1e-8)
                smoothness_penalty = acceleration_magnitude.mean()
                trajectory_smoothness_loss += smoothness_penalty
        trajectory_smoothness_loss = trajectory_smoothness_loss / batch_size

        # ========== 3. 总损失 ==========
        reconstruction_loss = (
                self.position_weight * position_loss +
                self.time_weight * time_loss +
                self.direction_weight * direction_loss +
                self.endpoint_weight * endpoint_loss +
                self.length_weight * length_loss +
                self.speed_weight * speed_loss +
                self.speed_deceleration_weight * speed_deceleration_loss +
                self.direction_smoothness_weight * direction_smoothness_loss +
                self.trajectory_smoothness_weight * trajectory_smoothness_loss
        )

        total_loss = reconstruction_loss + self.kl_weight * kl_loss

        return total_loss, {
            'total': total_loss.item(),
            'reconstruction': reconstruction_loss.item(),
            'kl': kl_loss.item(),
            'position': position_loss.item(),
            'time': time_loss.item(),
            'direction': direction_loss.item(),
            'endpoint': endpoint_loss.item(),
            'length': length_loss.item() if isinstance(length_loss, torch.Tensor) else length_loss,
            'speed': speed_loss.item(),
            'speed_decel': speed_deceleration_loss.item(),
            'dir_smooth': direction_smoothness_loss.item(),
            'traj_smooth': trajectory_smoothness_loss.item()
        }


criterion = CVAELoss(
    position_weight=3.0,
    time_weight=0.5,
    direction_weight=3.0,
    endpoint_weight=10.0,
    length_weight=2.0,
    speed_weight=1.5,
    speed_deceleration_weight=2.0,
    direction_smoothness_weight=2.5,
    trajectory_smoothness_weight=3.0,
    kl_weight=0.01  # KL散度权重（需要调节，太大会导致posterior collapse）
)


def train_model(model, dataloader, epochs=200, patience=15, min_delta=0.0001):
    """训练CVAE模型"""
    model.train()

    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    print(f"开始训练 CVAE 模型，最大 {epochs} 轮，早停耐心值: {patience}")
    print("=" * 80)

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_metrics = {
            'total': 0, 'reconstruction': 0, 'kl': 0,
            'position': 0, 'time': 0, 'direction': 0, 'endpoint': 0,
            'length': 0, 'speed': 0, 'speed_decel': 0,
            'dir_smooth': 0, 'traj_smooth': 0
        }
        num_batches = 0

        for task_context, target_sequence, seq_len in dataloader:
            task_context = task_context.to(device)
            target_sequence = target_sequence.to(device)

            optimizer.zero_grad()

            # CVAE前向传播
            reconstruction, mu, logvar, predicted_length_ratio = model(task_context, target_sequence)

            # 计算损失
            loss, metrics = criterion(reconstruction, target_sequence, mu, logvar, seq_len, predicted_length_ratio)

            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]
            num_batches += 1

        # 计算平均损失
        avg_loss = epoch_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}

        # 打印损失信息
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Total Loss: {avg_metrics['total']:.6f}")
        print(f"  - Reconstruction: {avg_metrics['reconstruction']:.6f}")
        print(f"  - KL Divergence: {avg_metrics['kl']:.6f}")
        print(f"  - Position: {avg_metrics['position']:.6f}")
        print(f"  - Endpoint: {avg_metrics['endpoint']:.6f}")
        print(f"  - Direction: {avg_metrics['direction']:.6f}")
        print(f"  - Dir Smooth: {avg_metrics['dir_smooth']:.6f}")
        print(f"  - Traj Smooth: {avg_metrics['traj_smooth']:.6f}")
        print(f"  - Speed: {avg_metrics['speed']:.6f}")
        print(f"  - Speed Decel: {avg_metrics['speed_decel']:.6f}")

        # 早停逻辑
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  ✓ 新的最佳损失！保存模型检查点")
        else:
            patience_counter += 1
            print(f"  → 损失未改善 ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print("\n" + "=" * 80)
                print(f"早停触发！连续 {patience} 个epoch损失未改善")
                print(f"最佳损失: {best_loss:.6f} (Epoch {epoch + 1 - patience})")
                print("恢复最佳模型...")
                model.load_state_dict(best_model_state)
                break

        print("-" * 80)

    # 训练完成
    if patience_counter < patience:
        print("\n" + "=" * 80)
        print(f"训练完成！达到最大轮数 {epochs}")
        print(f"最佳损失: {best_loss:.6f}")
        if best_model_state is not None:
            print("恢复最佳模型...")
            model.load_state_dict(best_model_state)

    print("=" * 80)

    # 保存模型
    print("\n保存模型到 'cvae_trajectory_predictor.pth'...")
    torch.save(model.state_dict(), 'cvae_trajectory_predictor.pth')
    print("CVAE 模型保存完成！")


if __name__ == '__main__':
    train_model(
        model,
        dataloader,
        epochs=250,
        patience=15,
        min_delta=0.0001
    )
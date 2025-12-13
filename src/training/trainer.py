"""
训练脚本 - 训练LSTM-CVAE模型和轨迹长度预测器
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from tqdm import tqdm

from config.config import Config
from src.data.dataset import create_data_loaders
from src.models.lstm_cvae import LSTMCVAE, compute_loss


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 创建模型保存目录
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)

        # 创建数据加载器
        print("加载数据...")
        self.train_loader, self.val_loader, self.norm_stats = create_data_loaders(
            config.DATA_FILE,
            batch_size=config.BATCH_SIZE,
            train_split=config.TRAIN_SPLIT
        )

        # 初始化模型（长度预测器已集成在模型中）
        print("初始化模型...")
        self.model = LSTMCVAE(config).to(self.device)

        # 优化器（包含所有参数，包括集成的长度预测器）
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # TensorBoard
        self.writer = SummaryWriter('runs/lstm_cvae_trajectory')

        # 最佳模型跟踪
        self.best_val_loss = float('inf')

        # Early Stopping 跟踪
        self.patience_counter = 0  # 记录连续多少个epoch没有改善
        self.best_epoch = 0  # 记录最佳模型是在哪个epoch

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_endpoint_loss = 0
        total_smoothness_loss = 0
        total_distribution_loss = 0
        total_direction_loss = 0
        total_step_uniformity_loss = 0
        total_boundary_loss = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.NUM_EPOCHS} [Train]')

        for batch_idx, batch in enumerate(pbar):
            # 将数据移到设备
            features = batch['features'].to(self.device)
            start_point = batch['start_point'].to(self.device)
            end_point = batch['end_point'].to(self.device)
            lengths = batch['length'].to(self.device)
            mask = batch['mask'].to(self.device)

            # 训练（模型现在包含所有部分，统一训练）
            self.optimizer.zero_grad()

            # 前向传播（模型现在返回5个值：后验均值/方差，先验均值/方差）
            reconstructed, mu_posterior, logvar_posterior, mu_prior, logvar_prior = self.model(
                features, start_point, end_point
            )

            # 计算损失（添加几何约束损失）
            # 训练时传入后验和先验分布参数
            (loss, recon_loss, kl_loss, endpoint_loss, smoothness_loss, distribution_loss,
             direction_loss, step_uniformity_loss, boundary_loss) = compute_loss(
                reconstructed, features,
                mu_posterior, logvar_posterior, mu_prior, logvar_prior,  # 传入后验和先验
                mask,
                kl_weight=self.config.KL_WEIGHT,
                endpoint_weight=self.config.ENDPOINT_WEIGHT,
                smoothness_weight=self.config.SMOOTHNESS_WEIGHT,
                distribution_weight=self.config.DISTRIBUTION_WEIGHT,
                direction_weight=self.config.DIRECTION_WEIGHT,
                step_uniformity_weight=self.config.STEP_UNIFORMITY_WEIGHT,
                boundary_weight=self.config.BOUNDARY_WEIGHT,
                num_bins=10
            )

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_endpoint_loss += endpoint_loss.item()
            total_smoothness_loss += smoothness_loss.item()
            total_distribution_loss += distribution_loss.item()
            total_direction_loss += direction_loss.item()
            total_step_uniformity_loss += step_uniformity_loss.item()
            total_boundary_loss += boundary_loss.item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'ep': f'{endpoint_loss.item():.4f}',
                'dir': f'{direction_loss.item():.4f}'
            })

        # 平均损失
        avg_loss = total_loss / len(self.train_loader)
        avg_recon_loss = total_recon_loss / len(self.train_loader)
        avg_kl_loss = total_kl_loss / len(self.train_loader)
        avg_endpoint_loss = total_endpoint_loss / len(self.train_loader)
        avg_smoothness_loss = total_smoothness_loss / len(self.train_loader)
        avg_distribution_loss = total_distribution_loss / len(self.train_loader)
        avg_direction_loss = total_direction_loss / len(self.train_loader)
        avg_step_uniformity_loss = total_step_uniformity_loss / len(self.train_loader)
        avg_boundary_loss = total_boundary_loss / len(self.train_loader)

        return (avg_loss, avg_recon_loss, avg_kl_loss, avg_endpoint_loss, avg_smoothness_loss,
                avg_distribution_loss, avg_direction_loss, avg_step_uniformity_loss, avg_boundary_loss)

    def validate(self, epoch):
        """验证"""
        self.model.eval()

        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_endpoint_loss = 0
        total_smoothness_loss = 0
        total_distribution_loss = 0
        total_direction_loss = 0
        total_step_uniformity_loss = 0
        total_boundary_loss = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.config.NUM_EPOCHS} [Val]')

            for batch in pbar:
                features = batch['features'].to(self.device)
                start_point = batch['start_point'].to(self.device)
                end_point = batch['end_point'].to(self.device)
                lengths = batch['length'].to(self.device)
                mask = batch['mask'].to(self.device)

                # 前向传播（模型现在返回5个值）
                reconstructed, mu_posterior, logvar_posterior, mu_prior, logvar_prior = self.model(
                    features, start_point, end_point
                )

                # 计算损失（添加几何约束损失）
                (loss, recon_loss, kl_loss, endpoint_loss, smoothness_loss, distribution_loss,
                 direction_loss, step_uniformity_loss, boundary_loss) = compute_loss(
                    reconstructed, features,
                    mu_posterior, logvar_posterior, mu_prior, logvar_prior,  # 传入后验和先验
                    mask,
                    kl_weight=self.config.KL_WEIGHT,
                    endpoint_weight=self.config.ENDPOINT_WEIGHT,
                    smoothness_weight=self.config.SMOOTHNESS_WEIGHT,
                    distribution_weight=self.config.DISTRIBUTION_WEIGHT,
                    direction_weight=self.config.DIRECTION_WEIGHT,
                    step_uniformity_weight=self.config.STEP_UNIFORMITY_WEIGHT,
                    boundary_weight=self.config.BOUNDARY_WEIGHT,
                    num_bins=10
                )

                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                total_endpoint_loss += endpoint_loss.item()
                total_smoothness_loss += smoothness_loss.item()
                total_distribution_loss += distribution_loss.item()
                total_direction_loss += direction_loss.item()
                total_step_uniformity_loss += step_uniformity_loss.item()
                total_boundary_loss += boundary_loss.item()

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}'
                })

        # 平均损失
        avg_loss = total_loss / len(self.val_loader)
        avg_recon_loss = total_recon_loss / len(self.val_loader)
        avg_kl_loss = total_kl_loss / len(self.val_loader)
        avg_endpoint_loss = total_endpoint_loss / len(self.val_loader)
        avg_smoothness_loss = total_smoothness_loss / len(self.val_loader)
        avg_distribution_loss = total_distribution_loss / len(self.val_loader)
        avg_direction_loss = total_direction_loss / len(self.val_loader)
        avg_step_uniformity_loss = total_step_uniformity_loss / len(self.val_loader)
        avg_boundary_loss = total_boundary_loss / len(self.val_loader)

        return (avg_loss, avg_recon_loss, avg_kl_loss, avg_endpoint_loss, avg_smoothness_loss,
                avg_distribution_loss, avg_direction_loss, avg_step_uniformity_loss, avg_boundary_loss)

    def train(self):
        """完整训练流程（带 Early Stopping）"""
        print(f"\n开始训练，最多 {self.config.NUM_EPOCHS} 个epochs...")
        print(f"Early Stopping: 连续 {self.config.EARLY_STOPPING_PATIENCE} 个epoch无改善将自动停止\n")

        for epoch in range(self.config.NUM_EPOCHS):
            # 训练
            (train_loss, train_recon, train_kl, train_ep, train_smooth, train_dist,
             train_dir, train_step_uni, train_boundary) = self.train_epoch(epoch)

            # 验证
            (val_loss, val_recon, val_kl, val_ep, val_smooth, val_dist,
             val_dir, val_step_uni, val_boundary) = self.validate(epoch)

            # 记录到TensorBoard
            self.writer.add_scalars('Loss/Total', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            self.writer.add_scalars('Loss/Reconstruction', {
                'train': train_recon,
                'val': val_recon
            }, epoch)
            self.writer.add_scalars('Loss/KL', {
                'train': train_kl,
                'val': val_kl
            }, epoch)
            self.writer.add_scalars('Loss/Endpoint', {
                'train': train_ep,
                'val': val_ep
            }, epoch)
            self.writer.add_scalars('Loss/Smoothness', {
                'train': train_smooth,
                'val': val_smooth
            }, epoch)
            self.writer.add_scalars('Loss/Distribution', {
                'train': train_dist,
                'val': val_dist
            }, epoch)
            self.writer.add_scalars('Loss/Direction', {
                'train': train_dir,
                'val': val_dir
            }, epoch)
            self.writer.add_scalars('Loss/StepUniformity', {
                'train': train_step_uni,
                'val': val_step_uni
            }, epoch)
            self.writer.add_scalars('Loss/Boundary', {
                'train': train_boundary,
                'val': val_boundary
            }, epoch)

            # 学习率调整
            self.scheduler.step(val_loss)

            # 打印统计
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print(f"Train - Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}, "
                  f"Endpoint: {train_ep:.4f}, Dir: {train_dir:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Recon: {val_recon:.4f}, KL: {val_kl:.4f}, "
                  f"Endpoint: {val_ep:.4f}, Dir: {val_dir:.4f}")

            # Early Stopping 检查
            improvement = self.best_val_loss - val_loss

            if improvement > self.config.MIN_DELTA:
                # 有显著改善，保存检查点
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(epoch)
                print(f"✓ 保存最佳模型 (val_loss: {val_loss:.4f}, 改善: {improvement:.6f})")
            else:
                # 没有显著改善
                self.patience_counter += 1
                print(f"⚠ 验证损失未改善 ({self.patience_counter}/{self.config.EARLY_STOPPING_PATIENCE})")

                # 检查是否触发 Early Stopping
                if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"\n{'='*60}")
                    print(f"Early Stopping 触发！")
                    print(f"连续 {self.config.EARLY_STOPPING_PATIENCE} 个epoch验证损失未改善")
                    print(f"最佳模型在 Epoch {self.best_epoch+1}，验证损失: {self.best_val_loss:.4f}")
                    print(f"{'='*60}\n")
                    break

        # 训练结束总结
        print("\n" + "="*60)
        print("训练完成！")
        print(f"总训练轮数: {epoch+1}")
        print(f"最佳模型: Epoch {self.best_epoch+1}, 验证损失: {self.best_val_loss:.4f}")
        print("="*60 + "\n")

        self.writer.close()

    def save_checkpoint(self, epoch):
        """保存模型检查点（单一检查点文件，不断覆盖）"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'norm_stats': self.norm_stats,
            'config': self.config
        }

        # 只保存一个检查点文件，不断覆盖
        torch.save(checkpoint, self.config.BEST_MODEL_PATH)

    def load_checkpoint(self, path):
        """加载模型检查点"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"加载检查点成功: {path}")
        return checkpoint['epoch']


if __name__ == '__main__':
    # 配置
    config = Config()

    # 创建训练器
    trainer = Trainer(config)

    # 开始训练
    trainer.train()
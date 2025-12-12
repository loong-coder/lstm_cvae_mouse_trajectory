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
from src.models.lstm_cvae import LSTMCVAE, TrajectoryLengthPredictor, compute_loss


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

        # 初始化模型
        print("初始化模型...")
        self.model = LSTMCVAE(config).to(self.device)
        self.length_predictor = TrajectoryLengthPredictor(
            input_dim=4,
            hidden_dim=config.LENGTH_PREDICTOR_HIDDEN_DIM,
            max_length=config.MAX_TRAJECTORY_LENGTH
        ).to(self.device)

        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.length_optimizer = optim.Adam(self.length_predictor.parameters(), lr=config.LEARNING_RATE)

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.length_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.length_optimizer, mode='min', factor=0.5, patience=5
        )

        # TensorBoard
        self.writer = SummaryWriter('runs/lstm_cvae_trajectory')

        # 最佳模型跟踪
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        self.length_predictor.train()

        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_length_loss = 0
        total_endpoint_loss = 0
        total_smoothness_loss = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.NUM_EPOCHS} [Train]')

        for batch_idx, batch in enumerate(pbar):
            # 将数据移到设备
            features = batch['features'].to(self.device)
            start_point = batch['start_point'].to(self.device)
            end_point = batch['end_point'].to(self.device)
            lengths = batch['length'].to(self.device)
            mask = batch['mask'].to(self.device)

            # ===== 训练主模型 (LSTM-CVAE) =====
            self.optimizer.zero_grad()

            # 前向传播
            reconstructed, mu, logvar = self.model(features, start_point, end_point)

            # 计算损失（使用增强损失）
            loss, recon_loss, kl_loss, endpoint_loss, smoothness_loss = compute_loss(
                reconstructed, features, mu, logvar, mask,
                kl_weight=self.config.KL_WEIGHT,
                endpoint_weight=1.0,
                smoothness_weight=0.1
            )

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # ===== 训练轨迹长度预测器 =====
            self.length_optimizer.zero_grad()

            # 预测长度
            predicted_lengths = self.length_predictor(start_point, end_point)

            # 计算长度预测损失（MSE）
            length_loss = nn.MSELoss()(predicted_lengths, lengths.float())

            # 反向传播
            length_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.length_predictor.parameters(), max_norm=1.0)
            self.length_optimizer.step()

            # 统计
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_endpoint_loss += endpoint_loss.item()
            total_smoothness_loss += smoothness_loss.item()
            total_length_loss += length_loss.item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'ep': f'{endpoint_loss.item():.4f}',
                'len': f'{length_loss.item():.4f}'
            })

        # 平均损失
        avg_loss = total_loss / len(self.train_loader)
        avg_recon_loss = total_recon_loss / len(self.train_loader)
        avg_kl_loss = total_kl_loss / len(self.train_loader)
        avg_endpoint_loss = total_endpoint_loss / len(self.train_loader)
        avg_smoothness_loss = total_smoothness_loss / len(self.train_loader)
        avg_length_loss = total_length_loss / len(self.train_loader)

        return avg_loss, avg_recon_loss, avg_kl_loss, avg_endpoint_loss, avg_smoothness_loss, avg_length_loss

    def validate(self, epoch):
        """验证"""
        self.model.eval()
        self.length_predictor.eval()

        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_endpoint_loss = 0
        total_smoothness_loss = 0
        total_length_loss = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.config.NUM_EPOCHS} [Val]')

            for batch in pbar:
                features = batch['features'].to(self.device)
                start_point = batch['start_point'].to(self.device)
                end_point = batch['end_point'].to(self.device)
                lengths = batch['length'].to(self.device)
                mask = batch['mask'].to(self.device)

                # 主模型（使用增强损失）
                reconstructed, mu, logvar = self.model(features, start_point, end_point)
                loss, recon_loss, kl_loss, endpoint_loss, smoothness_loss = compute_loss(
                    reconstructed, features, mu, logvar, mask,
                    kl_weight=self.config.KL_WEIGHT,
                    endpoint_weight=1.0,
                    smoothness_weight=0.1
                )

                # 长度预测器
                predicted_lengths = self.length_predictor(start_point, end_point)
                length_loss = nn.MSELoss()(predicted_lengths, lengths.float())

                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                total_endpoint_loss += endpoint_loss.item()
                total_smoothness_loss += smoothness_loss.item()
                total_length_loss += length_loss.item()

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'len_loss': f'{length_loss.item():.4f}'
                })

        # 平均损失
        avg_loss = total_loss / len(self.val_loader)
        avg_recon_loss = total_recon_loss / len(self.val_loader)
        avg_kl_loss = total_kl_loss / len(self.val_loader)
        avg_endpoint_loss = total_endpoint_loss / len(self.val_loader)
        avg_smoothness_loss = total_smoothness_loss / len(self.val_loader)
        avg_length_loss = total_length_loss / len(self.val_loader)

        return avg_loss, avg_recon_loss, avg_kl_loss, avg_endpoint_loss, avg_smoothness_loss, avg_length_loss

    def train(self):
        """完整训练流程"""
        print(f"\n开始训练 {self.config.NUM_EPOCHS} 个epochs...\n")

        for epoch in range(self.config.NUM_EPOCHS):
            # 训练
            train_loss, train_recon, train_kl, train_ep, train_smooth, train_len = self.train_epoch(epoch)

            # 验证
            val_loss, val_recon, val_kl, val_ep, val_smooth, val_len = self.validate(epoch)

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
            self.writer.add_scalars('Loss/Length', {
                'train': train_len,
                'val': val_len
            }, epoch)

            # 学习率调整
            self.scheduler.step(val_loss)
            self.length_scheduler.step(val_len)

            # 打印统计
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print(f"Train - Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}, Endpoint: {train_ep:.4f}, Length: {train_len:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Recon: {val_recon:.4f}, KL: {val_kl:.4f}, Endpoint: {val_ep:.4f}, Length: {val_len:.4f}")

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                print(f"✓ 保存最佳模型 (val_loss: {val_loss:.4f})")

            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)

        print("\n训练完成！")
        self.writer.close()

    def save_checkpoint(self, epoch, is_best=False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'length_predictor_state_dict': self.length_predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'length_optimizer_state_dict': self.length_optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'norm_stats': self.norm_stats,
            'config': self.config
        }

        if is_best:
            torch.save(checkpoint, self.config.BEST_MODEL_PATH)
        else:
            path = os.path.join(self.config.MODEL_SAVE_PATH, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """加载模型检查点"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.length_predictor.load_state_dict(checkpoint['length_predictor_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.length_optimizer.load_state_dict(checkpoint['length_optimizer_state_dict'])
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
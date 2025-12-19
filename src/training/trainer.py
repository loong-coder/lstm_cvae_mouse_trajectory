import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm

from src.data.dataset import TrajectoryDataset
from src.models.lstm_cvae import TrajectoryCVAE, elbo_loss


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 创建模型保存目录
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)

        # 创建数据集
        print("加载数据...")
        dataset = TrajectoryDataset(config.DATA_FILE, seq_len=config.SEQ_LEN)

        # 划分训练集和验证集
        train_size = int(len(dataset) * config.TRAIN_SPLIT)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

        print(f"训练集大小: {train_size}, 验证集大小: {val_size}")

        # 初始化模型
        print("初始化模型...")
        self.model = TrajectoryCVAE(
            feat_dim=config.FEAT_DIM,
            cond_dim=config.COND_DIM,
            latent_dim=config.LATENT_DIM,
            hidden_dim=config.HIDDEN_DIM
        ).to(self.device)

        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # 最佳模型跟踪
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.NUM_EPOCHS} [Train]')

        for batch_idx, (x, c) in enumerate(pbar):
            # 将数据移到设备
            x = x.to(self.device)  # [Batch, Seq, 5]
            c = c.to(self.device)  # [Batch, 4]

            # 前向传播
            self.optimizer.zero_grad()
            recon_x, mu, logvar = self.model(x, c, teacher_forcing_ratio=self.config.TEACHER_FORCING_RATIO)

            # 计算损失
            loss = elbo_loss(recon_x, x, mu, logvar, kld_weight=self.config.KLD_WEIGHT)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 统计
            total_loss += loss.item()

            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 平均损失
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self, epoch):
        """验证"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.config.NUM_EPOCHS} [Val]')

            for x, c in pbar:
                x = x.to(self.device)
                c = c.to(self.device)

                # 前向传播
                recon_x, mu, logvar = self.model(x, c, teacher_forcing_ratio=0.0)  # 验证时不使用teacher forcing

                # 计算损失
                loss = elbo_loss(recon_x, x, mu, logvar, kld_weight=self.config.KLD_WEIGHT)

                total_loss += loss.item()

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 平均损失
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def train(self):
        """完整训练流程"""
        print(f"\n开始训练，共 {self.config.NUM_EPOCHS} 个epochs...")

        for epoch in range(self.config.NUM_EPOCHS):
            # 训练
            train_loss = self.train_epoch(epoch)

            # 验证
            val_loss = self.validate(epoch)

            # 学习率调整
            self.scheduler.step(val_loss)

            # 打印统计
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch)
                print(f"✓ 保存最佳模型 (val_loss: {val_loss:.4f})")

        print("\n训练完成！")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")

    def save_checkpoint(self, epoch):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        torch.save(checkpoint, self.config.BEST_MODEL_PATH)

    def load_checkpoint(self, path):
        """加载模型检查点"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"加载检查点成功: {path}")
        return checkpoint['epoch']
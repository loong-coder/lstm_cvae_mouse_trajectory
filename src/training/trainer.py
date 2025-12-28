import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm

from src.data.dataset import TrajectoryDataset
from src.models.lstm_cvae import TrajectoryCVAE, elbo_loss, velocity_profile_loss, minimum_jerk_loss, seq_len_prediction_loss


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 创建模型保存目录
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)

        # 创建数据集
        print("加载数据...")
        dataset = TrajectoryDataset(
            config.DATA_FILE,
            max_seq_len=config.MAX_SEQ_LEN,
            min_seq_len=config.MIN_SEQ_LEN,
            pixels_per_step=config.PIXELS_PER_STEP
        )

        # 划分训练集和验证集
        train_size = int(len(dataset) * config.TRAIN_SPLIT)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # 创建数据加载器（使用自定义collate_fn处理变长序列）
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            collate_fn=TrajectoryDataset.collate_fn
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            collate_fn=TrajectoryDataset.collate_fn
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

        # 早停机制
        self.early_stopping_patience = config.EARLY_STOPPING_PATIENCE
        self.min_delta = config.MIN_DELTA
        self.patience_counter = 0

    def get_teacher_forcing_ratio(self, epoch):
        """计算Teacher Forcing比率，从1.0线性衰减到0.0"""
        return max(0.0, 1.0 - (epoch / self.config.NUM_EPOCHS))

    def get_kld_weight(self, epoch):
        """计算KLD权重，使用annealing策略从0逐步增加到目标值"""
        if epoch >= self.config.KLD_ANNEALING_EPOCHS:
            return self.config.KLD_WEIGHT_END
        else:
            # 线性增加
            progress = epoch / self.config.KLD_ANNEALING_EPOCHS
            return self.config.KLD_WEIGHT_START + (self.config.KLD_WEIGHT_END - self.config.KLD_WEIGHT_START) * progress

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_mse = 0
        total_kld = 0

        # 计算当前epoch的teacher forcing比率和KLD权重
        tf_ratio = self.get_teacher_forcing_ratio(epoch)
        kld_weight = self.get_kld_weight(epoch)

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.NUM_EPOCHS} [Train] TF={tf_ratio:.2f} KLD_W={kld_weight:.3f}')

        for batch_idx, (x, c, seq_lengths, masks) in enumerate(pbar):
            # 将数据移到设备
            x = x.to(self.device)  # [Batch, max_seq_len, 6]
            c = c.to(self.device)  # [Batch, 4]
            seq_lengths = seq_lengths.to(self.device)  # [Batch]
            masks = masks.to(self.device)  # [Batch, max_seq_len]

            # 前向传播（使用衰减的teacher forcing比率）
            self.optimizer.zero_grad()
            recon_x, mu, logvar, pred_seq_len = self.model(x, c, teacher_forcing_ratio=tf_ratio)

            # 计算基础ELBO损失（MSE + KLD）
            base_loss = elbo_loss(recon_x, x, mu, logvar, kld_weight=kld_weight)

            # 添加物理约束损失
            # 1. 速度曲线约束 - 强制"慢-快-慢"模式（Fitts's Law）
            vel_loss = velocity_profile_loss(recon_x)

            # 2. 最小加加速度约束 - 确保运动平滑自然
            jerk_loss = minimum_jerk_loss(recon_x)

            # 3. 序列长度预测损失 - 让模型学习从起点终点预测应该生成多少个点
            seq_len_loss = seq_len_prediction_loss(pred_seq_len, seq_lengths)

            # 总损失 = ELBO + 速度约束 + 加加速度约束 + 序列长度预测
            loss = base_loss + \
                   self.config.VELOCITY_PROFILE_WEIGHT * vel_loss + \
                   self.config.MINIMUM_JERK_WEIGHT * jerk_loss + \
                   self.config.SEQ_LEN_PRED_WEIGHT * seq_len_loss

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 统计
            total_loss += loss.item()

            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'seq_len_loss': f'{seq_len_loss.item():.4f}'})

        # 平均损失
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self, epoch):
        """验证"""
        self.model.eval()
        total_loss = 0

        # 使用与训练相同的KLD权重
        kld_weight = self.get_kld_weight(epoch)

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.config.NUM_EPOCHS} [Val]')

            for x, c, seq_lengths, masks in pbar:
                x = x.to(self.device)
                c = c.to(self.device)
                seq_lengths = seq_lengths.to(self.device)
                masks = masks.to(self.device)

                # 前向传播
                recon_x, mu, logvar, pred_seq_len = self.model(x, c, teacher_forcing_ratio=0.0)  # 验证时不使用teacher forcing

                # 计算基础ELBO损失（MSE + KLD）
                base_loss = elbo_loss(recon_x, x, mu, logvar, kld_weight=kld_weight)

                # 添加物理约束损失
                vel_loss = velocity_profile_loss(recon_x)
                jerk_loss = minimum_jerk_loss(recon_x)

                # 序列长度预测损失
                seq_len_loss = seq_len_prediction_loss(pred_seq_len, seq_lengths)

                # 总损失（与训练时相同）
                loss = base_loss + \
                       self.config.VELOCITY_PROFILE_WEIGHT * vel_loss + \
                       self.config.MINIMUM_JERK_WEIGHT * jerk_loss + \
                       self.config.SEQ_LEN_PRED_WEIGHT * seq_len_loss

                total_loss += loss.item()

                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'seq_len_loss': f'{seq_len_loss.item():.4f}'})

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
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

            # 早停检查
            if val_loss < self.best_val_loss - self.min_delta:
                # 验证损失有显著改善
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch)
                print(f"✓ 保存最佳模型 (val_loss: {val_loss:.4f})")
            else:
                # 验证损失没有改善
                self.patience_counter += 1
                print(f"⚠ 验证损失未改善 ({self.patience_counter}/{self.early_stopping_patience})")

                if self.patience_counter >= self.early_stopping_patience:
                    print(f"\n早停触发！验证损失已经 {self.early_stopping_patience} 个epoch未改善")
                    print(f"最佳验证损失: {self.best_val_loss:.4f}")
                    break

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
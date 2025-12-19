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
        # 将隐藏状态映射回特征空间（5维：x, y, v, a, d）
        self.fc_out = nn.Linear(hidden_dim, feat_dim)

    def reparameterize(self, mu, logvar):
        """重参数化技巧：使随机采样过程可导"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c, teacher_forcing_ratio=0.5):
        """
        x: 真实轨迹 [Batch, Seq, 5]
        c: 起终点条件 [Batch, 4]
        teacher_forcing_ratio: 训练时使用真值作为下一步输入的概率
        """
        batch_size, seq_len, _ = x.size()

        # 1. Encoder：提取轨迹特征
        # c_expand 形状：[Batch, Seq, 4]，为了与 x 拼接
        c_expand = c.unsqueeze(1).expand(-1, seq_len, -1)
        enc_in = torch.cat([x, c_expand], dim=-1) # [Batch, Seq, 9]
        _, (h_n, _) = self.enc_lstm(enc_in)

        mu = self.fc_mu(h_n[-1])       # [Batch, latent_dim]
        logvar = self.fc_logvar(h_n[-1])
        z = self.reparameterize(mu, logvar) # [Batch, latent_dim]

        # 2. Decoder：逐步生成轨迹
        # 准备 z 和 c 的拼接体，它们在每个时间步都是固定的输入
        z_c = torch.cat([z, c], dim=-1).unsqueeze(1) # [Batch, 1, latent+cond]

        outputs = []
        # 第一步的输入通常是真实轨迹的起始点
        curr_input = x[:, 0, :].unsqueeze(1) # [Batch, 1, 5]
        h_d, c_d = None, None # 初始化 Decoder 的 LSTM 状态

        for t in range(seq_len):
            # 将 [当前点, z, c] 拼接
            dec_in = torch.cat([curr_input, z_c], dim=-1) # [Batch, 1, 5+16+4]
            out, (h_d, c_d) = self.dec_lstm(dec_in, (h_d, c_d) if h_d is not None else None)
            pred = self.fc_out(out) # 预测当前步或下一步的值
            outputs.append(pred)

            # Teacher Forcing：决定下一步是用预测值还是真值
            is_teacher = random.random() < teacher_forcing_ratio
            if is_teacher and t < seq_len - 1:
                curr_input = x[:, t+1, :].unsqueeze(1) # 下一步用真实点
            else:
                curr_input = pred # 下一步用刚才预测的点

        return torch.cat(outputs, dim=1), mu, logvar

    def inference(self, c, seq_len=20):
        """推理模式：完全不需要真实轨迹，只需起点终点"""
        batch_size = c.size(0)
        device = c.device

        # 从标准正态分布中随机采样"风格"
        z = torch.randn(batch_size, self.latent_dim).to(device)
        z_c = torch.cat([z, c], dim=-1).unsqueeze(1)

        # 假设从全0或者从条件中的起点开始（需根据归一化后的 c 提取）
        curr_input = torch.zeros(batch_size, 1, 5).to(device)

        outputs = []
        h_d, c_d = None, None
        for _ in range(seq_len):
            dec_in = torch.cat([curr_input, z_c], dim=-1)
            out, (h_d, c_d) = self.dec_lstm(dec_in, (h_d, c_d) if h_d is not None else None)
            pred = self.fc_out(out)
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
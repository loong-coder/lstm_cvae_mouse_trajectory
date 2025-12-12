#!/usr/bin/env python
"""
训练脚本
运行: python scripts/train.py
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from src.training.trainer import Trainer


def main():
    """主函数"""
    # 配置
    config = Config()

    # 创建训练器
    trainer = Trainer(config)

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
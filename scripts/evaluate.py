#!/usr/bin/env python
"""
评估脚本
运行: python scripts/evaluate.py
"""
import sys
import os
import tkinter as tk

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from src.gui.evaluator import TrajectoryEvaluationGUI


def main():
    """主函数"""
    config = Config()

    # 检查模型文件是否存在
    if not os.path.exists(config.BEST_MODEL_PATH):
        print(f"错误: 找不到模型文件 {config.BEST_MODEL_PATH}")
        print("请先运行 python scripts/train.py 训练模型")
        return

    root = tk.Tk()
    app = TrajectoryEvaluationGUI(root, config.BEST_MODEL_PATH)
    root.mainloop()


if __name__ == '__main__':
    main()
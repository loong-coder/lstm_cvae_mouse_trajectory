#!/usr/bin/env python
"""
数据收集脚本
运行: python scripts/collect_data.py
"""
import sys
import os
import tkinter as tk

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gui.collector import MouseTrajectoryCollector


def main():
    """主函数"""
    root = tk.Tk()
    app = MouseTrajectoryCollector(root)
    root.mainloop()


if __name__ == '__main__':
    main()
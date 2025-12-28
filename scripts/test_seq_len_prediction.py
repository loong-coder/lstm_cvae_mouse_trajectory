"""
验证序列长度预测功能

这个脚本测试模型是否能够根据起点终点距离正确预测应该生成的序列长度
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from config.config import Config
from src.models.lstm_cvae import TrajectoryCVAE

def test_seq_len_prediction():
    """测试序列长度预测功能"""

    print("=" * 60)
    print("序列长度预测功能测试")
    print("=" * 60)

    # 初始化模型
    config = Config()
    model = TrajectoryCVAE(
        feat_dim=config.FEAT_DIM,
        cond_dim=config.COND_DIM,
        latent_dim=config.LATENT_DIM,
        hidden_dim=config.HIDDEN_DIM
    )
    model.eval()

    # 测试用例：不同距离的起点终点对
    test_cases = [
        # (start_x, start_y, end_x, end_y, 预期长度范围描述)
        (100, 100, 200, 100, "短距离 (100像素)"),
        (100, 100, 400, 100, "中等距离 (300像素)"),
        (100, 100, 700, 100, "长距离 (600像素)"),
        (100, 100, 1000, 100, "超长距离 (900像素)"),
        (100, 100, 100, 200, "短距离-垂直 (100像素)"),
        (100, 100, 400, 400, "中等距离-对角线 (~424像素)"),
    ]

    print("\n测试案例：")
    print("-" * 60)

    for start_x, start_y, end_x, end_y, description in test_cases:
        # 计算实际距离
        distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)

        # 使用公式计算预期长度
        expected_len = TrajectoryCVAE.calculate_seq_len(
            (start_x, start_y),
            (end_x, end_y),
            pixels_per_step=config.PIXELS_PER_STEP
        )

        # 创建条件张量（注意：这里使用的是未归一化的坐标）
        # 实际训练时会经过归一化，但这里只是测试预测网络的结构
        c = torch.FloatTensor([[start_x, start_y, end_x, end_y]])

        # 使用模型预测（注意：未训练的模型预测会是随机的）
        with torch.no_grad():
            pred_len = model.predict_seq_len(c)
            pred_len_value = pred_len[0].item()

        print(f"\n案例: {description}")
        print(f"  起点: ({start_x}, {start_y})")
        print(f"  终点: ({end_x}, {end_y})")
        print(f"  距离: {distance:.1f} 像素")
        print(f"  公式计算长度: {expected_len} 个点")
        print(f"  模型预测长度: {pred_len_value:.1f} 个点 (未训练，随机初始化)")
        print(f"  说明: 训练后，模型预测应该接近公式计算值")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print("\n关键点:")
    print("1. 公式计算：distance / pixels_per_step，限制在 [5, 50] 范围")
    print("2. 模型预测：seq_len_predictor 网络从条件向量预测长度")
    print("3. 训练目标：让模型预测值接近公式计算值（通过 seq_len_prediction_loss）")
    print("4. 推理时：使用 model.inference(c, use_predicted_len=True) 让模型自动决定长度")
    print("\n优势：")
    print("  - 模型可以学习比公式更复杂的长度模式")
    print("  - 可以根据数据分布自适应调整")
    print("  - 避免了硬编码的公式依赖")

def test_inference_with_predicted_len():
    """测试使用预测长度进行推理"""

    print("\n" + "=" * 60)
    print("使用预测长度进行推理测试")
    print("=" * 60)

    config = Config()
    model = TrajectoryCVAE(
        feat_dim=config.FEAT_DIM,
        cond_dim=config.COND_DIM,
        latent_dim=config.LATENT_DIM,
        hidden_dim=config.HIDDEN_DIM
    )
    model.eval()

    # 测试条件
    c = torch.FloatTensor([[100, 100, 400, 100]])  # 中等距离

    print("\n测试场景: 从 (100, 100) 到 (400, 100)")
    print(f"距离: 300 像素")
    print(f"公式计算长度: {TrajectoryCVAE.calculate_seq_len((100, 100), (400, 100), 30)} 个点")

    # 1. 固定长度推理
    print("\n方式1: 固定长度推理 (seq_len=20)")
    trajectory_fixed = model.inference(c, seq_len=20, use_predicted_len=False)
    print(f"  生成轨迹形状: {trajectory_fixed.shape}")

    # 2. 使用模型预测长度推理
    print("\n方式2: 使用模型预测长度推理")
    trajectory_predicted = model.inference(c, use_predicted_len=True)
    print(f"  生成轨迹形状: {trajectory_predicted.shape}")
    print(f"  (注意: 未训练的模型预测长度是随机的)")

    # 3. 使用公式计算长度推理
    print("\n方式3: 使用公式计算长度推理")
    formula_len = TrajectoryCVAE.calculate_seq_len((100, 100), (400, 100), 30)
    trajectory_formula = model.inference(c, seq_len=formula_len, use_predicted_len=False)
    print(f"  生成轨迹形状: {trajectory_formula.shape}")

    print("\n" + "=" * 60)
    print("推理测试完成！")
    print("=" * 60)
    print("\n训练后，方式2的模型预测长度应该接近方式3的公式计算长度")

if __name__ == '__main__':
    # 测试序列长度预测
    test_seq_len_prediction()

    # 测试推理
    test_inference_with_predicted_len()
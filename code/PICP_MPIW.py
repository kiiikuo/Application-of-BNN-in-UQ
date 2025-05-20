# PICP_MPIW。py

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def calculate_PICP(preds_mean, preds_var, true_values, z=1.96):
    preds_lower = preds_mean - z * torch.sqrt(preds_var)
    preds_upper = preds_mean + z * torch.sqrt(preds_var)
    coverage = (true_values >= preds_lower) & (true_values <= preds_upper)
    return coverage.float().mean().item()


def calculate_MPIW(preds_var, z=1.96):
    preds_width = 2 * z * torch.sqrt(preds_var)
    return preds_width.mean().item()


def find_oracle_MPIW(preds_mean, preds_var, true_values, target_PICP=0.95, max_iter=100, tolerance=1e-4,
                     z_initial=1.96):
    z = z_initial
    lower_bound = 0.1 * z
    upper_bound = 5 * z

    for _ in range(max_iter):
        PICP = calculate_PICP(preds_mean, preds_var, true_values, z)
        if abs(PICP - (1 - target_PICP)) < tolerance:
            return calculate_MPIW(preds_var, z)
        z *= 1.1 if PICP < (1 - target_PICP) else 0.9
        z = max(lower_bound, min(z, upper_bound))

    return calculate_MPIW(preds_var, z)


def visualize_PICP_MPIW(preds_mean, preds_var, true_values, alpha_range=(0.05, 0.3), step=0.05):
    """
    可视化不同 alpha 值对应的 PICP、MPIW 和 Oracle MPIW，使用双纵轴以适应不同指标尺度

    参数:
    - preds_mean: 预测均值，shape: (N, H, W)
    - preds_var: 预测方差，shape: (N, H, W)
    - true_values: 真实值，shape: (N, H, W)
    - alpha_range: alpha 值范围 (如 0.05 到 0.3)
    - step: alpha 增加的步长
    """
    alpha_values = np.arange(alpha_range[0], alpha_range[1] + step, step)

    picp_values = []
    mpiw_values = []
    oracle_mpiw_values = []

    for alpha in alpha_values:
        z = norm.ppf(1 - alpha / 2)
        picp = calculate_PICP(preds_mean, preds_var, true_values, z)
        mpiw = calculate_MPIW(preds_var, z)
        oracle_mpiw = find_oracle_MPIW(preds_mean, preds_var, true_values, target_PICP=1-alpha, z_initial=z)

        picp_values.append(picp)
        mpiw_values.append(mpiw)
        oracle_mpiw_values.append(oracle_mpiw)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 左轴：PICP
    ax1.plot(alpha_values, picp_values, color='blue', marker='o', label='PICP')
    ax1.plot(alpha_values, 1 - alpha_values, color='gray', linestyle=':', label='Theoretical PICP')
    ax1.set_xlabel('Alpha (1 - Confidence Level)')
    ax1.set_ylabel('PICP', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim([0, 1])

    # 右轴：MPIW 和 Oracle MPIW
    ax2 = ax1.twinx()
    ax2.plot(alpha_values, mpiw_values, color='green', marker='x', linestyle='-', label='MPIW')
    ax2.plot(alpha_values, oracle_mpiw_values, color='red', marker='s', linestyle='--', label='Oracle MPIW')
    ax2.set_ylabel('MPIW / Oracle MPIW', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # 联合图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.title('PICP, MPIW, Oracle MPIW vs Alpha')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

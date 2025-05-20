from Data_Loader import load_data
from Model import MC_CNN
from Train import train_model
from Predict import predict_with_uncertainty
from Visualize import visualize_prediction
from PICP_MPIW import visualize_PICP_MPIW

import torch


def normalize_tensor(x, mean=None, std=None):
    if mean is None:
        mean = x.mean()
    if std is None:
        std = x.std()
    return (x - mean) / (std + 1e-8), mean, std


def normalize_output(Y):
    Y_min = Y.min()
    Y_max = Y.max()
    Y_norm = (Y - Y_min) / (Y_max - Y_min + 1e-8)
    return Y_norm, Y_min, Y_max


def denormalize_output(Y_norm, Y_min, Y_max):
    return Y_norm * (Y_max - Y_min + 1e-8) + Y_min


# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载数据（多个滑动窗口样本）
train_inputs, train_targets = load_data(n=1, num_samples=50, seed=42)
val_inputs, val_targets = load_data(n=1, num_samples=5, seed=24)

# 数据迁移到设备
train_inputs = train_inputs.to(device)      # (50, 3, 320, 416)
train_targets = train_targets.to(device)    # (50, 320, 416)
val_inputs = val_inputs.to(device)
val_targets = val_targets.to(device)

# 区域图归一化（按通道整体归一化）
train_inputs, region_mean, region_std = normalize_tensor(train_inputs)
val_inputs = (val_inputs - region_mean) / (region_std + 1e-8)

# 输出归一化
train_targets, output_min, output_max = normalize_output(train_targets)
val_targets = (val_targets - output_min) / (output_max - output_min + 1e-8)

# 初始化模型
model = MC_CNN(dropout_rate=0.05).to(device)

# 训练模型
model = train_model(model,
                    region_input=train_inputs,
                    target=train_targets,
                    val_region_input=val_inputs,
                    val_target=val_targets,
                    device=device)

# 预测
mean, var = predict_with_uncertainty(model, val_inputs, mc_samples=100, device=device)

# 反归一化
mean = denormalize_output(mean, output_min, output_max)
var = var * (output_max - output_min + 1e-8) ** 2
val_targets = denormalize_output(val_targets, output_min, output_max)

# 可视化 PICP, MPIW
visualize_PICP_MPIW(mean, var, val_targets, alpha_range=(0.05, 0.3), step=0.05)

# 可视化预测结果（例如 idx=0）
visualize_prediction(mean, var, val_targets, idx=0)

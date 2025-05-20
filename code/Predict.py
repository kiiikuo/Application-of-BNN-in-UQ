# Predict.py

import torch


def predict_with_uncertainty(model, region_inputs, mc_samples=100, device='cpu'):
    """
    对输入进行 MC Dropout 推理，仅使用区域图作为输入，返回均值和方差
    """
    model.eval()  # 设置为评估模式（BN/Dropout）
    model.to(device)

    region_inputs = region_inputs.to(device)
    region_inputs = torch.nan_to_num(region_inputs, nan=0.0)

    preds = []

    with torch.no_grad():
        for _ in range(mc_samples):
            model.train()  # 启用 Dropout
            out = model(region_inputs)
            preds.append(out.unsqueeze(0))

    preds_stack = torch.cat(preds, dim=0)  # (mc_samples, batch, H, W)
    mean = preds_stack.mean(dim=0)
    var = preds_stack.var(dim=0)

    return mean, var

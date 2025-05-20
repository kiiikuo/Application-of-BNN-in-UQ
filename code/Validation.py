# Validation.py

import torch
import torch.nn.functional as F


def evaluate_model(model, region_val, Y_val, mc_samples=1, return_std=False):

    model.eval()
    region_val = region_val.to(torch.float32)
    Y_val = Y_val.to(torch.float32)

    preds_mc = []

    with torch.no_grad():
        for _ in range(mc_samples):
            if mc_samples > 1:
                model.train()
            else:
                model.eval()
            preds = model(region_val)
            preds_mc.append(preds.unsqueeze(0))

    preds_stack = torch.cat(preds_mc, dim=0)  # (mc_samples, N, 320, 416)
    preds_mean = preds_stack.mean(dim=0)

    valid_mask = ~torch.isnan(Y_val)
    mse = F.mse_loss(preds_mean[valid_mask], Y_val[valid_mask]).item()
    mae = F.l1_loss(preds_mean[valid_mask], Y_val[valid_mask]).item()

    if return_std:
        preds_std = preds_stack.std(dim=0)
        return mse, mae, preds_mean, preds_std
    else:
        return mse, mae, preds_mean

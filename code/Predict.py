# Predict.py

import torch


def predict_with_uncertainty(model, region_inputs, mc_samples=100, device='cpu'):

    model.eval()
    model.to(device)

    region_inputs = region_inputs.to(device)
    region_inputs = torch.nan_to_num(region_inputs, nan=0.0)

    preds = []

    with torch.no_grad():
        for _ in range(mc_samples):
            model.train()
            out = model(region_inputs)
            preds.append(out.unsqueeze(0))

    preds_stack = torch.cat(preds, dim=0)
    mean = preds_stack.mean(dim=0)
    var = preds_stack.var(dim=0)

    return mean, var

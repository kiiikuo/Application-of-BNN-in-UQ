# Train.py

import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def adaptive_smooth_l1_loss(preds, targets, factor=0.1, min_beta=1e-3, max_beta=1.0):
    with torch.no_grad():
        error = torch.abs(preds - targets)
        beta = factor * error.mean()
        beta = beta.clamp(min=min_beta, max=max_beta)

    diff = preds - targets
    abs_diff = diff.abs()
    loss = torch.where(
        abs_diff < beta,
        0.5 * diff ** 2 / beta,
        abs_diff - 0.5 * beta
    )
    return loss.mean()


def train_model(model, region_input, target, val_region_input, val_target,
                device='cpu', num_epochs=2000, lr=1e-3, weight_decay=1e-4,
                patience=200, mc_samples=100, save_path="best_model.pt",
                batch_size=32):
    
    from Validation import evaluate_model

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    tag = 0.995

    train_dataset = TensorDataset(region_input, target)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_region_input = val_region_input.to(device)
    val_target = val_target.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_region_input, batch_target in train_loader:
            batch_region_input = batch_region_input.to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad()
            preds = model(batch_region_input)

            loss = adaptive_smooth_l1_loss(preds, batch_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item() * batch_region_input.size(0)

        epoch_loss /= len(train_dataset)

        val_mse, val_mae, _ = evaluate_model(model, val_region_input, val_target, mc_samples)

        if (epoch + 1) % 100 == 0:
            print(f"[Epoch {epoch + 1:03d}] Train Loss: {epoch_loss:.4f} | Val MSE: {val_mse:.4f}, MAE: {val_mae:.4f}")

        if val_mse < best_val_loss * tag:
            best_val_loss = val_mse
            patience_counter = 0
            best_model_state = model.state_dict()
            torch.save(best_model_state, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

    if best_model_state is not None:
        model.load_state_dict(torch.load(save_path))
    else:
        print("Warning: No best model state was saved.")

    return model

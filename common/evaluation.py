#!/usr/bin/env python3
import torch

def evaluate_model(model, loader, device="cpu"):
    model.eval()
    loss_fn = torch.nn.MSELoss()
    total_loss = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb[:, -1, :].to(device)
            pred = model(xb)
            total_loss += loss_fn(pred, yb).item()
    return total_loss / len(loader)
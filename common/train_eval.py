import os
import torch
import torch.nn as nn
from common.utils import logging, save_checkpoint

def train_model(model, dir_path, model_name, train_loader, val_loader=None, 
                epochs=20, lr=0.001, device="cpu", stop_threshold=0.01):
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            # xb, yb = xb.to(device), yb[:, -1, :].to(device)
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optim.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    # xb, yb = xb.to(device), yb[:, -1, :].to(device)
                    xb, yb = xb.to(device), yb.to(device)
                    val_loss += loss_fn(model(xb), yb).item()
            val_loss /= len(val_loader)

        if val_loss is not None:
            logging.info(f"{model_name} | Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(dir_path, f"{model_name}_best.pth"))
                logging.info(f"New best {model_name} saved (Val Loss {val_loss:.4f})")
        else:
            logging.info(f"{model_name} | Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f}")

        if (val_loss and val_loss <= stop_threshold) or (not val_loss and train_loss <= stop_threshold):
            logging.info(f"{model_name} stopped early at epoch {epoch+1}")
            break

        save_checkpoint(model, dir_path, model_name, epoch)
    
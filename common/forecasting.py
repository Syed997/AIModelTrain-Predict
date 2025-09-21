#!/usr/bin/env python3
import numpy as np
import torch

def recursive_forecast(model, last_input, horizon, device='cpu'):
    model.eval()
    preds, inp = [], torch.from_numpy(last_input).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(horizon):
            out = model(inp)
            preds.append(out.cpu().numpy()[0])
            inp = torch.cat([inp[:, 1:, :], out.unsqueeze(1)], dim=1)
    return np.array(preds)

# anomaly_detect.py — PRODUCTION-GRADE, CALIBRATED, SAFE

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from lstm_autoencoder.model.lstm_autoencoder import LSTMAutoencoder
import os

from pathlib import Path
from datetime import datetime

# ==================== ANOMALY STORAGE ====================
ANOMALY_DIR = Path("anomalies")
(ANOMALY_DIR / "anomaly").mkdir(parents=True, exist_ok=True)
(ANOMALY_DIR / "critical").mkdir(parents=True, exist_ok=True)

MAX_SAVE_FILES = 100   # safety guard

# ==================== CONFIG ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/autoencoder_best.pth"
SEQ_LEN = 25
BATCH_SIZE = 64
DATA_PATH = "data/kafka_load.csv"

ANOMALY_PERCENTILE = 99.5
CRITICAL_PERCENTILE = 99.9
MAX_ANOMALY_RATE = 0.30

# ==================== MODEL ====================
print(f"[INFO] Loading model → {MODEL_PATH}")
model = LSTMAutoencoder(hidden_size=64, latent_size=16).to(DEVICE)

with torch.no_grad():
    dummy = torch.zeros(1, SEQ_LEN, 61, device=DEVICE)
    _ = model(dummy)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("[INFO] Model ready\n")

# ==================== DATA LOADING ====================
def load_data(file_path):
    print(f"[INFO] Loading data → {file_path}")
    df = pd.read_csv(file_path)

    drop_cols = [
        "timestamp", "topic", "trace_id", "span_id", "parent_span_id",
        "attributes_code.filepath", "attributes_http.url",
        "attributes_url.full", "attributes_user_agent.original",
        "name", "body", "exception_message",
        "exception_stacktrace", "exception_type",
        "resource_attributes_service.name"
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    numeric_df = df.select_dtypes(include=[np.number]).copy()

    feature_names = np.load("models/feature_names.npy", allow_pickle=True)

    for col in feature_names:
        if col not in numeric_df.columns:
            numeric_df[col] = 0.0

    numeric_df = numeric_df[feature_names]

    for col in numeric_df.columns:
        if any(k in col.lower() for k in ["count", "total", "size", "byte", "request", "event"]):
            numeric_df[col] = numeric_df[col].diff().fillna(0).clip(lower=0)

    numeric_df = numeric_df.iloc[1:].fillna(0).reset_index(drop=True)

    if len(numeric_df) < SEQ_LEN:
        raise ValueError("Not enough data for inference")

    data = numeric_df.values.astype(np.float32)

    scaler_min = np.load("models/scaler_min.npy")
    scaler_max = np.load("models/scaler_max.npy")

    scale = scaler_max - scaler_min
    scale[scale == 0] = 1.0

    data_norm = (data - scaler_min) / scale
    data_norm = np.clip(data_norm, 0, 1)

    print(f"[INFO] Prepared {len(data_norm)} samples × {data_norm.shape[1]} features")
    return data_norm

# ==================== DATASET ====================
class AutoencoderDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx):
        seq = self.data[idx: idx + self.seq_len]
        return torch.tensor(seq), torch.tensor(seq)

# ==================== INFERENCE ====================
data_norm = load_data(DATA_PATH)
dataset = AutoencoderDataset(data_norm, SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

print("\n[INFO] Running inference...")

errors = []
windows = []   # <<< NEW (store each window)

with torch.no_grad():
    for batch_x, _ in loader:
        batch_x = batch_x.to(DEVICE)
        recon = model(batch_x)
        batch_err = torch.mean((batch_x - recon) ** 2, dim=[1, 2]).cpu().numpy()

        for i in range(len(batch_err)):
            errors.append(batch_err[i])
            windows.append(batch_x[i].cpu().numpy())

errors = np.array(errors)

# ==================== CALIBRATED THRESHOLDS ====================
p_main = np.percentile(errors, ANOMALY_PERCENTILE)
p_critical = np.percentile(errors, CRITICAL_PERCENTILE)

anomalies = errors > p_main
critical = errors > p_critical
anomaly_rate = anomalies.mean()

# ==================== SAVE ANOMALY WINDOWS ====================
timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
saved = 0

for idx, (err, window) in enumerate(zip(errors, windows)):
    if err <= p_main:
        continue

    level = "critical" if err > p_critical else "anomaly"
    fname = f"window_{idx:06d}_err_{err:.6f}_{timestamp}.txt"
    fpath = ANOMALY_DIR / level / fname

    with open(fpath, "w") as f:
        f.write(f"Window index     : {idx}\n")
        f.write(f"Reconstruction MSE: {err:.8f}\n")
        f.write(f"Main threshold   : {p_main:.8f}\n")
        f.write(f"Critical threshold: {p_critical:.8f}\n")
        f.write(f"Severity         : {level}\n")
        f.write(f"Sequence length  : {SEQ_LEN}\n")
        f.write(f"Num features     : {window.shape[1]}\n\n")
        f.write("=== Window Data (normalized) ===\n")
        np.savetxt(f, window, fmt="%.6f")

    saved += 1
    if saved >= MAX_SAVE_FILES:
        break

# ==================== REPORT ====================
print("\n" + "=" * 80)
print("PRODUCTION ANOMALY SUMMARY")
print("=" * 80)
print(f"Total windows        : {len(errors)}")
print(f"Anomaly threshold    : {ANOMALY_PERCENTILE}th percentile")
print(f"Critical threshold   : {CRITICAL_PERCENTILE}th percentile")
print(f"Detected anomalies   : {anomalies.sum()} ({anomaly_rate * 100:.2f}%)")
print(f"Critical anomalies   : {critical.sum()}")

if anomaly_rate > MAX_ANOMALY_RATE:
    print("\n[WARNING] DISTRIBUTION DRIFT — alerts suppressed")
elif critical.any():
    print("\n[ALERT] CRITICAL INCIDENT DETECTED")
elif anomalies.any():
    print("\n[ALERT] Anomalous behavior detected")
else:
    print("\n[OK] System operating normally")

print("=" * 80)

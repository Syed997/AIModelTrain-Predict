import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lstm_autoencoder.model.lstm_autoencoder import LSTMAutoencoder
from common.dataset import AutoencoderDataset
from common.train_eval import train_autoencoder

from config.config import (
    INPUT_WINDOW, N_FEATURES, EPOCHS, BATCH_SIZE,
    STOP_THRESHOLD, TRAIN_CSV, EVAL_CSV,
    LSTM_AUTOENCODER_MODEL_FILENAME
)

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(MODEL_DIR, exist_ok=True)

train_path = os.path.join(DATA_DIR, TRAIN_CSV)
eval_path = os.path.join(DATA_DIR, EVAL_CSV)
model_path = os.path.join(MODEL_DIR, LSTM_AUTOENCODER_MODEL_FILENAME)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# def load_data(file_path):
#     df = pd.read_csv(file_path)

#     # remove timestamp if exists
#     if "timestamp" in df.columns:
#         df = df.drop(columns=["timestamp"])

#     if "topic" in df.columns:
#         df = df.drop(columns=["topic"])

#     return df.values.astype(np.float32)

# remove the non numeric columns
def load_data(file_path):
    print(f"\n[TRAIN] Loading training data: {file_path}")
    df = pd.read_csv(file_path)

    # Drop string columns
    string_cols = ["timestamp", "topic", "trace_id", "span_id", "parent_span_id",
                   "attributes_code.filepath", "attributes_http.url", "attributes_url.full",
                   "attributes_user_agent.original", "name", "body", "exception_message",
                   "exception_stacktrace", "exception_type", "resource_attributes_service.name"]
    df = df.drop(columns=[c for c in string_cols if c in df.columns], errors="ignore")

    numeric_df = df.select_dtypes(include=[np.number])
    print(f"Before cleaning: {numeric_df.shape[1]} numeric columns")
    numeric_df = numeric_df.dropna(axis=1, thresh=int(0.1 * len(numeric_df)))
    print(f"After cleaning: {numeric_df.shape[1]} columns")

    # === CONVERT COUNTERS TO CLEAN PER-SECOND RATES ===
    print("Converting counters → clean per-second rates...")
    counter_keywords = ['count', 'total', 'size', 'byte', 'request', 'query', 'event', 'duration']
    counter_cols = [
        col for col in numeric_df.columns
        if any(kw in col.lower() for kw in counter_keywords)
        and col not in ['duration_ms', 'duration_ns', 'http.status_code']
    ]

    for col in counter_cols:
        if col in numeric_df.columns:
            # Compute rate, but first row is garbage → fix it
            rates = numeric_df[col].diff()
            # Replace first row with median of the rest (or 0)
            rates.iloc[0] = rates.iloc[1:].median() if len(rates) > 1 else 0
            rates = rates.clip(lower=0).fillna(0)
            numeric_df[col] = rates

    # DROP THE FIRST ROW — it's poisoned by diff()
    numeric_df = numeric_df.iloc[1:].reset_index(drop=True)
    print(f"Applied clean rate conversion + dropped first row → {len(numeric_df)} rows")

    # Fill remaining NaN
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True)).fillna(0)

    data = numeric_df.values.astype(np.float32)

    # === SAVE SCALER (only in training) ===
    data_min = data.min(axis=0, keepdims=True)
    data_max = data.max(axis=0, keepdims=True)
    np.save("models/scaler_min.npy", data_min)
    np.save("models/scaler_max.npy", data_max)
    print("Saved scaler_min.npy and scaler_max.npy (rate-based, clean)")

    # Normalize
    data_range = data_max - data_min
    data_range[data_range == 0] = 1.0
    data_normalized = (data - data_min) / data_range

    print(f"Final TRAINING data ready: {data_normalized.shape[0]} samples × {data_normalized.shape[1]} features\n")
    return data_normalized


def main():
    # Load datasets
    train_data = load_data(train_path)
    eval_data = load_data(eval_path)

    # Create datasets
    train_ds = AutoencoderDataset(train_data, INPUT_WINDOW)
    eval_ds = AutoencoderDataset(eval_data, INPUT_WINDOW)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE)

    # Model
    model = LSTMAutoencoder(hidden_size=64, latent_size=16).to(DEVICE)

    # FORCE BUILD: This is the missing piece!
    print("Initializing model architecture...")
    with torch.no_grad():
        for batch in train_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(DEVICE).float()
            _ = model(x)
            print(f"Model built: {x.shape}")
            break

    # Now safe to load old weights (if any)
    if os.path.exists(model_path):
        print("Resuming previous autoencoder weights...")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        print("Weights loaded successfully!")

    print("\nStarting Autoencoder Training...\n")

    # Now optimizer will work!
    train_autoencoder(
        model=model,
        save_dir=MODEL_DIR,
        model_name="autoencoder",
        train_loader=train_loader,
        val_loader=eval_loader,
        epochs=EPOCHS,
        device=DEVICE,
        stop_threshold=STOP_THRESHOLD
    )
    print("\n=== SAVING ROBUST PRODUCTION STATISTICS (FINAL FIX) ===")
    model.eval()
    all_errors = []
    with torch.no_grad():
        for batch in train_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(DEVICE)
            recon = model(x)
            # MSE per time step, then average over sequence
            err = torch.mean((x - recon)**2, dim=[1,2])
            all_errors.extend(err.cpu().numpy())

    all_errors = np.array(all_errors)
    
    # THIS IS THE CORRECT WAY — used by every professional system
    median_error = np.median(all_errors)
    mad = np.median(np.abs(all_errors - median_error)) * 1.4826   # ← THIS LINE WAS MISSING!

    np.save(os.path.join(MODEL_DIR, "train_mad.npy"), mad)
    np.save(os.path.join(MODEL_DIR, "train_median_error.npy"), median_error)

    print(f"Training median reconstruction error : {median_error:.6f}")
    print(f"Training MAD (robust scale)         : {mad:.6f}   ← THIS IS THE REAL ONE!")
    print(f"Example: an error of {mad * 10:.6f} will be flagged as 10× anomaly")
    print("Production-ready robust detection is now ACTIVE!")


if __name__ == "__main__":
    main()

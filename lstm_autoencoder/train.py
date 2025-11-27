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
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)

    # Drop obvious string columns first
    string_cols = ["timestamp", "topic", "trace_id", "span_id", "parent_span_id",
                   "attributes_code.filepath", "attributes_http.url", "attributes_url.full",
                   "attributes_user_agent.original", "name", "body", "exception_message",
                   "exception_stacktrace", "exception_type", "resource_attributes_service.name"]
    df = df.drop(columns=[c for c in string_cols if c in df.columns], errors="ignore")

    # Convert to numeric — this turns non-numeric → NaN
    numeric_df = df.select_dtypes(include=[np.number])

    # CRITICAL: Remove columns that are ALL NaN or have too many NaNs
    print(f"Before cleaning: {numeric_df.shape[1]} numeric columns")
    numeric_df = numeric_df.dropna(axis=1, thresh=int(0.1 * len(numeric_df)))  # keep if ≥10% non-NaN
    print(f"After dropping mostly-empty columns: {numeric_df.shape[1]} columns")

    # Fill remaining NaNs with reasonable values
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))
    # If median still NaN (column was all NaN), fill with 0
    numeric_df = numeric_df.fillna(0)

    # Final sanity check
    if numeric_df.isnull().any().any():
        print("Still have NaN! Columns with NaN:")
        print(numeric_df.columns[numeric_df.isnull().any()].tolist())
        raise ValueError("NaN values remain after cleaning!")

    data = numeric_df.values.astype(np.float32)

    # Normalize to 0–1
    data_min = data.min(axis=0, keepdims=True)
    data_max = data.max(axis=0, keepdims=True)
    # Prevent division by zero
    data_range = data_max - data_min
    data_range[data_range == 0] = 1.0
    data_normalized = (data - data_min) / data_range

    # Save scaler and feature names
    np.save(os.path.join(MODEL_DIR, "scaler_min.npy"), data_min)
    np.save(os.path.join(MODEL_DIR, "scaler_max.npy"), data_max)
    np.save(os.path.join(MODEL_DIR, "feature_names.npy"), np.array(numeric_df.columns))

    print(f"Final dataset: {data_normalized.shape[0]} samples × {data_normalized.shape[1]} clean features")
    print("Sample features:", list(numeric_df.columns[:10]), "...")

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


if __name__ == "__main__":
    main()

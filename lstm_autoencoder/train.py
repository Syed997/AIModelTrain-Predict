import os
import numpy as np
import pandas as pd
import torch
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


def load_data(file_path):
    df = pd.read_csv(file_path)

    # remove timestamp if exists
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    return df.values.astype(np.float32)


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
    model = LSTMAutoencoder(n_features=N_FEATURES)

    # Resume training if model exists
    if os.path.exists(model_path):
        print("Resuming previous autoencoder weights...")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    print("\nStarting Autoencoder Training...\n")

    # Train
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

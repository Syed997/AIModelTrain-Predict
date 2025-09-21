import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from tcn.model.tcn_model import TCNForecast
from common.dataset import TimeSeriesDataset
from common.train_eval import train_model
from config.config import (
    INPUT_WINDOW, FORECAST_HORIZON, N_FEATURES, EPOCHS,
    THRESHOLDS, STOP_THRESHOLD, BATCH_SIZE, TRAIN_CSV, TIMESERIES_MODE,
    TCN_MODEL_FILENAME, TRAIN_SPLIT, EVAL_CSV
    )
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
saved_model_path = os.path.join(MODEL_DIR, TCN_MODEL_FILENAME)
train_data = os.path.join(DATA_DIR, TRAIN_CSV)
eval_data = os.path.join(DATA_DIR, EVAL_CSV)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# INPUT_WINDOW, FORECAST_HORIZON, N_FEATURES, EPOCHS = 300, 300, 5, 5

def main():
    # os.makedirs("models", exist_ok=True)
    train = pd.read_csv(train_data).drop(columns=["timestamp"]).values.astype(np.float32)
    eval_dataset = pd.read_csv(eval_data).drop(columns=["timestamp"]).values.astype(np.float32)

    dataset = TimeSeriesDataset(train, INPUT_WINDOW, FORECAST_HORIZON, TIMESERIES_MODE)
    dataset_eval = TimeSeriesDataset(eval_dataset, INPUT_WINDOW, FORECAST_HORIZON, TIMESERIES_MODE)
    # n_train = int(TRAIN_SPLIT * len(dataset))
    # train_ds, val_ds = random_split(dataset, [n_train, len(dataset)-n_train])
    train_ds, val_ds = dataset, dataset_eval

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = TCNForecast(N_FEATURES)
    model_path = saved_model_path
    if os.path.exists(model_path):
        print("Resuming TCN...")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    print("Training TCN...")
    train_model(model, MODEL_DIR, "tcn", train_loader, val_loader, epochs=EPOCHS, device=DEVICE, stop_threshold=STOP_THRESHOLD)

if __name__ == "__main__":
    main()

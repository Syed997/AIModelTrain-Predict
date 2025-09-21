import os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from lstm.model.lstm_model import LSTMForecast
from common.dataset import TimeSeriesDataset
from common.train_eval import train_model
# from dotenv import load_dotenv
# dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.env")
# load_dotenv(dotenv_path)
from config.config import (
    INPUT_WINDOW, FORECAST_HORIZON, N_FEATURES, EPOCHS,
    THRESHOLDS, STOP_THRESHOLD, BATCH_SIZE, EVAL_CSV, TIMESERIES_MODE,
    LSTM_MODEL_FILENAME, TRAIN_CSV, TRAIN_SPLIT
    )

# INPUT_WINDOW = int(os.getenv("INPUT_WINDOW"))
# FORECAST_HORIZON = int(os.getenv("FORECAST_HORIZON"))
# N_FEATURES = int(os.getenv("N_FEATURES"))
# EPOCHS = int(os.getenv("EPOCHS"))

# THRESHOLDS = list(map(int, os.getenv("THRESHOLDS").split(',')))

# STOP_THRESHOLD = float(os.getenv("STOP_THRESHOLD"))
# BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
# EVAL_CSV = os.getenv("EVAL_CSV")
# TIMESERIES_MODE = os.getenv("TIMESERIES_MODE")
# LSTM_MODEL_FILENAME = os.getenv("LSTM_MODEL_FILENAME")
# TRAIN_CSV = os.getenv("TRAIN_CSV")
# TRAIN_SPLIT = float(os.getenv("TRAIN_SPLIT"))


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
saved_model_path = os.path.join(MODEL_DIR, LSTM_MODEL_FILENAME)
train_data = os.path.join(DATA_DIR, TRAIN_CSV)
eval_data = os.path.join(DATA_DIR, EVAL_CSV)  # using part of train as eval

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



    

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    # print(len(train), len(eval_dataset))
    # print(train)
    # print(len(dataset))
    # print(dataset[0])
    # print(len(train_ds), len(val_ds))
    # print(len(train_loader), len(val_loader))
    # print("*************")
    # sys.exit(0)
    model = LSTMForecast(N_FEATURES)
    model_path = saved_model_path
    if os.path.exists(model_path):
        print("Resuming LSTM...")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    print("Training LSTM...")
    train_model(model, MODEL_DIR, "lstm", train_loader, val_loader, epochs=EPOCHS, device=DEVICE, stop_threshold=STOP_THRESHOLD)

if __name__ == "__main__":
    main()

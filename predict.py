#!/usr/bin/env python3
import os, sys
import numpy as np
import pandas as pd
import torch
from common.dataset import TimeSeriesDataset
from common.plot_anomalies import plot_anomalies
from common.forecasting import recursive_forecast
from common.evaluation import evaluate_model
from torch.utils.data import DataLoader
from lstm.model.lstm_model import LSTMForecast
from tcn.model.tcn_model import TCNForecast
from config.config import (
    INPUT_WINDOW, FORECAST_HORIZON, N_FEATURES, TIMESERIES_MODE,
    BATCH_SIZE, TEST_CSV, LSTM_MODEL_FILENAME, TCN_MODEL_FILENAME
    )

# TODO: need to implement prediction graph

# import yaml

# config_path = os.path.join("config", "config.yaml")
# with open(config_path, "r") as f:
#     config = yaml.safe_load(f)

# =====================
# Main
# =====================
if __name__ == "__main__":
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # INPUT_WINDOW, FORECAST_HORIZON, N_FEATURES, EPOCHS = 300, 300, 5, 5
    # thresholds = [5, 5, 5, 5, 5]
    # INPUT_WINDOW = config["input_window"]
    # FORECAST_HORIZON = config["forecast_horizon"]
    # N_FEATURES = config["n_features"]
    # EPOCHS = config["epochs"]
    # thresholds = config["thresholds"]
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # project root
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    lstm_saved_model_path = os.path.join(MODEL_DIR, LSTM_MODEL_FILENAME)
    tcn_saved_model_path = os.path.join(MODEL_DIR, TCN_MODEL_FILENAME)
    test_data = os.path.join(DATA_DIR, TEST_CSV)

    # ---- Load train/test CSV ----
    test_df = pd.read_csv(test_data)

    # Drop timestamp column (keep only features)
    test = test_df.drop(columns=["timestamp"]).values.astype(np.float32)

    # Build test dataset (sliding windows from test file)
    test_dataset = TimeSeriesDataset(test, INPUT_WINDOW, FORECAST_HORIZON, TIMESERIES_MODE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # ---- Models ----
    lstm, tcn = LSTMForecast(N_FEATURES), TCNForecast(N_FEATURES)

    # ---- Reload Models for Testing ----
    lstm.load_state_dict(torch.load(lstm_saved_model_path, map_location=DEVICE))
    tcn.load_state_dict(torch.load(tcn_saved_model_path, map_location=DEVICE))
    lstm.to(DEVICE); tcn.to(DEVICE)

    # ---- Final Test Evaluation ----
    test_loss_lstm = evaluate_model(lstm, test_loader, DEVICE)
    test_loss_tcn = evaluate_model(tcn, test_loader, DEVICE)
    print(f"\nFinal Test Loss (LSTM): {test_loss_lstm:.4f}")
    print(f"Final Test Loss (TCN): {test_loss_tcn:.4f}")

    # ---- Forecast ----
    last_input = test[-INPUT_WINDOW:]
    lstm_preds = recursive_forecast(lstm, last_input, FORECAST_HORIZON, DEVICE)
    print("************")
    print(lstm_preds)
    sys.exit(0)
    tcn_preds = recursive_forecast(tcn, last_input, FORECAST_HORIZON, DEVICE)
    true_vals = test[-FORECAST_HORIZON:]

    # ---- Print samples ----
    print("\nLast 5 input rows:")
    print(last_input[-5:])
    print("\nPredicted row at t+3000 (LSTM):", lstm_preds[-1])
    print("Predicted row at t+3000 (TCN):", tcn_preds[-1])

    # ---- Compute MSE ----
    mse_lstm = np.mean((lstm_preds - true_vals) ** 2, axis=0)
    mse_tcn = np.mean((tcn_preds - true_vals) ** 2, axis=0)
    print("\nMSE per feature (LSTM):", mse_lstm, "| Avg:", np.mean(mse_lstm))
    print("MSE per feature (TCN):", mse_tcn, "| Avg:", np.mean(mse_tcn))

    # ---- Visualization ----
#     plot_anomalies(
#     true_vals=true_vals,
#     lstm_preds=lstm_preds,
#     tcn_preds=tcn_preds,
#     thresholds=thresholds,
#     save_path="./anomaly_plot/forecast_plot.png"
# )

    # plt.figure(figsize=(15, 12))
    # for i in range(N_FEATURES):
    #     plt.subplot(N_FEATURES, 1, i + 1)
    #     plt.plot(true_vals[:, i], label='Real', color='blue')
    #     plt.plot(lstm_preds[:, i], label='LSTM', color='green')
    #     plt.plot(tcn_preds[:, i], label='TCN', color='orange')
    #     # Anomaly markers
    #     lstm_anom = np.where(lstm_preds[:, i] > thresholds[i])[0]
    #     tcn_anom = np.where(tcn_preds[:, i] > thresholds[i])[0]
    #     plt.scatter(lstm_anom, lstm_preds[lstm_anom, i], marker='x', color='red', s=50,
    #                 label='LSTM Anomaly' if i == 0 else "")
    #     plt.scatter(tcn_anom, tcn_preds[tcn_anom, i], marker='o', facecolors='none', edgecolors='red', s=50,
    #                 label='TCN Anomaly' if i == 0 else "")
    #     plt.title(f'Feature {i}')
    #     if i == 0:
    #         plt.legend()
    # plt.tight_layout()
    # plt.savefig("forecast_plot.png")
    # print("\nPlot saved as forecast_plot.png")

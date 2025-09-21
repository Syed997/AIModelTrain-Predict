#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def plot_anomalies(true_vals, lstm_preds, tcn_preds, thresholds, save_path="forecast_plot.png"):
    """
    Plot anomalies for each feature in one plot per anomaly.
    
    Parameters:
        true_vals (np.ndarray): True test values, shape (HORIZON, N_FEATURES)
        lstm_preds (np.ndarray): LSTM predictions, shape (HORIZON, N_FEATURES)
        tcn_preds (np.ndarray): TCN predictions, shape (HORIZON, N_FEATURES)
        thresholds (list or np.ndarray): Threshold per feature for anomaly detection
        save_path (str): Where to save the plots
    """
    N_FEATURES = true_vals.shape[1]

    # ---- Find anomalies ----
    anomalies = []
    for i in range(N_FEATURES):
        lstm_anom_idx = np.where(lstm_preds[:, i] > thresholds[i])[0]
        tcn_anom_idx = np.where(tcn_preds[:, i] > thresholds[i])[0]
        anomalies.extend(lstm_anom_idx)
        anomalies.extend(tcn_anom_idx)
    anomalies = sorted(list(set(anomalies)))  # Unique sorted indices

    if not anomalies:
        print("⚠️ No anomalies detected based on thresholds.")
        return

    # ---- Plot each anomaly in a separate figure ----
    for idx in anomalies:
        plt.figure(figsize=(15, 6))
        for i in range(N_FEATURES):
            plt.plot(true_vals[:, i], label=f'Feature {i} True', color=f'C{i}', alpha=0.6)
            plt.plot(lstm_preds[:, i], label=f'Feature {i} LSTM', linestyle='--', color=f'C{i}', alpha=0.6)
            plt.plot(tcn_preds[:, i], label=f'Feature {i} TCN', linestyle=':', color=f'C{i}', alpha=0.6)

            # Highlight anomaly point
            if lstm_preds[idx, i] > thresholds[i]:
                plt.scatter(idx, lstm_preds[idx, i], color='red', marker='x', s=80, label='LSTM Anomaly' if i==0 else "")
            if tcn_preds[idx, i] > thresholds[i]:
                plt.scatter(idx, tcn_preds[idx, i], color='red', marker='o', s=80, facecolors='none', label='TCN Anomaly' if i==0 else "")

        plt.title(f"Anomaly at t={idx}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_path.replace('.png','')}_t{idx}.png")
        plt.close()
    print(f"\nPlots saved for {len(anomalies)} anomalies.")

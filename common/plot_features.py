#!/usr/bin/env python3
# plot_train_data.py

import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_data(start_row=0, end_row=None, file_path="", thresholds=None):
    """
    Plot telemetry features and mark anomalies.

    Args:
        start_row (int): starting row to plot
        end_row (int or None): ending row to plot
        file_path (str): CSV file path
        thresholds (dict or None): feature-wise threshold dictionary, e.g.
                                   {"cpu":85, "disk_io":200, "network_usage":200}
                                   If None, defaults to 85 for all features.
    """
    df = pd.read_csv(file_path)

    # Convert timestamp column to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Slice the dataframe
    end_row = end_row or len(df)
    df_slice = df.iloc[start_row:end_row]

    # Features (exclude timestamp)
    features = [col for col in df.columns if col != 'timestamp']

    # Default thresholds
    if thresholds is None:
        thresholds = {feat: 85 for feat in features}

    # Create figure
    plt.figure(figsize=(15, 12))
    for i, feat in enumerate(features):
        ax = plt.subplot(len(features), 1, i+1)
        
        # Plot the feature
        ax.plot(df_slice['timestamp'], df_slice[feat], label=feat)
        ax.set_ylabel(feat)
        ax.grid(True)

        # Highlight anomalies (values > threshold for that feature)
        feat_thresh = thresholds.get(feat, 85)  # fallback
        anomalies = df_slice[df_slice[feat] > feat_thresh]
        if not anomalies.empty:
            ax.scatter(anomalies['timestamp'], anomalies[feat], 
                       color='red', marker='x', s=80, label="Anomaly")

        if i == 0:
            ax.set_title(f"Telemetry Features from row {start_row} to {end_row}")
            ax.legend()

    plt.xlabel("Timestamp")
    plt.tight_layout()
    out_file = f"train_data_plot_{start_row}_{end_row}.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Plot saved as {out_file}")



# def plot_data(start_row=0, end_row=None, file_path=""):
#     df = pd.read_csv(file_path)

#     # Convert timestamp column to datetime
#     df['timestamp'] = pd.to_datetime(df['timestamp'])

#     # Slice the dataframe
#     end_row = end_row or len(df)  # if end_row is None, use full length
#     df_slice = df.iloc[start_row:end_row]

#     # Features (exclude timestamp)
#     features = [col for col in df.columns if col != 'timestamp']

#     # Create figure
#     plt.figure(figsize=(15, 12))
#     for i, feat in enumerate(features):
#         plt.subplot(len(features), 1, i+1)
#         plt.plot(df_slice['timestamp'], df_slice[feat], label=feat)
#         plt.ylabel(feat)
#         plt.grid(True)
#         if i == 0:
#             plt.title(f"Telemetry Features from row {start_row} to {end_row}")
#             plt.legend()
#     plt.xlabel("Timestamp")
#     plt.tight_layout()
#     plt.savefig(f"train_data_plot_{start_row}_{end_row}.png")
#     plt.close()
#     print(f"Plot saved as train_data_plot_{start_row}_{end_row}.png")


# if __name__ == "__main__":
#     plot_data(start_row=100, end_row=500, file_path="./data/test.csv")
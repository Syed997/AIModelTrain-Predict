import torch
from torch.utils.data import Dataset

# class TimeSeriesDataset(Dataset):
#     def __init__(self, data, input_window=10, forecast_horizon=3):
#         self.data = torch.from_numpy(data).float()
#         self.input_window = input_window
#         self.forecast_horizon = forecast_horizon

#     def __len__(self):
#         return len(self.data) - self.input_window - self.forecast_horizon + 1

#     def __getitem__(self, idx):
#         x = self.data[idx:idx+self.input_window]
#         y = self.data[idx+self.input_window:idx+self.input_window+self.forecast_horizon]
#         return x, y


# class TimeSeriesDataset(Dataset):
#     def _init_(self, data, input_window=10, forecast_horizon=3, mode="test"):
#         self.data = torch.from_numpy(data).float()
#         self.input_window = input_window
#         self.forecast_horizon = forecast_horizon
#         self.mode = mode

#     def _len_(self):
#         return len(self.data) - self.input_window - self.forecast_horizon + 1

#     def _getitem_(self, idx):
#         x = self.data[idx:idx + self.input_window]

#         if self.mode == "test":
#             y = self.data[idx + self.input_window : idx + self.input_window + self.forecast_horizon]
#         else:  # mode == "train"
#             y = self.data[idx + self.input_window]  # next value after input_window

#         return x, y

class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_window=10, forecast_horizon=5, mode="test"):
        self.data = torch.from_numpy(data).float()
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.mode = mode

    def __len__(self):
        return len(self.data) - self.input_window - self.forecast_horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.input_window]
        # if self.mode == "test":
        #     y = self.data[idx + self.input_window : idx + self.input_window + self.forecast_horizon]
        # else:
        #     y = self.data[idx + self.input_window]
        if self.mode == "train":
            y = self.data[idx + self.input_window]  # single step, shape: [n_features]
        elif self.mode == "test":
            y = self.data[idx + self.input_window : idx + self.input_window + self.forecast_horizon]  # multiple steps
        return x, y

class AutoencoderDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx):
        seq = self.data[idx : idx + self.seq_len]
        seq = torch.tensor(seq, dtype=torch.float32)
        return seq, seq  # input = target (reconstruction)
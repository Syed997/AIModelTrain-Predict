import numpy as np
from dataset import TimeSeriesDataset  # replace with actual import path
from torch.utils.data import Dataset
# import dataset

data = np.arange(200).astype(np.float32)
input_window = 10
forecast_horizon = 5
idx = 100

# Train mode
train_ds = TimeSeriesDataset(data, input_window, forecast_horizon, mode="train")
x_train, y_train = train_ds[idx]
print("Train mode:")
print("x:", x_train.numpy())
print("y:", y_train.item())  # should be 110

# Test mode
test_ds = TimeSeriesDataset(data, input_window, forecast_horizon, mode="test")
x_test, y_test = test_ds[idx]
print("\nTest mode:")
print("x:", x_test.numpy())
print("y:", y_test.numpy())  # should be [110, 111, 112, 113, 114]


# print(dataset.__file__)
import torch.nn as nn

class LSTMForecast(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

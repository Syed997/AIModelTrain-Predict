import torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x if self.chomp_size == 0 else x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, k, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_inputs, n_outputs, k, stride=stride, padding=padding, dilation=dilation),
            Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(n_outputs, n_outputs, k, stride=stride, padding=padding, dilation=dilation),
            Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout)
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.net(x) + (x if self.downsample is None else self.downsample(x)))

class TCNForecast(nn.Module):
    def __init__(self, n_inputs, n_channels=[64, 64], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(n_channels):
            dilation_size = 2 ** i
            in_ch = n_inputs if i == 0 else n_channels[i-1]
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, 1, dilation_size, padding, dropout))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(n_channels[-1], n_inputs)
    def forward(self, x):
        out = self.tcn(x.transpose(1,2))
        return self.fc(out[:, :, -1])

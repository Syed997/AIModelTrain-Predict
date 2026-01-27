import torch
import torch.nn as nn

# class LSTMAutoencoder(nn.Module):
#     def __init__(self, n_features, hidden_size=64, latent_size=32):
#         super().__init__()

#         # Encoder
#         self.encoder_lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
#         self.encoder_linear = nn.Linear(hidden_size, latent_size)

#         # Decoder
#         self.decoder_lstm = nn.LSTM(latent_size, hidden_size, batch_first=True)
#         self.decoder_linear = nn.Linear(hidden_size, n_features)

#     def forward(self, x):
#         # Encode
#         enc_output, _ = self.encoder_lstm(x)
#         latent = self.encoder_linear(enc_output[:, -1, :])  # compress last step

#         # Repeat latent across sequence length
#         latent_repeated = latent.unsqueeze(1).repeat(1, x.size(1), 1)

#         # Decode
#         dec_output, _ = self.decoder_lstm(latent_repeated)
#         reconstructed = self.decoder_linear(dec_output)

#         return reconstructed


class LSTMAutoencoder(nn.Module):
    def __init__(self, hidden_size=64, latent_size=32, num_layers=1, n_features=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.n_features = n_features
        self.built = False

        # Build immediately if n_features is provided (for loading checkpoints)
        if n_features is not None:
            self._build(n_features)

    def _build(self, n_features):
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.encoder_linear = nn.Linear(self.hidden_size, self.latent_size)

        self.decoder_lstm = nn.LSTM(
            input_size=self.latent_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.decoder_linear = nn.Linear(self.hidden_size, n_features)

        # Move layers to correct device
        if torch.cuda.is_available():
            self.to(torch.device("cuda"))

        self.built = True
        print(f"Autoencoder built for {n_features} features")

    def forward(self, x):
        if not self.built:
            n_features = x.size(-1)
            self._build(n_features)

        enc_output, _ = self.encoder_lstm(x)
        last_hidden = enc_output[:, -1, :]
        latent = self.encoder_linear(last_hidden)
        latent_repeated = latent.unsqueeze(1).repeat(1, x.size(1), 1)
        dec_output, _ = self.decoder_lstm(latent_repeated)
        reconstructed = self.decoder_linear(dec_output)
        return reconstructed
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_size=64, latent_size=32):
        super().__init__()

        # Encoder
        self.encoder_lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.encoder_linear = nn.Linear(hidden_size, latent_size)

        # Decoder
        self.decoder_lstm = nn.LSTM(latent_size, hidden_size, batch_first=True)
        self.decoder_linear = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        # Encode
        enc_output, _ = self.encoder_lstm(x)
        latent = self.encoder_linear(enc_output[:, -1, :])  # compress last step

        # Repeat latent across sequence length
        latent_repeated = latent.unsqueeze(1).repeat(1, x.size(1), 1)

        # Decode
        dec_output, _ = self.decoder_lstm(latent_repeated)
        reconstructed = self.decoder_linear(dec_output)

        return reconstructed

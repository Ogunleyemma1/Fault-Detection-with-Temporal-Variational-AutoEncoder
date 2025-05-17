import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim=3, latent_dim=3, hidden_dim=16, num_layers=2, dropout=0.2):
        """
        input_dim: Number of input features (e.g., 3 for 1DOF system)
        latent_dim: Size of the latent vector
        hidden_dim: Hidden dimension for LSTM layers
        num_layers: Number of stacked LSTM layers
        dropout: Dropout rate for regularization (used between LSTM layers)
        """
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ----------------------------
        # Encoder: LSTM → Latent Parameters
        # ----------------------------
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # ----------------------------
        # Decoder: Latent → LSTM → Output
        # ----------------------------
        self.fc_latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """
        x: shape (batch, seq_len, input_dim)
        Returns: mu, logvar
        """
        _, (h_n, _) = self.encoder_lstm(x)  # h_n: (num_layers, batch, hidden_dim)
        h_last = h_n[-1]  # take top layer's final hidden state
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        """
        z: shape (batch, latent_dim)
        seq_len: number of time steps to decode
        Returns: reconstruction (batch, seq_len, input_dim)
        """
        # Expand z to (batch, seq_len, hidden_dim)
        h0 = torch.tanh(self.fc_latent_to_hidden(z)).unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder_lstm(h0)
        out = self.output_layer(decoded)
        return out

    def forward(self, x):
        """
        x: input sequence (batch, seq_len, input_dim)
        Returns: recon_x, mu, logvar
        """
        seq_len = x.size(1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, seq_len)
        return recon, mu, logvar

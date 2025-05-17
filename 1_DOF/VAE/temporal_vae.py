# temporal_vae.py

import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim=3, latent_dim=3):
        """
        Fully connected VAE for 1DOF system (displacement, velocity, acceleration).
        input_dim: Input feature size (default 3 for 1DOF)
        latent_dim: Size of latent vector
        """
        super(VAE, self).__init__()
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        # Latent space
        self.fc_mu = nn.Linear(8, latent_dim)
        self.fc_logvar = nn.Linear(8, latent_dim)

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def encode(self, x):
        """
        x: Input tensor of shape (batch_size, input_dim)
        Returns: mu and logvar of latent space
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Sample latent vector z from mu and logvar using reparameterization trick
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decode latent vector to reconstruction
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Full forward pass through encoder, latent space, and decoder
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

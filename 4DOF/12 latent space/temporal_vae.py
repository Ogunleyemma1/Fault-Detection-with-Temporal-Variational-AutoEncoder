# temporal_vae.py

import torch
import torch.nn as nn

# ----------------------------
#  Variational Autoencoder (VAE) Definition
# ----------------------------

class VAE(nn.Module):
    def __init__(self, input_dim=12, latent_dim=12):
        """
        input_dim: Number of input features (12 for 4DOF [disp, vel, acc])
        latent_dim: Size of the compressed latent representation
        """
        super(VAE, self).__init__()

        #  Encoder Network (Input → Hidden → Compressed Representation)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 12),  # First hidden layer with 32 neurons
            nn.ReLU(),
            nn.Linear(12, 12),         # Second hidden layer with 16 neurons
            nn.ReLU(),
        )

        # Output layers from encoder to estimate parameters of latent Gaussian distribution
        self.fc_mu = nn.Linear(12, latent_dim)        # Mean of latent space
        self.fc_logvar = nn.Linear(12, latent_dim)    # Log-variance of latent space

        #  Decoder Network (Latent → Hidden → Reconstructed Input)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 12),  # Reverse path of encoder
            nn.ReLU(),
            nn.Linear(12, 12),
            nn.ReLU(),
            nn.Linear(12, input_dim),
            
        )

    def encode(self, x):
        """Encodes input into latent space mean and log-variance"""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """
        Applies the reparameterization trick:
        z = mu + std * eps, where eps ~ N(0,1)
        Allows gradient to flow through stochastic layer
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decodes latent variable back into reconstructed input"""
        return self.decoder(z)

    def forward(self, x):
        """
        Full forward pass: encode → sample → decode
        Returns: reconstructed input, mean, and log-variance
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

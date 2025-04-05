# temporal_vae.py

import torch
import torch.nn as nn

# ----------------------------
#  Variational Autoencoder (VAE) Definition
# ----------------------------

class VAE(nn.Module):
    def __init__(self, input_dim=12, latent_dim=3):
        """
        input_dim: Number of input features (12 for 4DOF [disp, vel, acc])
        latent_dim: Size of the compressed latent representation
        """
        super(VAE, self).__init__()

        #  Encoder Network (Input → Hidden → Compressed Representation)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),  # First hidden layer with 32 neurons
            nn.ReLU(),
            nn.Linear(32, 16),         # Second hidden layer with 16 neurons
            nn.ReLU(),
        )

        # Output layers from encoder to estimate parameters of latent Gaussian distribution
        self.fc_mu = nn.Linear(16, latent_dim)        # Mean of latent space
        self.fc_logvar = nn.Linear(16, latent_dim)    # Log-variance of latent space

        #  Decoder Network (Latent → Hidden → Reconstructed Input)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),  # Reverse path of encoder
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)    # Final layer returns reconstructed input of original size
        )

    def encode(self, x):
        """Encodes input into latent space mean and log-variance"""
        h = self.encoder(x)                          # Forward pass through encoder network
        return self.fc_mu(h), self.fc_logvar(h)      # Predict mean and log-variance of latent distribution

    def reparameterize(self, mu, logvar):
        """
        Applies the reparameterization trick:
        z = mu + std * eps, where eps ~ N(0,1)
        Allows gradient to flow through stochastic layer
        """
        std = torch.exp(0.5 * logvar)                # Convert log-variance to standard deviation
        eps = torch.randn_like(std)                  # Random sample from standard normal
        return mu + eps * std                        # Sample from N(mu, sigma^2)

    def decode(self, z):
        """Decodes latent variable back into reconstructed input"""
        return self.decoder(z)                       # Forward pass through decoder network

    def forward(self, x):
        """
        Full forward pass: encode → sample → decode
        Returns: reconstructed input, mean, and log-variance
        """
        mu, logvar = self.encode(x)                  # Encode input to latent parameters
        z = self.reparameterize(mu, logvar)          # Reparameterize to sample latent z
        recon = self.decode(z)                       # Decode to reconstruct input
        return recon, mu, logvar
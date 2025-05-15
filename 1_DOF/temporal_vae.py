import torch
import torch.nn as nn

# ----------------------------
#  Variational Autoencoder (VAE) Definition
# ----------------------------

class VAE(nn.Module):
    def __init__(self, input_dim=3, latent_dim=2):
        """
        input_dim: Number of input features (3 for 1DOF system: displacement, velocity, acceleration)
        latent_dim: Size of the latent representation (compressed feature vector)
        """
        super(VAE, self).__init__()

        # ----------------------------
        # Encoder Network (Input → Latent)
        # ----------------------------
        # Maps input features to a compressed hidden representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),  # First hidden layer with 16 neurons
            nn.ReLU(),
            nn.Linear(16, 8),          # Second hidden layer with 8 neurons
            nn.ReLU(),
        )

        # Latent space output: mean and log-variance
        self.fc_mu = nn.Linear(8, latent_dim)         # μ: mean of the latent Gaussian
        self.fc_logvar = nn.Linear(8, latent_dim)     # log(σ²): log variance of the latent Gaussian

        # ----------------------------
        # Decoder Network (Latent → Reconstructed Input)
        # ----------------------------
        # Maps latent space back to reconstructed input features
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),  # First decoder layer
            nn.ReLU(),
            nn.Linear(8, 16),          # Second decoder layer
            nn.ReLU(),
            nn.Linear(16, input_dim)   # Output layer (matches original input dimension)
        )

    # ----------------------------
    # Encoding: x → z (via μ, logvar)
    # ----------------------------
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    # ----------------------------
    # Reparameterization Trick: z = μ + σ * ε
    # ----------------------------
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)     # Convert log-variance to std
        eps = torch.randn_like(std)       # Sample ε ~ N(0, I)
        return mu + eps * std             # Return sampled latent vector z

    # ----------------------------
    # Decoding: z → x̂
    # ----------------------------
    def decode(self, z):
        return self.decoder(z)

    # ----------------------------
    # Full Forward Pass: x → x̂, μ, logvar
    # ----------------------------
    def forward(self, x):
        mu, logvar = self.encode(x)       # Encode input
        z = self.reparameterize(mu, logvar)  # Sample latent vector
        recon = self.decode(z)            # Decode reconstruction
        return recon, mu, logvar

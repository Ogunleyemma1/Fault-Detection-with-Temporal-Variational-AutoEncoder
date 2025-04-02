import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 🔹 Load and Preprocess Data
# ----------------------------

# Load dataset from CSV file into a Pandas DataFrame
df = pd.read_csv("vae_input_data.csv")

# Convert DataFrame to NumPy array and ensure data type is float32 (PyTorch default)
input = df.values.astype(np.float32)

# Normalize each feature (column) to have mean=0 and std=1
mean = input.mean(axis=0)
std = input.std(axis=0)
input_norm = (input - mean) / std  # Standardization

# Convert normalized NumPy array to PyTorch tensor
tensor_data = torch.tensor(input_norm)

# Wrap tensor into a Dataset and use DataLoader for batching and shuffling
dataset = TensorDataset(tensor_data)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)  # Mini-batch size of 64 with shuffling

# ----------------------------
# 🔹 Variational Autoencoder (VAE) Definition
# ----------------------------

class VAE(nn.Module):
    def __init__(self, input_dim=12, latent_dim=3):
        """
        input_dim: Number of input features (12 for 4DOF × [disp, vel, acc])
        latent_dim: Size of the compressed latent representation
        """
        super(VAE, self).__init__()

        # 🔸 Encoder Network (Input → Hidden → Compressed Representation)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),  # First hidden layer with 32 neurons
            nn.ReLU(),
            nn.Linear(32, 16),         # Second hidden layer with 16 neurons
            nn.ReLU(),
        )

        # Output layers from encoder to estimate parameters of latent Gaussian distribution
        self.fc_mu = nn.Linear(16, latent_dim)        # Mean of latent space
        self.fc_logvar = nn.Linear(16, latent_dim)    # Log-variance of latent space

        # 🔸 Decoder Network (Latent → Hidden → Reconstructed Input)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),  # Reverse path of encoder
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)    # Final layer returns reconstructed input of original size
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
        std = torch.exp(0.5 * logvar)      # Convert log-variance to standard deviation
        eps = torch.randn_like(std)        # Random sample from standard normal
        return mu + eps * std              # Sample from N(mu, sigma^2)
    
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

# ----------------------------
# 🔹 Loss Function (VAE)
# ----------------------------

def loss_function(recon, x, mu, logvar):
    """
    Combines:
    - Reconstruction loss (how close the output is to input)
    - KL divergence (regularizes latent space toward standard normal)
    """
    recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')  # Mean Squared Error over all elements
    kld = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())     # KL Divergence between encoded and standard normal
    return recon_loss + kld

# ----------------------------
# 🔹 Training the VAE
# ----------------------------

# Select device: GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the VAE model and move it to the selected device
model = VAE().to(device)

# Initialize Adam optimizer with learning rate of 0.001
optimizer = optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 50              # Number of complete passes over the entire dataset
losses = []                # Store average loss per epoch for visualization

# Training Loop
for epoch in range(n_epochs):
    model.train()          # Set model to training mode
    total_loss = 0         # Accumulate loss over batches
    
    for batch in dataloader:
        x_batch = batch[0].to(device)  # Get input batch and move to device
        
        optimizer.zero_grad()          # Clear gradients from previous step
        
        # Forward pass through the VAE
        recon, mu, logvar = model(x_batch)
        
        # Compute total loss = reconstruction + KL divergence
        loss = loss_function(recon, x_batch, mu, logvar)
        
        loss.backward()     # Compute gradients via backpropagation
        optimizer.step()    # Update model weights
        
        total_loss += loss.item()  # Accumulate loss
    
    # Compute average loss over the full dataset
    avg_loss = total_loss / len(dataloader.dataset)
    losses.append(avg_loss)        # Store for plotting
    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")  # Progress update

# ----------------------------
# 🔹 Plot Training Loss
# ----------------------------

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss of VAE")
plt.grid()
plt.show()

# ----------------------------
# 🔹 Save Trained Model Weights
# ----------------------------

torch.save(model.state_dict(), "temporal_vae_model.pt")

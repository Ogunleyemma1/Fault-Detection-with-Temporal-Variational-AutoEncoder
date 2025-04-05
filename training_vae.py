# training_vae.py

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from temporal_vae import VAE  # Import VAE model

# ----------------------------
#  Train VAE Model
# ----------------------------

def train_vae():
    # Load and preprocess input data
    df = pd.read_csv("vae_input_data.csv")
    input = df.values.astype(np.float32)

    # Normalize each feature (mean=0, std=1)
    mean = input.mean(axis=0)
    std = input.std(axis=0)
    input_norm = (input - mean) / std

    # Save mean and std for reuse in testing
    np.save("vae_mean.npy", mean)
    np.save("vae_std.npy", std)

    # Convert to PyTorch tensor and DataLoader
    tensor_data = torch.tensor(input_norm)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)  # Mini-batch learning

    # Define VAE loss function combining reconstruction + KL divergence
    def loss_function(recon, x, mu, logvar):
        recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')  # Reconstruction error
        kld = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())     # KL divergence term
        return recon_loss + kld

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)                     # Initialize model
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Optimizer setup

    n_epochs = 50
    losses = []

    # ----------------------------
    #  Training Loop
    # ----------------------------
    for epoch in range(n_epochs):
        model.train()          # Enable training mode (activates dropout/batchnorm if used)
        total_loss = 0

        for batch in dataloader:
            x_batch = batch[0].to(device)        # Move input to device
            optimizer.zero_grad()                # Reset gradients
            recon, mu, logvar = model(x_batch)   # Forward pass
            loss = loss_function(recon, x_batch, mu, logvar)  # Compute loss
            loss.backward()                      # Backpropagate
            optimizer.step()                     # Update weights
            total_loss += loss.item()            # Accumulate loss

        avg_loss = total_loss / len(dataloader.dataset)  # Average loss per epoch
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

    # ----------------------------
    #  Plot Training Loss
    # ----------------------------
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss of VAE")
    plt.grid()
    plt.show()

    # Save model weights
    torch.save(model.state_dict(), "temporal_vae_model.pt")

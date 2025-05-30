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

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(latent_dim=12).to(device)  # Make sure this matches the trained model architecture
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 150
    losses = []
    recon_losses = []
    kld_losses = []

    # ----------------------------
    #  Training Loop
    # ----------------------------
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kld = 0

        # KL warm-up: gradually increase KL weight
        kl_weight = min(1.0, epoch / 50.0)  # Linear ramp up for first 50 epochs

        for batch in dataloader:
            x_batch = batch[0].to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x_batch)

            # Feature-wise weights (more weight on acceleration components)
            weights = torch.tensor([1.0]*4 + [1.0]*4 + [1.0]*4, dtype=torch.float32, device=recon.device)
            recon_loss = ((weights * (recon - x_batch) ** 2)).sum()
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Apply KL warm-up to avoid collapse
            loss = recon_loss + kl_weight * kld

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld.item()

        avg_loss = total_loss / len(dataloader.dataset)
        avg_recon = total_recon / len(dataloader.dataset)
        avg_kld = total_kld / len(dataloader.dataset)
        losses.append(avg_loss)
        recon_losses.append(avg_recon)
        kld_losses.append(avg_kld)

        print(f"Epoch {epoch + 1}/{n_epochs} | Total Loss: {avg_loss:.4f} | Recon Loss: {avg_recon:.4f} | KL: {avg_kld:.4f} | KL Weight: {kl_weight:.2f}")

    # ----------------------------
    #  Plot Training Loss
    # ----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Total Loss")
    plt.plot(recon_losses, label="Reconstruction Loss")
    plt.plot(kld_losses, label="KL Divergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Components")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Save model weights
    torch.save(model.state_dict(), "temporal_vae_model.pt")

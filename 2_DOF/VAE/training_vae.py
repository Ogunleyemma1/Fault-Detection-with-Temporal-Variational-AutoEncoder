import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from temporal_vae import VAE  # Import VAE model
from structure2DOF import get_system_matrices, system_config, get_force_function  # Updated to import system config and force

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
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(latent_dim=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 300
    losses = []
    recon_losses = []
    kld_losses = []
    phys_losses = []

    # Get physical system matrices and external force
    M, C, K = get_system_matrices(device)
    F_ext = get_force_function()       


    # ----------------------------
    #  Training Loop
    # ----------------------------
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kld = 0
        total_phys = 0

        kl_weight = min(1.0, epoch / 50.0)
        lambda_phys = min(5.0, epoch / 10.0)


        for batch in dataloader:
            x_batch = batch[0].to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x_batch)

            # Weighted reconstruction loss
            weights = torch.ones_like(x_batch[0])
            recon_loss = ((weights * (recon - x_batch) ** 2)).sum()

            # KL Divergence
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Physics constraint loss
            x_rec = recon[:, 0:2]   # x1, x2
            v_rec = recon[:, 2:4]   # v1, v2
            a_rec = recon[:, 4:6]   # a1, a2



            # Time batch for physics (assumes consistent time span)
            t_batch = torch.linspace(0, system_config["T_total"], steps=x_rec.shape[0], device=device)
            F_batch = F_ext(t_batch).T

            # Residual: M*a + C*v + K*x - F
            residual = M @ a_rec.T + C @ v_rec.T + K @ x_rec.T - F_batch
            
            # Normalize residual column-wise (per sample)
            residual_norm = residual / (residual.norm(dim=0, keepdim=True) + 1e-8)
            physics_loss = torch.mean(residual_norm.pow(2))

            # Total loss
            loss = recon_loss + kl_weight * kld + lambda_phys * physics_loss
            loss.backward() # Computes Gradient of Loss
            optimizer.step() # Update model parameters to minize loss

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld.item()
            total_phys += physics_loss.item()

        # Logging
        dataset_size = len(dataloader.dataset)
        losses.append(total_loss / dataset_size)
        recon_losses.append(total_recon / dataset_size)
        kld_losses.append(total_kld / dataset_size)
        phys_losses.append(total_phys / dataset_size)

        print(f"Epoch {epoch + 1}/{n_epochs} | "
              f"Total Loss: {losses[-1]:.4f} | Recon: {recon_losses[-1]:.4f} | "
              f"KL: {kld_losses[-1]:.4f} | Phys: {phys_losses[-1]:.4f} | "
              f"λ_phys: {lambda_phys:.5f}")

    # ----------------------------
    #  Plot Training Loss
    # ----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Total Loss")
    plt.plot(recon_losses, label="Reconstruction Loss")
    plt.plot(kld_losses, label="KL Divergence")
    plt.plot(phys_losses, label="Physics Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Components")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    torch.save(model.state_dict(), "temporal_vae_model.pt")

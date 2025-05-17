import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from temporal_vae import VAE
from structure1DOF import get_system_matrices, system_config, get_force_function

def train_vae(seq_len=100):
    # Load and normalize input data
    df = pd.read_csv("vae_input_data.csv")
    data = df.values.astype(np.float32)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data_norm = (data - mean) / std
    np.save("vae_mean.npy", mean)
    np.save("vae_std.npy", std)

    # Generate overlapping sequences
    sequences = [data_norm[i:i + seq_len] for i in range(len(data_norm) - seq_len + 1)]
    sequences = np.stack(sequences)
    dataset = TensorDataset(torch.tensor(sequences))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dim=3, latent_dim=3).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)  # L2 regularization

    M, C, K = get_system_matrices(device)
    F_ext = get_force_function()

    # Scheduler for learning rate decay
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=np.exp(np.log(1e-6 / 1e-3) / 300))

    n_epochs = 300
    losses, recon_losses, kld_losses, phys_losses = [], [], [], []

    for epoch in range(n_epochs):
        model.train()
        total_loss = total_recon = total_kld = total_phys = 0.0
        kl_weight = min(1.0, epoch / 50.0)
        lambda_phys = min(10.0, epoch / 10.0)

        for batch in dataloader:
            x_batch = batch[0].to(device)  # (B, T, 3)
            recon, mu, logvar = model(x_batch)

            # Reconstruction & KL divergence
            recon_loss = torch.mean((recon - x_batch) ** 2)
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            # Denormalize outputs
            x_rec = recon[:, :, 0] * std[0] + mean[0]
            v_rec = recon[:, :, 1] * std[1] + mean[1]
            a_rec = recon[:, :, 2] * std[2] + mean[2]

            # Compute physics loss in real units
            B, T = x_rec.shape
            t = torch.linspace(0, system_config["T_total"], steps=T, device=device)
            F_batch = F_ext(t).T.expand(B, -1)  # (B, T)
            res = M * a_rec + C * v_rec + K * x_rec - F_batch

            residual_norm = res / (res.norm(dim=1, keepdim=True) + 1e-8)
            physics_loss = torch.mean(residual_norm ** 2)

            # Total loss
            loss = recon_loss + kl_weight * kld + lambda_phys * physics_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld.item()
            total_phys += physics_loss.item()

        scheduler.step()

        # Logging
        n_samples = len(dataloader.dataset)
        losses.append(total_loss / n_samples)
        recon_losses.append(total_recon / n_samples)
        kld_losses.append(total_kld / n_samples)
        phys_losses.append(total_phys / n_samples)

        print(f"Epoch {epoch+1}/{n_epochs} | Total: {losses[-1]:.4f} | Recon: {recon_losses[-1]:.4f} | "
              f"KL: {kld_losses[-1]:.4f} | Phys: {phys_losses[-1]:.4f} | λ_phys: {lambda_phys:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    # Plot training losses
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Total Loss")
    plt.plot(recon_losses, label="Reconstruction Loss")
    plt.plot(kld_losses, label="KL Divergence")
    plt.plot(phys_losses, label="Physics Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Components (1DOF LSTM-VAE)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    torch.save(model.state_dict(), "temporal_vae_model.pt")

if __name__ == "__main__":
    train_vae()

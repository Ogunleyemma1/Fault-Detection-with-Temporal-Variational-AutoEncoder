import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, TensorDataset

from temporal_vae import VAE

def kl_anneal_function(epoch, n_epochs, start=0.0, stop=1.0, anneal_ratio=0.3):
    # Sigmoid annealing for KL weight
    x = (epoch - (n_epochs * anneal_ratio)) / (n_epochs * anneal_ratio)
    return float(stop / (1.0 + np.exp(-x * 5)))

def train_vae(seq_len=100):
    # === REPLACE DATA LOADING BLOCK ===
    # 1. Load all healthy_seed CSVs
    HEALTHY_DIR = "data_generation/healthy_runs"
    dfs = []
    for f in sorted(os.listdir(HEALTHY_DIR)):
        if f.startswith("healthy_seed") and f.endswith(".csv"):
            df = pd.read_csv(os.path.join(HEALTHY_DIR, f))
            dfs.append(df.values.astype(np.float32))
    data = np.vstack(dfs)  # shape (n_samples, 12)
    print(f"Loaded data shape: {data.shape}")

    # 2. Normalize
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data_norm = (data - mean) / std
    np.save("vae_mean.npy", mean)
    np.save("vae_std.npy", std)

    # 3. Windowing
    windows = [data_norm[i:i+seq_len] for i in range(len(data_norm) - seq_len + 1)]
    windows = np.stack(windows)
    print(f"Total windows: {len(windows)}")

    # 4. Use only first 40% for training
    n_total = len(windows)
    n_train = int(0.4 * n_total)
    windows = windows[:n_train]
    print(f"Using {len(windows)} windows for training (40% of all healthy data)")

    dataset = TensorDataset(torch.tensor(windows, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dim=12, latent_dim=16, hidden_dim=128, num_layers=2, dropout=0.3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    n_epochs = 50
    losses, recon_losses, kld_losses = [], [], []

    for epoch in range(n_epochs):
        model.train()
        total_loss = total_recon = total_kld = 0.0

        kl_weight = kl_anneal_function(epoch, n_epochs)

        for batch in dataloader:
            x_batch = batch[0].to(device)
            recon, mu, logvar = model(x_batch)

            recon_loss = nn.functional.mse_loss(recon, x_batch)
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            loss = recon_loss + kl_weight * kld

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld.item()

        scheduler.step()

        losses.append(total_loss / len(dataloader))
        recon_losses.append(total_recon / len(dataloader))
        kld_losses.append(total_kld / len(dataloader))

        print(f"Epoch {epoch+1:03}/{n_epochs} | Total: {losses[-1]:.4f} | Recon: {recon_losses[-1]:.4f} | "
              f"KL: {kld_losses[-1]:.4f} | KL Weight: {kl_weight:.2f}")

    torch.save(model.state_dict(), "temporal_vae_model.pt")

    # --- SAVE LOSSES AND PLOT ---
    plot_dir = "VAE_Training_Plot"
    os.makedirs(plot_dir, exist_ok=True)

    np.save(os.path.join(plot_dir, "vae_total_loss.npy"), np.array(losses))
    np.save(os.path.join(plot_dir, "vae_recon_loss.npy"), np.array(recon_losses))
    np.save(os.path.join(plot_dir, "vae_kld_loss.npy"), np.array(kld_losses))

    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Total Loss")
    plt.plot(recon_losses, label="Reconstruction Loss")
    plt.plot(kld_losses, label="KL Divergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "vae_training_loss_plot.png"))
    plt.show()

if __name__ == "__main__":
    train_vae(seq_len=100)

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from temporal_vae import VAE
import os

def kl_anneal_function(epoch, n_epochs, start=0.0, stop=1.0, anneal_ratio=0.3):
    x = (epoch - (n_epochs * anneal_ratio)) / (n_epochs * anneal_ratio)
    return float(stop / (1.0 + np.exp(-x * 5)))

def train_vae(seq_len=100):
    print("Training 2DOF Temporal VAE (Original + Drifted + Amplitude + LowFreq)...")

    # --- Load and normalize data ---
    df = pd.read_csv("2DOF_signal_clean_variants.csv")
    df = df.iloc[:int(0.5 * len(df))]  # FIRST 50% FOR TRAINING
    data = df.drop(columns=["time"]).values.astype(np.float32)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std == 0] = 1e-6
    data_norm = (data - mean) / std
    np.save("vae_mean.npy", mean)
    np.save("vae_std.npy", std)

    # --- Prepare sequences (sliding window) ---
    sequences = [data_norm[i:i + seq_len] for i in range(len(data_norm) - seq_len + 1)]
    sequences = np.stack(sequences)
    dataset = TensorDataset(torch.tensor(sequences, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(
        input_dim=data.shape[1],  # 24 for 2DOF (4x6)
        latent_dim=5,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    n_epochs = 150
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

        print(f"Epoch {epoch+1:03}/{n_epochs} | Total: {losses[-1]:.5f} | Recon: {recon_losses[-1]:.5f} | "
              f"KL: {kld_losses[-1]:.5f} | KL Weight: {kl_weight:.2f}")

    torch.save(model.state_dict(), "temporal_vae_model.pt")
    print("✅ VAE model saved to temporal_vae_model.pt")

    # --- Plotting Losses ---
    os.makedirs("VAE Training Plot 2DOF", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Total Loss")
    plt.plot(recon_losses, label="Reconstruction Loss")
    plt.plot(kld_losses, label="KL Divergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("2DOF VAE Training Losses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("VAE Training Plot 2DOF/loss_plot_2dof.png")
    plt.show()

    # --- Latent Space Visualization (by variant) ---
    print("🔍 Extracting latent space representations by signal variant...")
    variant_names = ["Original", "Drifted", "Amplitude", "LowFreq"]
    group_size = 4  # Number of variants per variable
    n_vars = 6  # x1, x2, v1, v2, a1, a2
    variant_indices = {
        name: [i for i in range(j, data_norm.shape[1], group_size)]
        for j, name in enumerate(variant_names)
    }

    latent_all = []
    label_all = []

    max_samples = 500
    for label in variant_names:
        variant_data = np.zeros_like(data_norm)
        variant_data[:, variant_indices[label]] = data_norm[:, variant_indices[label]]
        sub_seqs = [variant_data[i:i + seq_len] for i in range(min(len(variant_data) - seq_len + 1, max_samples))]
        sub_seqs = torch.tensor(np.stack(sub_seqs), dtype=torch.float32).to(device)
        with torch.no_grad():
            mu, _ = model.encode(sub_seqs)
        latent_all.append(mu.cpu().numpy())
        label_all.extend([label] * len(mu))

    latent_all = np.concatenate(latent_all, axis=0)
    label_all = np.array(label_all)

    latent_2d = PCA(n_components=2).fit_transform(latent_all)
    colors = {"Original": "blue", "Drifted": "orange", "Amplitude": "purple", "LowFreq": "red"}
    plt.figure(figsize=(8, 6))
    for label in variant_names:
        idx = label_all == label
        plt.scatter(latent_2d[idx, 0], latent_2d[idx, 1], label=label, alpha=0.7, s=12, c=colors[label])
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("2DOF Latent Space PCA by Signal Type")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("VAE Training Plot 2DOF/latent_space_pca_2dof.png")
    plt.show()

    latent_cols = [f"z{i+1}" for i in range(latent_all.shape[1])]
    df_latent = pd.DataFrame(latent_all, columns=latent_cols)
    df_latent["SignalType"] = label_all
    df_melt = df_latent.melt(id_vars="SignalType", value_vars=latent_cols,
                             var_name="LatentDim", value_name="Value")
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df_melt, x="LatentDim", y="Value", hue="SignalType")
    plt.title("2DOF Latent Dimension Boxplot by Signal Type")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("VAE Training Plot 2DOF/latent_boxplot_2dof.png")
    plt.show()

if __name__ == "__main__":
    train_vae(seq_len=100)

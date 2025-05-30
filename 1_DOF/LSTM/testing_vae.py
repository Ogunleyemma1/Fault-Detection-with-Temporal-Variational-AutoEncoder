import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from temporal_vae import VAE

def test_vae(seq_len=80):
    print("Testing Temporal VAE (1DOF)...")
    df = pd.read_csv("vae_input_data.csv")
    data = df.values.astype(np.float32)
    mean = np.load("vae_mean.npy")
    std = np.load("vae_std.npy")
    data_norm = (data - mean) / std

    sequences = [data_norm[i:i + seq_len] for i in range(len(data_norm) - seq_len + 1)]
    sequences = np.stack(sequences)
    x = torch.tensor(sequences, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE(input_dim=3, latent_dim=8, hidden_dim=32, num_layers=2, dropout=0.2).to(device)
    model.load_state_dict(torch.load("temporal_vae_model.pt", map_location=device))
    model.eval()

    with torch.no_grad():
        x = x.to(device)
        recon, mu, logvar = model(x)
        recon_np = recon.cpu().numpy()

    full_len = len(data_norm)
    full_recon = np.zeros_like(data_norm)
    count = np.zeros((full_len, 1))
    for i in range(len(recon_np)):
        full_recon[i:i + seq_len] += recon_np[i]
        count[i:i + seq_len] += 1
    count[count == 0] = 1
    full_recon /= count

    recon_denorm = (full_recon * std) + mean
    pd.DataFrame(recon_denorm, columns=df.columns).to_csv("vae_reconstruction.csv", index=False)
    print("Reconstruction saved to vae_reconstruction.csv")

    time = np.arange(len(data_norm))
    dof_labels = ['x1', 'v1', 'a1']
    fig, axs = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
    for i in range(3):
        axs[i].plot(time, data_norm[:, i], label='Input (Normalized)', color='tab:blue')
        axs[i].plot(time, full_recon[:, i], label='Recon (Normalized)', color='tab:orange', alpha=0.7)
        axs[i].set_ylabel(dof_labels[i])
        axs[i].legend()
        axs[i].grid(True)
    axs[-1].set_xlabel("Time Step")
    plt.suptitle("1DOF LSTM-VAE Normalized Input vs Reconstruction")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_vae(seq_len=80)

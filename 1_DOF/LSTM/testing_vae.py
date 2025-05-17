import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from temporal_vae import VAE

def test_vae(seq_len=100):
    # ----------------------------
    # Load normalized input data
    # ----------------------------
    df = pd.read_csv("vae_input_data.csv")  # e.g., [x1, v1, a1]
    data = df.values.astype(np.float32)

    # ----------------------------
    # Load normalization stats
    # ----------------------------
    mean = np.load("vae_mean.npy")
    std = np.load("vae_std.npy")

    # Normalize
    data_norm = (data - mean) / std  # Shape: (timesteps, 3)

    # ----------------------------
    # Create sliding windows (input sequences)
    # ----------------------------
    sequences = []
    for i in range(len(data_norm) - seq_len):
        sequences.append(data_norm[i:i + seq_len])
    sequences = np.stack(sequences)  # Shape: (num_seq, seq_len, 3)

    # Convert to tensor
    x = torch.tensor(sequences)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # Load trained VAE model
    # ----------------------------
    model = VAE(input_dim=3, latent_dim=3).to(device)
    model.load_state_dict(torch.load("temporal_vae_model.pt", map_location=device))
    model.eval()

    # ----------------------------
    # Run model inference
    # ----------------------------
    with torch.no_grad():
        x = x.to(device)
        recon, mu, logvar = model(x)
        recon_np = recon.cpu().numpy()  # Shape: (num_seq, seq_len, 3)

    # ----------------------------
    # Reconstruct full sequence using overlap-averaging
    # ----------------------------
    full_len = len(data_norm)
    full_recon = np.zeros_like(data_norm)       # shape (timesteps, 3)
    count = np.zeros((full_len, 1))             # keep track of how many times each timestep was predicted

    for i in range(len(recon_np)):
        full_recon[i:i+seq_len] += recon_np[i]
        count[i:i+seq_len] += 1

    count[count == 0] = 1  # avoid division by zero
    full_recon /= count    # final averaged reconstruction (normalized)

    # ----------------------------
    # Save reconstructed denormalized data
    # ----------------------------
    recon_denorm = (full_recon * std) + mean
    recon_df = pd.DataFrame(recon_denorm, columns=df.columns)
    recon_df.to_csv("vae_reconstruction.csv", index=False)
    print("Reconstruction saved to vae_reconstruction.csv")

    # ----------------------------
    # Plot full comparison (normalized)
    # ----------------------------
    time = np.arange(len(data_norm))
    dof_labels = ['x1', 'v1', 'a1']

    fig, axs = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
    for i in range(3):
        axs[i].plot(time, data_norm[:, i], label='Input (Normalized)', color='tab:blue')
        axs[i].plot(time, full_recon[:, i], label='Recon (Unfiltered)', color='tab:orange', alpha=0.7)
        axs[i].set_ylabel(dof_labels[i])
        axs[i].legend()
        axs[i].grid(True)

    axs[-1].set_xlabel("Time Step")
    plt.suptitle("Full Time Series Comparison: Normalized Input vs Reconstruction")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_vae()

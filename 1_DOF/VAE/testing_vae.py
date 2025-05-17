import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from temporal_vae import VAE

# ----------------------------
# Butterworth Filter
# ----------------------------
def butterworth_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

def test_vae(cutoff=5.0, fs=1000.0, order=4):
    # ----------------------------
    # Load input data
    # ----------------------------
    df = pd.read_csv("vae_input_data.csv")  # Assumes columns: x1, v1, a1
    data = df.values.astype(np.float32)

    # ----------------------------
    # Load normalization parameters
    # ----------------------------
    mean = np.load("vae_mean.npy")
    std = np.load("vae_std.npy")

    # Normalize input
    data_norm = (data - mean) / std
    x = torch.tensor(data_norm, dtype=torch.float32)

    # ----------------------------
    # Load trained VAE model
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dim=3, latent_dim=3).to(device)
    model.load_state_dict(torch.load("temporal_vae_model.pt", map_location=device))
    model.eval()

    # ----------------------------
    # Run inference
    # ----------------------------
    with torch.no_grad():
        x = x.to(device)
        recon, mu, logvar = model(x)
        recon_np = recon.cpu().numpy()

    # ----------------------------
    # Apply Butterworth filter to smooth reconstruction
    # ----------------------------
    recon_np_filtered = butterworth_filter(recon_np, cutoff=cutoff, fs=fs, order=order)

    # ----------------------------
    # Denormalize output
    # ----------------------------
    recon_denorm = (recon_np_filtered * std) + mean
    recon_df = pd.DataFrame(recon_denorm, columns=df.columns)
    recon_df.to_csv("vae_reconstruction.csv", index=False)
    print(" Filtered reconstruction saved to vae_reconstruction.csv")

    # ----------------------------
    # Plot normalized input vs filtered reconstruction
    # ----------------------------
    time = np.arange(len(data_norm))
    dof_labels = ['x1', 'v1', 'a1']

    fig, axs = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
    for i in range(3):
        axs[i].plot(time, data_norm[:, i], label='Input (Normalized)', color='tab:blue')
        axs[i].plot(time, recon_np_filtered[:, i], label='Recon (Filtered)', color='tab:orange', alpha=0.7)
        axs[i].set_ylabel(dof_labels[i])
        axs[i].legend()
        axs[i].grid(True)

    axs[-1].set_xlabel("Time Step")
    plt.suptitle("Full Time Series Comparison: Normalized Input vs Filtered Reconstruction")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_vae()

# testing_vae.py

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from temporal_vae import VAE
from scipy.signal import butter, filtfilt

# ----------------------------
# Butterworth Filter Function
# ----------------------------
def butterworth_filter(data, cutoff, fs, order=4):
    """
    Applies a low-pass Butterworth filter to suppress high-frequency noise.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def test_vae(apply_filter=True, cutoff=5.0, fs=100.0, order=4):
    # ----------------------------
    # Load input data (expects normalized motion data)
    # ----------------------------
    df = pd.read_csv("vae_input_data.csv")  # Assumes columns: ['x1', 'v1', 'a1']
    input = df.values.astype(np.float32)

    # ----------------------------
    # Load normalization statistics
    # ----------------------------
    mean = np.load("vae_mean.npy")
    std = np.load("vae_std.npy")

    # ----------------------------
    # Normalize the input for testing
    # ----------------------------
    input_norm = (input - mean) / std
    x = torch.tensor(input_norm)

    # ----------------------------
    # Load trained VAE model
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dim=input.shape[1], latent_dim=3).to(device)
    model.load_state_dict(torch.load("temporal_vae_model.pt", map_location=device))
    model.eval()

    # ----------------------------
    # Perform inference to get reconstruction
    # ----------------------------
    with torch.no_grad():
        x = x.to(device)
        recon, mu, logvar = model(x)
        recon_np = recon.cpu().numpy()

    # ----------------------------
    # Apply Butterworth smoothing filter to reconstruction (optional)
    # ----------------------------
    if apply_filter:
        for i in range(recon_np.shape[1]):
            recon_np[:, i] = butterworth_filter(recon_np[:, i], cutoff=cutoff, fs=fs, order=order)

    # ----------------------------
    # Plot normalized input vs reconstruction for each DOF and variable
    # ----------------------------
    dof_labels = ['x', 'v', 'a']
    num_dofs = input.shape[1] // 3
    timesteps = input.shape[0]

    for i, label in enumerate(dof_labels):
        fig, axs = plt.subplots(num_dofs, 1, figsize=(14, 2.5 * num_dofs), sharex=True)
        axs = [axs] if num_dofs == 1 else axs

        for dof in range(num_dofs):
            idx = i * num_dofs + dof
            axs[dof].plot(input_norm[:, idx], label="Normalized Input", color="tab:blue")
            axs[dof].plot(recon_np[:, idx], label="Normalized Recon", color="tab:orange", alpha=0.7)
            axs[dof].set_title(f"DOF {dof+1} - Normalized Comparison - {label}{dof+1}")
            axs[dof].legend()
            axs[dof].grid(True)

        plt.tight_layout()
        plt.show()

    # ----------------------------
    # Denormalize and save reconstruction to CSV
    # ----------------------------
    recon_denorm = (recon_np * std) + mean
    recon_df = pd.DataFrame(recon_denorm, columns=df.columns)
    recon_df.to_csv("vae_reconstruction_filtered.csv", index=False)
    print("Filtered reconstruction saved to vae_reconstruction_filtered.csv")

if __name__ == "__main__":
    test_vae()

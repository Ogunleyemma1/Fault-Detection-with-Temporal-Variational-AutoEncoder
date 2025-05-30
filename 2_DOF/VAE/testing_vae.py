import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from temporal_vae import VAE

# Butterworth low-pass filter function
def butterworth_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

def test_vae(cutoff=5.0, fs=100.0, order=4):
    # Load input data
    df = pd.read_csv("vae_input_data.csv")
    input = df.values.astype(np.float32)

    # Load normalization stats
    mean = np.load("vae_mean.npy")
    std = np.load("vae_std.npy")

    # Normalize input
    input_norm = (input - mean) / std
    x = torch.tensor(input_norm)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dim=input.shape[1], latent_dim=3).to(device)
    model.load_state_dict(torch.load("temporal_vae_model.pt", map_location=device))
    model.eval()

    # Inference
    with torch.no_grad():
        x = x.to(device)
        recon, mu, logvar = model(x)
        recon_np = recon.cpu().numpy()

    # Apply Butterworth filter to reconstruction
    recon_np_filtered = butterworth_filter(recon_np, cutoff=cutoff, fs=fs, order=order)

    # Plot normalized input vs filtered reconstruction
    num_features = input.shape[1]
    num_dofs = num_features // 3
    timesteps = input.shape[0]
    variable_labels = ['x', 'v', 'a']

    for i, label in enumerate(variable_labels):  # loop over x, v, a
        fig, axs = plt.subplots(num_dofs, 1, figsize=(14, 2.5 * num_dofs), sharex=True)
        if num_dofs == 1:
            axs = [axs]

        for dof in range(num_dofs):
            idx = i * num_dofs + dof
            axs[dof].plot(input_norm[:, idx], label='Normalized Input', color='tab:blue')
            axs[dof].plot(recon_np_filtered[:, idx], label='Filtered Recon', color='tab:orange', alpha=0.8)
            axs[dof].set_title(f"DOF {dof+1} - Normalized Comparison - {label}{dof+1}")
            axs[dof].legend()
            axs[dof].grid(True)

        plt.tight_layout()
        plt.show()

    # Save denormalized filtered output
    recon_denorm = (recon_np_filtered * std) + mean
    recon_df = pd.DataFrame(recon_denorm, columns=df.columns)
    recon_df.to_csv("vae_reconstruction_filtered.csv", index=False)
    print("Filtered reconstruction saved to vae_reconstruction_filtered.csv")

if __name__ == "__main__":
    test_vae()

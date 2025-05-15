# testing_vae.py

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from temporal_vae import VAE

def test_vae():
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

    # ----------------------------
    #  Plot for all DOFs and variables
    # ----------------------------
    num_features = input.shape[1]
    num_dofs = num_features // 3
    timesteps = input.shape[0]
    variable_labels = ['x', 'v', 'a']
    plot_colors = ['tab:blue', 'tab:orange']

    for i, label in enumerate(variable_labels):  # x, v, a
        fig, axs = plt.subplots(num_dofs, 1, figsize=(14, 2.5 * num_dofs), sharex=True)
        if num_dofs == 1:
            axs = [axs]

        for dof in range(num_dofs):
            idx = i * num_dofs + dof
            axs[dof].plot(input_norm[:, idx], label='Normalized Input', color=plot_colors[0])
            axs[dof].plot(recon_np[:, idx], label='Normalized Recon', color=plot_colors[1], alpha=0.7)
            axs[dof].set_title(f"DOF {dof+1} - Normalized Comparison - {label}{dof+1}")
            axs[dof].legend()
            axs[dof].grid(True)

        plt.tight_layout()
        plt.show()

    # ----------------------------
    #  Save Denormalized Output
    # ----------------------------
    recon_denorm = (recon_np * std) + mean
    recon_df = pd.DataFrame(recon_denorm, columns=df.columns)
    recon_df.to_csv("vae_reconstruction.csv", index=False)
    print("Reconstruction saved to vae_reconstruction.csv")

if __name__ == "__main__":
    test_vae()

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
    model = VAE(latent_dim=3).to(device)
    model.load_state_dict(torch.load("temporal_vae_model.pt", map_location=device))
    model.eval()

    # Inference
    with torch.no_grad():
        x = x.to(device)
        recon, mu, logvar = model(x)
        recon_np = recon.cpu().numpy()

    # ----------------------------
    #  Plot for all DOFs
    # ----------------------------
    dof_labels = ['x', 'v', 'a']
    num_dofs = 4
    timesteps = 1000  # Plot first 1000 points

    for i, label in enumerate(dof_labels):  # x, v, a
        fig, axs = plt.subplots(num_dofs, 1, figsize=(14, 10), sharex=True)
        for dof in range(num_dofs):
            idx = i * num_dofs + dof  # col index in data
            axs[dof].plot(input_norm[:, idx][:timesteps], label='Normalized Input')
            axs[dof].plot(recon_np[:, idx][:timesteps], label='Normalized Recon', alpha=0.7)
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

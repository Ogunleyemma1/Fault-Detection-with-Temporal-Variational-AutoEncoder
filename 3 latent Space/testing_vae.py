# testing_vae.py

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from temporal_vae import VAE

# ----------------------------
#  Test/Infer using Trained VAE
# ----------------------------

def test_vae():
    # Load input data from CSV
    df = pd.read_csv("vae_input_data.csv")
    input = df.values.astype(np.float32)

    # Load saved normalization parameters from training
    mean = np.load("vae_mean.npy")
    std = np.load("vae_std.npy")

    # Apply normalization to match training scale
    input_norm = (input - mean) / std

    # Convert input to PyTorch tensor
    x = torch.tensor(input_norm)

    # ----------------------------
    #  Load Trained VAE Model
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(latent_dim=12).to(device)  # Make sure this matches the trained model architecture
    model.load_state_dict(torch.load("temporal_vae_model.pt", map_location=device))
    model.eval()  # Set to evaluation mode to disable dropout, etc.

    # ----------------------------
    #  Run Inference (Reconstruction)
    # ----------------------------
    with torch.no_grad():
        x = x.to(device)
        recon, mu, logvar = model(x)
        recon_np = recon.cpu().numpy()  # Get normalized reconstruction

    # ----------------------------
    #  Plot normalized input vs output
    # ----------------------------
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 10))
    for i, col in enumerate(['x1', 'v1', 'a1']):
        idx = df.columns.get_loc(col)
        plt.subplot(3, 1, i+1)
        plt.plot(input_norm[:, idx][:1000], label='Normalized Input')
        plt.plot(recon_np[:, idx][:1000], label='Normalized Recon', alpha=0.7)
        plt.title(f"Normalized Comparison - {col}")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    # ----------------------------
    #  Denormalize and Save Output
    # ----------------------------
    recon_denorm = (recon_np * std) + mean
    recon_df = pd.DataFrame(recon_denorm, columns=df.columns)
    recon_df.to_csv("vae_reconstruction.csv", index=False)
    print("Reconstruction saved to vae_reconstruction.csv")

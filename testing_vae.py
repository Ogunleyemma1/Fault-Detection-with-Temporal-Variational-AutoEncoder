# testing_vae.py

import torch
import numpy as np
import pandas as pd
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

    # Apply normalization
    input_norm = (input - mean) / std

    # Convert input to PyTorch tensor
    x = torch.tensor(input_norm)

    # Initialize model and load trained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)
    model.load_state_dict(torch.load("temporal_vae_model.pt", map_location=device))
    model.eval()  # Set to evaluation mode

    # Perform reconstruction without gradient tracking
    with torch.no_grad():
        x = x.to(device)
        recon, mu, logvar = model(x)
        recon = recon.cpu().numpy()  # Convert back to NumPy

    # Undo normalization to get real-world values
    recon_denorm = (recon * std) + mean

    # Save output to CSV for analysis
    recon_df = pd.DataFrame(recon_denorm, columns=df.columns)
    recon_df.to_csv("vae_reconstruction.csv", index=False)
    print("Reconstruction saved to vae_reconstruction.csv")

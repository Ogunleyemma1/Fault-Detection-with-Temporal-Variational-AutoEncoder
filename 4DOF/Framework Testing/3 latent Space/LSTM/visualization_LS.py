import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from temporal_vae import VAE

# Load data
df = pd.read_csv("vae_input_data.csv")
data = df.values.astype(np.float32)
mean = np.load("vae_mean.npy")
std = np.load("vae_std.npy")
data_norm = (data - mean) / std

# Create sequences as in training
seq_len = 100
sequences = [data_norm[i:i + seq_len] for i in range(len(data_norm) - seq_len + 1)]
sequences = np.stack(sequences)
x = torch.tensor(sequences, dtype=torch.float32)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(input_dim=12, latent_dim=16, hidden_dim=128, num_layers=2, dropout=0.3).to(device)
model.load_state_dict(torch.load("temporal_vae_model.pt", map_location=device))
model.eval()

# Encode data, collect latent means
with torch.no_grad():
    x = x.to(device)
    mu, logvar = model.encode(x)
    mu_np = mu.cpu().numpy()

# If latent_dim > 3, reduce to 2D with PCA for visualization
from sklearn.decomposition import PCA

if mu_np.shape[1] > 3:
    pca = PCA(n_components=2)
    mu_proj = pca.fit_transform(mu_np)
    fig = plt.figure()
    plt.scatter(mu_proj[:,0], mu_proj[:,1], alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Latent Space (PCA projection)")
    plt.show()
elif mu_np.shape[1] == 3:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(mu_np[:,0], mu_np[:,1], mu_np[:,2], alpha=0.7)
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_zlabel("z3")
    plt.title("3D Latent Space")
    plt.show()
else:
    plt.scatter(mu_np[:,0], np.zeros_like(mu_np[:,0]), alpha=0.7)
    plt.xlabel("z1")
    plt.title("1D Latent Space")
    plt.show()

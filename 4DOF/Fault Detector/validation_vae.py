import os
import glob
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# --- CONFIG ---
SEQ_LEN = 100
HEALTHY_DIR = "data_generation/healthy_runs"
MODEL_PATH = "temporal_vae_model.pt"
MEAN_PATH = "vae_mean.npy"
STD_PATH = "vae_std.npy"
PLOT_DIR = "VAE_Validation_and_Thresholding_Plots"
HEALTHY_FRAC = (0.4, 0.7)  # Use 40-70% of each healthy file
THRESHOLD_PERCENTILE = 99  # Use 99th percentile for threshold

os.makedirs(PLOT_DIR, exist_ok=True)

def load_vae():
    from temporal_vae import VAE
    params = dict(input_dim=12, latent_dim=16, hidden_dim=128, num_layers=2, dropout=0.3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(**params).to(device)
    vae.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    vae.eval()
    mean, std = np.load(MEAN_PATH), np.load(STD_PATH)
    return vae, mean, std, device

def get_windows(files, seq_len, frac=None):
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if frac is not None:
            n = len(df)
            start = int(frac[0] * n)
            end = int(frac[1] * n)
            df = df.iloc[start:end]
        dfs.append(df)
    data = pd.concat(dfs, axis=0).values.astype(np.float32)
    if len(data) < seq_len:
        return np.empty((0, seq_len, data.shape[1]))
    return np.stack([data[i:i+seq_len] for i in range(len(data)-seq_len+1)])

def mse_metric(x, y):
    return np.mean((x - y)**2)

def main():
    # --- Load model and normalization ---
    vae, vae_mean, vae_std, device = load_vae()
    # --- Healthy data windows ---
    healthy_files = sorted(glob.glob(os.path.join(HEALTHY_DIR, "*.csv")))
    healthy_windows = get_windows(healthy_files, SEQ_LEN, HEALTHY_FRAC)
    healthy_norm = (healthy_windows - vae_mean) / vae_std

    # --- Compute MSE for each healthy window ---
    healthy_mse = []
    for w in healthy_norm:
        x = torch.tensor(w[np.newaxis], dtype=torch.float32).to(device)
        recon = vae(x)[0].detach().cpu().numpy().squeeze()
        healthy_mse.append(mse_metric(w, recon))
    healthy_mse = np.array(healthy_mse)

    # --- Compute threshold based on percentile ---
    mse_threshold = np.percentile(healthy_mse, THRESHOLD_PERCENTILE)
    print(f"Statistical Threshold for MSE (at {THRESHOLD_PERCENTILE}th percentile): {mse_threshold:.5f}")
    print(f"Healthy MSE: min={healthy_mse.min():.5f}, max={healthy_mse.max():.5f}, mean={healthy_mse.mean():.5f}")

    # --- Plot distribution and threshold ---
    plt.figure(figsize=(10,6))
    plt.hist(healthy_mse, bins=60, alpha=0.8, color='tab:blue', label='Healthy MSE')
    plt.axvline(mse_threshold, color='red', linestyle='--', label=f'{THRESHOLD_PERCENTILE}th percentile (threshold)')
    plt.xlabel("MSE Reconstruction Error")
    plt.ylabel("Count")
    plt.yscale('log')
    plt.legend()
    plt.title(f"Healthy Data MSE Distribution\nThreshold (red): {mse_threshold:.5f}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "healthy_mse_hist_with_threshold.png"))
    plt.show()
    plt.close()

    # --- Save threshold to JSON ---
    threshold_json = os.path.join(PLOT_DIR, "mse_threshold_statistical.json")
    with open(threshold_json, "w") as f:
        json.dump({
            "mse_threshold_percentile": THRESHOLD_PERCENTILE,
            "threshold": float(mse_threshold)
        }, f, indent=2)
    print(f"Threshold saved to {threshold_json}")

if __name__ == "__main__":
    main()

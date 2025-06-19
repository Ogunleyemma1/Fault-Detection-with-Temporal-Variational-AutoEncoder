# diagnostics_feature_drift.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.stats import skew, kurtosis
from temporal_vae import VAE

# ---- Local model params ----
MODEL_PARAMS = dict(input_dim=12, latent_dim=16, hidden_dim=128, num_layers=2, dropout=0.3)

# --- User settings (adapt if needed) ---
SEQ_LEN = 100
NORMAL_FILE = "vae_input_data.csv"
VAE_MEAN_FILE = "vae_mean.npy"
VAE_STD_FILE = "vae_std.npy"
VAE_MODEL_FILE = "temporal_vae_model.pt"

# Define splits in FRACTIONS of the dataset length
splits = {
    "train": (0.0, 0.4),   # VAE training
    "val":   (0.4, 0.7),   # VAE/SVM/CNN validation
    "test":  (0.7, 1.0)    # Full-pipeline test
}

# --- Helper functions ---
def get_windows(df, start_frac, end_frac, seq_len=SEQ_LEN):
    n = len(df)
    start, end = int(n * start_frac), int(n * end_frac)
    sub = df.iloc[start:end].values.astype(np.float32)
    if len(sub) < seq_len:
        return None
    return np.stack([sub[i:i+seq_len] for i in range(len(sub)-seq_len+1)])

def extract_stats(windows):
    means, stds, skews, kurts = [], [], [], []
    for win in windows:
        means.append(np.nanmean(win, axis=0))
        stds.append(np.nanstd(win, axis=0))
        skews.append([np.nan_to_num(skew(win[:, i]), nan=0.0) for i in range(win.shape[1])])
        kurts.append([np.nan_to_num(kurtosis(win[:, i]), nan=0.0) for i in range(win.shape[1])])
    return (
        np.array(means), np.array(stds), np.array(skews), np.array(kurts)
    )

def extract_vae_mse(windows, vae, device):
    mses = []
    with torch.no_grad():
        for win in windows:
            x = torch.tensor(win[np.newaxis], dtype=torch.float32).to(device)
            recon = vae(x)[0].cpu().numpy().squeeze()
            mses.append(np.mean((win - recon) ** 2))
    return np.array(mses)

# --- Load all ---
df = pd.read_csv(NORMAL_FILE)
mean = np.load(VAE_MEAN_FILE)
std = np.load(VAE_STD_FILE)
std[std == 0] = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(**MODEL_PARAMS).to(device)
vae.load_state_dict(torch.load(VAE_MODEL_FILE, map_location=device))
vae.eval()

feature_dict = {}

for split_name, (frac_start, frac_end) in splits.items():
    print(f"\nExtracting features for {split_name} split...")
    windows = get_windows(df, frac_start, frac_end)
    if windows is None:
        print(f"  Not enough data for {split_name} split.")
        continue
    # Normalize with TRAIN mean/std
    windows_norm = (windows - mean) / std
    means, stds, skews, kurts = extract_stats(windows_norm)
    vae_mses = extract_vae_mse(windows_norm, vae, device)
    feature_dict[split_name] = {
        "means": means, "stds": stds, "skews": skews, "kurts": kurts, "vae_mse": vae_mses
    }
    print(f"  Windows: {windows.shape[0]}, VAE MSE avg: {vae_mses.mean():.5f}")

# --- Plot distributions ---
def plot_histogram(feat_arrs, labels, title, xlabel, bins=50):
    plt.figure(figsize=(8,4))
    for arr, lbl in zip(feat_arrs, labels):
        arr_flat = arr.flatten()
        plt.hist(arr_flat, bins=bins, alpha=0.6, label=lbl, density=True, histtype="stepfilled")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_histogram(
    [feature_dict[k]["means"] for k in feature_dict],
    list(feature_dict.keys()),
    "Feature Means (all features, all windows)", "Mean"
)
plot_histogram(
    [feature_dict[k]["stds"] for k in feature_dict],
    list(feature_dict.keys()),
    "Feature Stds (all features, all windows)", "Std"
)
plot_histogram(
    [feature_dict[k]["vae_mse"] for k in feature_dict],
    list(feature_dict.keys()),
    "VAE MSE per Window (normalized)", "VAE MSE"
)

# Optional: plot one feature index at a time for deeper debugging
feat_idx = 0  # change for other features (0–11)
plot_histogram(
    [feature_dict[k]["means"][:, feat_idx] for k in feature_dict],
    list(feature_dict.keys()),
    f"Means of Feature {feat_idx+1} across splits", f"Mean of feature {feat_idx+1}"
)

# You can add more plots for skews/kurts if needed

print("Done. Check the histograms for overlap/drift between splits.")


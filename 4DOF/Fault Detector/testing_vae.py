# testing_vae.py

import torch
import numpy as np
import pandas as pd
import os
from temporal_vae import VAE

# --- Configuration ---
SEQ_LEN = 100        # Length of sliding window for VAE input
NUM_DOFS = 4         # Number of DOFs (x, v, a per DOF)
MODEL_PARAMS = dict(input_dim=12, latent_dim=16, hidden_dim=128, num_layers=2, dropout=0.3)

# Dataset files (label: csv file path)
DATASETS = {
    "Normal": "vae_input_data.csv",
    "Struct_k2_Reduced": "structural_fault_k2_reduced.csv",
    "Struct_c1_Increased": "structural_fault_c1_increased.csv",
    "Sensor_x1_Zero": "sensor_fault_x1_zero.csv",
    "Sensor_v3_Noisy": "sensor_fault_v3_noisy.csv"
}
# Thresholds for decision logic
T_RECON_LOW, T_RECON_HIGH = 0.05, 0.15

# --- Related channels map for correlation checks ---
def related_channels_map(num_dofs):
    """
    For each channel index, specify which indices are physically related
    Used for interpreting correlation structure: e.g. x1 <-> v1, a1
    """
    m = {}
    for i in range(num_dofs):
        m[i] = [i + num_dofs, i + 2 * num_dofs]      # x: [v, a]
        m[i + num_dofs] = [i, i + 2 * num_dofs]      # v: [x, a]
        m[i + 2 * num_dofs] = [i, i + num_dofs]      # a: [x, v]
    return m
RELATED_CHANNELS = related_channels_map(NUM_DOFS)

# --- Model loader ---
def load_vae(model_path="temporal_vae_model.pt", mean_path="vae_mean.npy", std_path="vae_std.npy"):
    """
    Loads trained VAE and normalization stats.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = np.load(mean_path), np.load(std_path)
    model = VAE(**MODEL_PARAMS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, mean, std, device

# --- Data Preparation ---
def preprocess(filepath, mean, std, seq_len):
    """
    Loads, normalizes, and stacks data into overlapping windows for VAE input.
    """
    if not os.path.exists(filepath):
        return None
    data = pd.read_csv(filepath).values.astype(np.float32)
    data_norm = (data - mean) / std
    if len(data_norm) < seq_len:
        return None
    # Create overlapping sequences
    sequences = np.stack([data_norm[i:i+seq_len] for i in range(len(data_norm) - seq_len + 1)])
    return sequences

# --- Fault Detection Logic ---
def classify_fault(orig, recon):
    """
    Classify a time window as Normal, Sensor Fault, or Structural Fault using
    reconstruction error and Pearson correlation structure.
    """
    mse = np.mean((orig - recon) ** 2)
    per_ch_mse = np.mean((orig - recon) ** 2, axis=0)
    corr = np.corrcoef(orig.T) if orig.shape[1] > 1 else np.ones((orig.shape[1], orig.shape[1]))
    corr = np.nan_to_num(corr)  # Replace NaNs (from constant channels) with 0

    # Focus on the channel with the highest individual error
    worst_idx = np.argmax(per_ch_mse)
    # Get the average absolute correlation between this channel and its physical peers
    if worst_idx in RELATED_CHANNELS and RELATED_CHANNELS[worst_idx]:
        peer_corrs = [corr[worst_idx, idx] for idx in RELATED_CHANNELS[worst_idx] if 0 <= idx < corr.shape[1]]
        avg_abs_peer_corr = np.mean(np.abs(peer_corrs)) if peer_corrs else 0
    else:
        avg_abs_peer_corr = 0

    # Main logic:
    if mse <= T_RECON_LOW:
        return "Normal"
    elif mse > T_RECON_HIGH:
        # Sensor Fault: worst channel is *not* correlated with peers (abs(corr) ~ 0)
        if avg_abs_peer_corr < 0.3:
            return "Sensor Fault"
        # Structural Fault: worst channel *remains correlated* with its peers (abs(corr) large)
        else:
            return "Structural Fault"
    else:
        return "Ambiguous (Moderate MSE)"

# --- Main Pipeline ---
def main():
    model, mean, std, device = load_vae()
    results = []
    for label, path in DATASETS.items():
        seqs = preprocess(path, mean, std, SEQ_LEN)
        if seqs is None:
            print(f"Skipping {label}: Not enough data or file missing.")
            continue
        x = torch.tensor(seqs, dtype=torch.float32).to(device)
        with torch.no_grad():
            recon, mu, logvar = model(x)
            recon_np = recon.cpu().numpy()
        preds = []
        for orig, recon_win in zip(seqs, recon_np):
            pred = classify_fault(orig, recon_win)
            results.append({"dataset": label, "predicted_fault": pred})
            preds.append(pred)
        # Print quick summary for this dataset
        print(f"{label}: {pd.Series(preds).value_counts().to_dict()}")

    pd.DataFrame(results).to_csv("fault_detection_results.csv", index=False)
    print("Results saved to fault_detection_results.csv")

if __name__ == "__main__":
    main()

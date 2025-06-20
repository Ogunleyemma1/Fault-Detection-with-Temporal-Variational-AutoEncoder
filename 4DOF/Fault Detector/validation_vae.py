import os
import glob
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from scipy.stats import pearsonr

# --- CONFIG ---
SEQ_LEN = 100
HEALTHY_DIR = "data_generation/healthy_runs"
FAULT_DIR = "data_generation/faults"
MODEL_PATH = "temporal_vae_model.pt"
MEAN_PATH = "vae_mean.npy"
STD_PATH = "vae_std.npy"
PLOT_DIR = "VAE_Validation_and_Thresholding_Plots"
HEALTHY_FRAC = (0.4, 0.7)
FAULT_FRAC = (0, 0.3)  # Use first 30% of each fault file

os.makedirs(PLOT_DIR, exist_ok=True)

# --- UTILS ---
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
    return np.stack([data[i:i+seq_len] for i in range(len(data)-seq_len+1)]) if len(data) >= seq_len else np.empty((0, seq_len, data.shape[1]))

def list_fault_files():
    files = []
    for root, dirs, fs in os.walk(FAULT_DIR):
        for f in fs:
            if f.endswith(".csv") and not f.startswith("deviation_stats"):
                files.append(os.path.join(root, f))
    return sorted(files)

def mse_metric(x, y):
    return np.mean((x - y)**2)

def correlation_complement(x, y):
    r = [pearsonr(x[:,i], y[:,i])[0] if np.std(x[:,i]) > 1e-8 and np.std(y[:,i]) > 1e-8 else 0. for i in range(x.shape[1])]
    return 1.0 - np.mean(np.nan_to_num(r))

# --- MAIN ---
def main():
    # --- Load model and normalization ---
    vae, vae_mean, vae_std, device = load_vae()
    # --- Healthy ---
    healthy_files = sorted(glob.glob(os.path.join(HEALTHY_DIR, "*.csv")))
    healthy_windows = get_windows(healthy_files, SEQ_LEN, HEALTHY_FRAC)
    healthy_norm = (healthy_windows - vae_mean) / vae_std

    # --- Faults: separate sensor and structural ---
    fault_files = list_fault_files()
    sensor_faults, structural_faults = [], []
    for f in fault_files:
        if "sensor_faults" in f: sensor_faults.append(f)
        else: structural_faults.append(f)

    # --- Collect all errors ---
    def process_group(files, label):
        all_windows, all_mse, all_corr, file_stats = [], [], [], {}
        for f in files:
            df = pd.read_csv(f)
            n = len(df)
            use = df.iloc[int(FAULT_FRAC[0]*n):int(FAULT_FRAC[1]*n)]
            win = get_windows([f], SEQ_LEN, FAULT_FRAC)
            win_norm = (win - vae_mean) / vae_std if len(win) > 0 else np.empty((0,SEQ_LEN,12))
            mses, corrs = [], []
            for w in win_norm:
                x = torch.tensor(w[np.newaxis], dtype=torch.float32).to(device)
                recon = vae(x)[0].detach().cpu().numpy().squeeze()
                mses.append(mse_metric(w, recon))
                corrs.append(correlation_complement(w, recon))
                all_windows.append(w)
            mses, corrs = np.array(mses), np.array(corrs)
            file_stats[os.path.basename(f)] = (mses.min() if len(mses) else np.nan, mses.max() if len(mses) else np.nan)
            all_mse.extend(mses)
            all_corr.extend(corrs)
        return np.array(all_mse), np.array(all_corr), file_stats

    healthy_mse, healthy_corr = [], []
    for w in healthy_norm:
        x = torch.tensor(w[np.newaxis], dtype=torch.float32).to(device)
        recon = vae(x)[0].detach().cpu().numpy().squeeze()
        healthy_mse.append(mse_metric(w, recon))
        healthy_corr.append(correlation_complement(w, recon))
    healthy_mse, healthy_corr = np.array(healthy_mse), np.array(healthy_corr)

    mse_struct, corr_struct, struct_stats = process_group(structural_faults, "Structural Fault")
    mse_sensor, corr_sensor, sensor_stats = process_group(sensor_faults, "Sensor Fault")

    # --- LOGGING ---
    print(f"\nHEALTHY DATA: min MSE = {healthy_mse.min():.5f}, max MSE = {healthy_mse.max():.5f}")
    for k,v in struct_stats.items():
        print(f"  Structural Fault {k.replace('.csv','')}: min = {v[0]:10.5f}, max = {v[1]:10.5f}")
    for k,v in sensor_stats.items():
        print(f"  Sensor Fault {k.replace('.csv','')}: min = {v[0]:10.5f}, max = {v[1]:10.5f}")

    # --- HISTOGRAMS (MSE, CorrComp, zoomed) ---
    def plot_hist(data_groups, labels, title, xlabel, fname):
        plt.figure(figsize=(12,7))
        for d,l,c in zip(data_groups,labels,["tab:blue","tab:orange","tab:green"]):
            plt.hist(d, bins=60, alpha=0.7, label=l, color=c)
        plt.yscale('log')
        plt.xlabel(xlabel)
        plt.ylabel('Count')
        plt.legend()
        plt.title(f'Error Distribution (log scale): {title}')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, fname))
        plt.show()

    plot_hist([healthy_mse, mse_struct, mse_sensor], ['Healthy','Structural Faults','Sensor Faults'], "MSE Reconstruction Error", "MSE Reconstruction Error", "hist_mse_all.png")
    plot_hist([healthy_corr, corr_struct, corr_sensor], ['Healthy','Structural Faults','Sensor Faults'], "Correlation Complement Error", "Correlation Complement Error", "hist_corrcomp_all.png")
    plot_hist([healthy_mse, mse_struct], ['Healthy','Structural Faults'], "Healthy vs Structural (MSE)", "MSE Reconstruction Error", "hist_mse_zoomed.png")

    # --- THRESHOLD OPTIMIZATION (MSE & CorrComp, all) ---
    def optimize_threshold(errors_healthy, errors_fault, metric_name):
        y = np.concatenate([np.zeros(len(errors_healthy)), np.ones(len(errors_fault))])
        all_errors = np.concatenate([errors_healthy, errors_fault])
        thresholds = np.linspace(all_errors.min(), all_errors.max(), 50)
        prf, acc = [], []
        for th in thresholds:
            y_pred = (all_errors > th).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary', zero_division=0)
            a = accuracy_score(y, y_pred)
            prf.append([p,r,f1])
            acc.append(a)
        prf, acc = np.array(prf), np.array(acc)
        best_idx = np.argmax(prf[:,2])
        best_th = thresholds[best_idx]
        # --- PRF Plot ---
        plt.figure(figsize=(10,6))
        plt.plot(thresholds, prf[:,0], label="Precision")
        plt.plot(thresholds, prf[:,1], label="Recall")
        plt.plot(thresholds, prf[:,2], label="F1-score")
        plt.axvline(best_th, color='red', linestyle='--', label=f"Best F1: {prf[best_idx,2]:.4f}\nTh: {best_th:.2f}")
        plt.xlabel(f"Threshold ({metric_name})")
        plt.ylabel("Score")
        plt.title(f"Threshold Optimization: PRF (Healthy vs Faults) ({metric_name})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"prf_{metric_name.lower()}.png"))
        plt.show()
        # --- Accuracy Plot ---
        plt.figure(figsize=(8,4))
        plt.plot(thresholds, acc, label="Accuracy", color='purple')
        plt.axvline(best_th, color='red', linestyle='--', label=f"Best F1: {prf[best_idx,2]:.4f}\nTh: {best_th:.2f}")
        plt.xlabel(f"Threshold ({metric_name})")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy vs Threshold ({metric_name})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"acc_{metric_name.lower()}.png"))
        plt.show()
        # --- Save best threshold as JSON ---
        with open(os.path.join(PLOT_DIR, f"optimal_threshold_{metric_name.lower()}.json"), "w") as f:
            json.dump({"best_threshold": float(best_th), "best_F1": float(prf[best_idx,2])}, f, indent=2)
        print(f"\n--- {metric_name} ---\nPrecision: {prf[best_idx,0]:.4f}, Recall: {prf[best_idx,1]:.4f}, F1: {prf[best_idx,2]:.4f}, Best Threshold: {best_th:.5f}")

    # Use ALL faults (structural + sensor) for optimization
    optimize_threshold(healthy_mse, np.concatenate([mse_struct, mse_sensor]), "MSE")
    optimize_threshold(healthy_corr, np.concatenate([corr_struct, corr_sensor]), "Correlation Complement")

if __name__ == "__main__":
    main()

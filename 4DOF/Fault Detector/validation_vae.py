import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score, roc_curve, auc,
    precision_recall_curve, classification_report, confusion_matrix, roc_auc_score, average_precision_score
)
from scipy.stats import pearsonr
import os
import glob
from temporal_vae import VAE

SEQ_LEN = 100
HEALTHY_DIR = "data_generation/healthy_runs"
FAULT_DIR = "data_generation/faults"
MODEL_PATH = "temporal_vae_model.pt"
MEAN_PATH = "vae_mean.npy"
STD_PATH = "vae_std.npy"

def preprocess_windows(df, seq_len):
    data = df.values.astype(np.float32)
    if len(data) < seq_len:
        return None
    return np.stack([data[i:i+seq_len] for i in range(len(data) - seq_len + 1)])

def get_windows_from_csvs(files, seq_len):
    windows = []
    for f in files:
        df = pd.read_csv(f)
        win = preprocess_windows(df, seq_len)
        if win is not None:
            windows.append(win)
    if windows:
        return np.concatenate(windows, axis=0)
    else:
        return np.empty((0, seq_len, 12))

def load_vae():
    MODEL_PARAMS = dict(input_dim=12, latent_dim=16, hidden_dim=128, num_layers=2, dropout=0.3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(**MODEL_PARAMS).to(device)
    vae.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    vae.eval()
    mean = np.load(MEAN_PATH)
    std = np.load(STD_PATH)
    return vae, mean, std, device

def get_healthy_windows_30percent(seq_len):
    files = sorted(glob.glob(os.path.join(HEALTHY_DIR, "*.csv")))
    print("Healthy files found:", files)
    if not files:
        raise FileNotFoundError(f"No healthy CSV files found in {HEALTHY_DIR}")
    all_df = pd.concat([pd.read_csv(f) for f in files], axis=0)
    n_total = len(all_df)
    start = int(n_total * 0.4)
    end = int(n_total * 0.7)
    sub_df = all_df.iloc[start:end]
    return preprocess_windows(sub_df, seq_len)

def mse_metric(x, y):
    return np.mean((x - y) ** 2)

def correlation_complement(x, y):
    try:
        r = [pearsonr(x[:,i], y[:,i])[0] if not np.isnan(np.std(x[:,i])) and not np.isnan(np.std(y[:,i])) else 0. for i in range(x.shape[1])]
        r = np.nan_to_num(r)
        return 1.0 - np.mean(r)
    except Exception:
        return 1.0

def main():
    print("Loading VAE and normalization stats...")
    vae, vae_mean, vae_std, device = load_vae()

    print("Extracting healthy data (next 30%)...")
    healthy_windows = get_healthy_windows_30percent(SEQ_LEN)
    healthy_windows = (healthy_windows - vae_mean) / vae_std

    # --- Fault data per file ---
    print("\nExtracting fault data (first 30%) and calculating per-file error stats...")
    fault_files = sorted(glob.glob(os.path.join(FAULT_DIR, "*.csv")))
    print("Fault files found:", fault_files)

    per_fault_stats = []

    all_fault_windows = []
    all_fault_labels = []
    fault_file_labels = []

    with torch.no_grad():
        for f in fault_files:
            df = pd.read_csv(f)
            n = len(df)
            use_end = int(n * 0.3)
            df_slice = df.iloc[:use_end]
            win = preprocess_windows(df_slice, SEQ_LEN)
            if win is not None:
                win_norm = (win - vae_mean) / vae_std
                errs = []
                for i in range(win_norm.shape[0]):
                    x = torch.tensor(win_norm[i][np.newaxis], dtype=torch.float32).to(device)
                    recon = vae(x)[0].cpu().numpy().squeeze()
                    mse = mse_metric(win_norm[i], recon)
                    errs.append(mse)
                    all_fault_windows.append(win_norm[i])
                    all_fault_labels.append(1)
                    fault_file_labels.append(os.path.basename(f))
                errs = np.array(errs)
                print(f"\nFault file: {os.path.basename(f)}")
                print(f"  Num windows: {len(errs)}")
                print(f"  MSE range: {errs.min()}  {errs.max()}")
                print(f"  MSE mean: {errs.mean()}")
                print(f"  Sample errors: {errs[:10]}")
                per_fault_stats.append([os.path.basename(f), len(errs), errs.min(), errs.max(), errs.mean()])
            else:
                print(f"File {os.path.basename(f)} produced no windows (too short)")

    # --- Healthy errors ---
    healthy_errs = []
    with torch.no_grad():
        for i in range(healthy_windows.shape[0]):
            x = torch.tensor(healthy_windows[i][np.newaxis], dtype=torch.float32).to(device)
            recon = vae(x)[0].cpu().numpy().squeeze()
            mse = mse_metric(healthy_windows[i], recon)
            healthy_errs.append(mse)

    healthy_errs = np.array(healthy_errs)
    all_data = np.concatenate([healthy_windows, np.stack(all_fault_windows)], axis=0)
    all_labels = np.array([0]*len(healthy_windows) + [1]*len(all_fault_windows))

    # --- Print summary table ---
    print("\n==== Fault File MSE Error Summary ====")
    print(f"{'File':35s} {'Num':>5s} {'Min':>8s} {'Max':>8s} {'Mean':>8s}")
    for row in per_fault_stats:
        print(f"{row[0]:35s} {row[1]:5d} {row[2]:8.2f} {row[3]:8.2f} {row[4]:8.2f}")
    print(f"{'Healthy (ref)':35s} {len(healthy_errs):5d} {healthy_errs.min():8.2f} {healthy_errs.max():8.2f} {healthy_errs.mean():8.2f}")

    # --- Histogram plot (overlap) ---
    plt.figure(figsize=(12,6))
    plt.hist(healthy_errs, bins=40, alpha=0.5, label='Healthy')
    for (fname, _, _, _, _), f in zip(per_fault_stats, fault_files):
        df = pd.read_csv(f)
        n = len(df)
        use_end = int(n * 0.3)
        df_slice = df.iloc[:use_end]
        win = preprocess_windows(df_slice, SEQ_LEN)
        if win is not None:
            win_norm = (win - vae_mean) / vae_std
            file_errs = []
            with torch.no_grad():
                for i in range(win_norm.shape[0]):
                    x = torch.tensor(win_norm[i][np.newaxis], dtype=torch.float32).to(device)
                    recon = vae(x)[0].cpu().numpy().squeeze()
                    mse = mse_metric(win_norm[i], recon)
                    file_errs.append(mse)
            plt.hist(file_errs, bins=40, alpha=0.5, label=fname)
    plt.xlabel("MSE Error")
    plt.ylabel("Count")
    plt.title("Per-Fault Type MSE Error Distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Overall metrics/plots as before ---
    fault_errs = np.array([mse_metric(x, vae(torch.tensor(x[np.newaxis], dtype=torch.float32).to(device))[0].cpu().numpy().squeeze()) for x in np.stack(all_fault_windows)])
    y = np.concatenate([np.zeros(len(healthy_errs)), np.ones(len(fault_errs))])
    mse_errors = np.concatenate([healthy_errs, fault_errs])

    auroc = roc_auc_score(y, mse_errors)
    auprc = average_precision_score(y, mse_errors)
    print(f"\nAUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")

    # ROC curve
    fpr, tpr, roc_thresh = roc_curve(y, mse_errors)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'ROC (AUC={auroc:.3f})')
    plt.plot([0,1],[0,1],'--',color='grey')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (MSE error)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # PR curve
    prec, rec, pr_thresh = precision_recall_curve(y, mse_errors)
    plt.figure(figsize=(6,5))
    plt.plot(rec, prec, label=f'PRC (AUC={auprc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (MSE error)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Threshold optimization
    thresholds = np.linspace(mse_errors.min(), mse_errors.max(), 100)
    results = []
    for th in thresholds:
        y_pred = (mse_errors > th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary', zero_division=0)
        results.append([th, p, r, f1])
    results = np.array(results)
    best_idx = np.argmax(results[:,3])
    best_th = results[best_idx,0]
    plt.figure(figsize=(8,6))
    plt.plot(thresholds, results[:,1], label="Precision")
    plt.plot(thresholds, results[:,2], label="Recall")
    plt.plot(thresholds, results[:,3], label="F1-score")
    plt.axvline(best_th, color='red', linestyle='--', label=f"Best F1: {results[best_idx,3]:.4f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Optimization (PRF tradeoff, MSE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Error distribution plot (log scale)
    plt.figure(figsize=(8,6))
    plt.hist(healthy_errs, bins=50, alpha=0.5, label='Healthy')
    plt.hist(fault_errs, bins=50, alpha=0.5, label='Fault')
    plt.xlabel('MSE Reconstruction Error')
    plt.ylabel('Count')
    plt.legend()
    plt.yscale("log")
    plt.title('Error distribution: healthy vs fault (log scale)')
    plt.tight_layout()
    plt.show()

    print("Healthy MSE range:", healthy_errs.min(), healthy_errs.max())
    print("Fault MSE range:", fault_errs.min(), fault_errs.max())
    print("Sample healthy errors:", healthy_errs[:10])
    print("Sample fault errors:", fault_errs[:10])

    # "Manual" threshold
    th_manual = (healthy_errs.max() + fault_errs.min()) / 2
    y_pred_manual = (mse_errors > th_manual).astype(int)
    print("\nManual threshold (midpoint):", th_manual)
    print(classification_report(y, y_pred_manual, target_names=["Healthy", "Fault"]))
    print("Confusion matrix:\n", confusion_matrix(y, y_pred_manual))

if __name__ == "__main__":
    main()

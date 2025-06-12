import torch
import numpy as np
import pandas as pd
import os
import json
from temporal_vae import VAE
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# --- Config ---
SEQ_LEN = 100
MODEL_PARAMS = dict(input_dim=12, latent_dim=16, hidden_dim=128, num_layers=2, dropout=0.3)
NORMAL_DATA = "validation_data.csv"
FAULT_DATA = {
    "Struct_k2_Reduced": "structural_fault_k2_reduced.csv",
    "Struct_c1_Increased": "structural_fault_c1_increased.csv",
    "Sensor_x1_Zero": "sensor_fault_x1_zero.csv",
    "Sensor_v3_Noisy": "sensor_fault_v3_noisy.csv"
}
FAULT_FRAC = 0.25

def load_model(mean_path="vae_mean.npy", std_path="vae_std.npy", model_path="temporal_vae_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = np.load(mean_path), np.load(std_path)
    std[std == 0] = 1e-6
    model = VAE(**MODEL_PARAMS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, mean, std, device

def preprocess(file, mean, std, seq_len, input_dim, frac=1.0):
    if not os.path.exists(file): return None
    df = pd.read_csv(file)
    if frac < 1.0: df = df.iloc[:int(len(df)*frac)]
    data = df.values.astype(np.float32)
    if data.shape[1] != input_dim or len(data) < seq_len: return None
    data = (data - mean) / std
    return np.stack([data[i:i+seq_len] for i in range(len(data)-seq_len+1)])

def calc_diffs(model, device, sequences):
    mse, corr_comp = [], []
    with torch.no_grad():
        t = torch.tensor(sequences, dtype=torch.float32).to(device)
        for i in range(t.size(0)):
            orig = sequences[i]
            recon = model(t[i:i+1])[0].cpu().numpy().squeeze()
            mse.append(np.mean((orig - recon)**2))
            o_flat, r_flat = orig.flatten(), recon.flatten()
            if np.std(o_flat) < 1e-6 or np.std(r_flat) < 1e-6: p = 0
            else: p = np.corrcoef(o_flat, r_flat)[0,1]
            corr_comp.append(1-np.nan_to_num(p))
    return np.array(mse), np.array(corr_comp)

def find_all_metrics(values, labels):
    thresholds = np.linspace(values.min(), values.max(), 100)
    precisions, recalls, f1s, accs = [], [], [], []
    for t in thresholds:
        preds = (values > t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        acc = accuracy_score(labels, preds)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        accs.append(acc)
    best_f1_idx = np.argmax(f1s)
    return thresholds, precisions, recalls, f1s, accs, best_f1_idx

def plot_prf(thresholds, precisions, recalls, f1s, best_f1_idx, title):
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label="Precision", marker='o')
    plt.plot(thresholds, recalls, label="Recall", marker='o')
    plt.plot(thresholds, f1s, label="F1-Score", marker='o', linewidth=2)
    plt.axvline(thresholds[best_f1_idx], color='r', linestyle='--', label=f'F1 Threshold ({thresholds[best_f1_idx]:.3f})')
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.show()

def plot_accuracy(thresholds, accs, best_f1_idx, title):
    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, accs, label="Accuracy", color='purple', marker='o')
    plt.axvline(thresholds[best_f1_idx], color='r', linestyle='--', label=f'F1 Threshold ({thresholds[best_f1_idx]:.3f})')

    # Annotate only the accuracy at the F1 threshold
    plt.annotate(f'Acc at F1: {accs[best_f1_idx]:.3f}\nThresh: {thresholds[best_f1_idx]:.3f}',
                 xy=(thresholds[best_f1_idx], accs[best_f1_idx]),
                 xytext=(thresholds[best_f1_idx], accs[best_f1_idx]+0.08),
                 arrowprops=dict(facecolor='red', arrowstyle='->'),
                 fontsize=11, color='red')
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.show()

def main():
    print("--- Optimizing Thresholds ---")
    model, mean, std, device = load_model()
    input_dim = MODEL_PARAMS['input_dim']

    # Normal data
    norm_seq = preprocess(NORMAL_DATA, mean, std, SEQ_LEN, input_dim)
    if norm_seq is None:
        print("Could not process normal data."); return
    norm_mse, norm_corr = calc_diffs(model, device, norm_seq)
    labels_norm = np.zeros(len(norm_mse))

    # Fault data
    faults_mse, faults_corr = [], []
    for label, file in FAULT_DATA.items():
        fault_seq = preprocess(file, mean, std, SEQ_LEN, input_dim, frac=FAULT_FRAC)
        if fault_seq is not None:
            m, c = calc_diffs(model, device, fault_seq)
            faults_mse.append(m)
            faults_corr.append(c)
        else:
            print(f"Skipped {label}: no data after preprocessing.")

    if not faults_mse:
        print("No fault data processed."); return

    faults_mse = np.concatenate(faults_mse)
    faults_corr = np.concatenate(faults_corr)
    labels_fault = np.ones(len(faults_mse))

    # All together
    mse_all = np.concatenate([norm_mse, faults_mse])
    corr_all = np.concatenate([norm_corr, faults_corr])
    labels_all = np.concatenate([labels_norm, labels_fault])

    # MSE metrics and plots
    thresholds_mse, precisions_mse, recalls_mse, f1s_mse, accs_mse, best_f1_idx_mse = find_all_metrics(mse_all, labels_all)
    plot_prf(thresholds_mse, precisions_mse, recalls_mse, f1s_mse, best_f1_idx_mse, "PRF vs Threshold (MSE-like)")
    plot_accuracy(thresholds_mse, accs_mse, best_f1_idx_mse, "Accuracy vs Threshold (MSE-like)")

    print(f"Best MSE threshold (F1): {thresholds_mse[best_f1_idx_mse]:.4f} (F1={f1s_mse[best_f1_idx_mse]:.3f}, Accuracy={accs_mse[best_f1_idx_mse]:.3f})")

    # Correlation metrics and plots
    thresholds_corr, precisions_corr, recalls_corr, f1s_corr, accs_corr, best_f1_idx_corr = find_all_metrics(corr_all, labels_all)
    plot_prf(thresholds_corr, precisions_corr, recalls_corr, f1s_corr, best_f1_idx_corr, "PRF vs Threshold (Correlation Complement)")
    plot_accuracy(thresholds_corr, accs_corr, best_f1_idx_corr, "Accuracy vs Threshold (Correlation Complement)")

    print(f"Best Corr threshold (F1): {thresholds_corr[best_f1_idx_corr]:.4f} (F1={f1s_corr[best_f1_idx_corr]:.3f}, Accuracy={accs_corr[best_f1_idx_corr]:.3f})")

    print(f"\nSummary:")
    print(f"Accuracy at best F1 for MSE: {accs_mse[best_f1_idx_mse]:.3f} at threshold {thresholds_mse[best_f1_idx_mse]:.4f}")
    print(f"Accuracy at best F1 for Corr Complement: {accs_corr[best_f1_idx_corr]:.3f} at threshold {thresholds_corr[best_f1_idx_corr]:.4f}")

    # Save
    thresholds = {
        "MSE_optimal_F1": float(thresholds_mse[best_f1_idx_mse]),
        "CorrComplement_optimal_F1": float(thresholds_corr[best_f1_idx_corr]),
        "MSE_normal_max": float(thresholds_mse[best_f1_idx_mse]*0.8),
        "MSE_fault_min": float(thresholds_mse[best_f1_idx_mse]*1.2)
    }
    with open("optimized_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    print("Saved thresholds to optimized_thresholds.json")

if __name__ == "__main__":
    main()

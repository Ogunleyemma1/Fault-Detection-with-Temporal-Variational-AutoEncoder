# test_vae_cnn_pipeline.py

import torch
import numpy as np
import pandas as pd
import json
import joblib
from temporal_vae import VAE
from cnn_model import CNN, SEQ_LEN, NUM_FEATURES
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PARAMS = dict(input_dim=12, latent_dim=16, hidden_dim=128, num_layers=2, dropout=0.3)
DROPOUT = 0.4

TEST_FILES = {
    "Normal": ("vae_input_data.csv", "Normal", (0.7, 1.0)),  # Use last 30%
    "Struct_k2_Reduced": ("structural_fault_k2_reduced.csv", "Structural Fault", (0.75, 1.0)),
    "Struct_c1_Increased": ("structural_fault_c1_reduced.csv", "Structural Fault", (0.75, 1.0)),
    "Sensor_x1_Zero": ("sensor_fault_x1_zero.csv", "Sensor Fault", (0.75, 1.0)),
    "Sensor_v3_Noisy": ("sensor_fault_v3_noisy.csv", "Sensor Fault", (0.75, 1.0)),
}

def extract_features_and_vae_mse(windows, vae, mean, std, device):
    feat_list, vae_mse_list = [], []
    with torch.no_grad():
        for win in windows:
            feats = []
            for i in range(win.shape[1]):
                x = win[:, i]
                feats.extend([
                    np.nanmean(x), np.nanstd(x), np.nanmin(x), np.nanmax(x),
                    np.nan_to_num(skew(x), nan=0.0, posinf=0.0, neginf=0.0),
                    np.nan_to_num(kurtosis(x), nan=0.0, posinf=0.0, neginf=0.0)
                ])
            win_tensor = torch.tensor(win[np.newaxis], dtype=torch.float32).to(device)
            recon = vae(win_tensor)[0].cpu().numpy().squeeze()
            vae_mse = np.mean((win - recon) ** 2)
            vae_mse_list.append(vae_mse)
            feat_list.append(feats)
    feat_arr = np.array(feat_list)
    vae_arr = np.array(vae_mse_list)[:, None]
    return np.hstack([feat_arr, vae_arr])

def preprocess_windows(filepath, mean, std, seq_len, frac_range=(0.0, 1.0)):
    df = pd.read_csv(filepath)
    n = len(df)
    start, end = int(n * frac_range[0]), int(n * frac_range[1])
    df = df.iloc[start:end]
    data = df.values.astype(np.float32)
    if len(data) < seq_len:
        return None
    data_norm = (data - mean) / std
    return np.stack([data_norm[i:i+seq_len] for i in range(len(data_norm) - seq_len + 1)])

def class_names():
    return ["Normal", "Sensor Fault", "Structural Fault"]

def main():
    svm_scaler = joblib.load("svm_scaler.pkl")
    svm_model = joblib.load("svm_model.pkl")
    with open("svm_best_threshold.json") as f:
        svm_best_threshold = float(json.load(f)["svm_best_threshold"])
    vae, vae_mean, vae_std, vae_device = VAE(**MODEL_PARAMS).to("cpu"), np.load("vae_mean.npy"), np.load("vae_std.npy"), torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.load_state_dict(torch.load("temporal_vae_model.pt", map_location=vae_device))
    vae.eval()
    cnn, cnn_device = CNN(input_channels=2, dropout=DROPOUT).to(vae_device), vae_device
    cnn.load_state_dict(torch.load("cnn_model.pt", map_location=cnn_device))
    cnn.eval()

    y_true, y_pred = [], []
    all_svm_scores, all_true_labels_svm = [], []

    for test_key, (filepath, fault_class, frac_range) in TEST_FILES.items():
        print(f"Processing: {filepath}  |  True label: {fault_class}")
        windows = preprocess_windows(filepath, vae_mean, vae_std, SEQ_LEN, frac_range=frac_range)
        if windows is None:
            print(f"Skipping {filepath}: not enough data")
            continue
        feats = extract_features_and_vae_mse(windows, vae, vae_mean, vae_std, vae_device)
        feats_scaled = svm_scaler.transform(feats)
        svm_scores_batch = svm_model.decision_function(feats_scaled)
        all_svm_scores.extend(svm_scores_batch)
        all_true_labels_svm.extend([0 if fault_class == "Normal" else 1] * len(svm_scores_batch))

        with torch.no_grad():
            for i, win in enumerate(windows):
                is_fault = svm_scores_batch[i] > svm_best_threshold
                if not is_fault:
                    y_pred.append(0)
                else:
                    win_tensor = torch.tensor(win[np.newaxis], dtype=torch.float32).to(cnn_device)
                    recon = vae(win_tensor)[0].cpu().numpy().squeeze()
                    recon_error = (win - recon) ** 2
                    stacked = np.stack([win, recon_error], axis=0)
                    x_cnn = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0).to(cnn_device)
                    out = cnn(x_cnn)
                    pred_cnn = out.argmax(1).item()
                    y_pred.append(pred_cnn + 1)  # 1=Sensor, 2=Structural
                if fault_class == "Normal":
                    y_true.append(0)
                elif fault_class == "Sensor Fault":
                    y_true.append(1)
                elif fault_class == "Structural Fault":
                    y_true.append(2)

    # Plot SVM score distribution for debug
    all_svm_scores = np.array(all_svm_scores)
    all_true_labels_svm = np.array(all_true_labels_svm)
    plt.figure(figsize=(10,5))
    plt.hist(all_svm_scores[all_true_labels_svm == 0], bins=50, alpha=0.5, label='Normal', color='cornflowerblue')
    plt.hist(all_svm_scores[all_true_labels_svm == 1], bins=50, alpha=0.5, label='Fault', color='sandybrown')
    plt.axvline(svm_best_threshold, color='red', linestyle='--', label='SVM Threshold')
    plt.legend(); plt.title("SVM Score Distribution (Test)"); plt.xlabel("SVM Decision Function Score"); plt.ylabel("Count"); plt.grid(True); plt.tight_layout(); plt.show()

    print("\nResults Summary:")
    print(classification_report(y_true, y_pred, target_names=class_names()))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names(), yticklabels=class_names())
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Test Confusion Matrix (Final 30%)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

import os
import glob
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from temporal_vae import VAE
from cnn_model import CNN, SEQ_LEN, NUM_FEATURES
from sklearn.metrics import classification_report, confusion_matrix
import random

# =========================== CONFIGURATION ===========================
MODEL_PARAMS = dict(input_dim=12, latent_dim=16, hidden_dim=128, num_layers=2, dropout=0.3)
DROPOUT = 0.5                     # Should match the CNN definition
CNN_MODEL_PATH = "cnn_model.pt"
HEALTHY_DIR = "data_generation/healthy_runs"
FAULT_DIR = "data_generation/faults"
THRESHOLD_JSON = "VAE_Validation_and_Thresholding_Plots/mse_threshold_statistical.json"
PLOT_DIR = "Test_Results_Plots_Final"
SEED = 42

os.makedirs(PLOT_DIR, exist_ok=True)

# Deterministic run for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def class_names():
    """Class label mapping for output/plots."""
    return ["Normal", "Sensor Fault", "Structural Damage"]

def load_vae_and_stats(mean_path="vae_mean.npy", std_path="vae_std.npy", model_path="temporal_vae_model.pt"):
    mean, std = np.load(mean_path), np.load(std_path)
    std[std == 0] = 1e-6  # Prevent divide by zero
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(**MODEL_PARAMS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, mean, std, device

def load_cnn(model_path=CNN_MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(input_channels=2, dropout_rate=DROPOUT).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def get_windows_from_file(filepath, mean, std, seq_len, frac_range=(0.7, 1.0)):
    df = pd.read_csv(filepath)
    n = len(df)
    start, end = int(n * frac_range[0]), int(n * frac_range[1])
    if end - start < seq_len:
        return None
    data = df.iloc[start:end].values.astype(np.float32)
    data_norm = (data - mean) / std
    return np.stack([data_norm[i:i + seq_len] for i in range(len(data_norm) - seq_len + 1)])

def get_healthy_windows(healthy_dir, mean, std, seq_len, frac_range=(0.7, 1.0)):
    healthy_files = sorted(glob.glob(os.path.join(healthy_dir, "*.csv")))
    all_windows = []
    for fpath in healthy_files:
        windows = get_windows_from_file(fpath, mean, std, seq_len, frac_range)
        if windows is not None:
            all_windows.append(windows)
    return all_windows

def get_fault_windows(fault_dir, mean, std, seq_len, frac_range=(0.7, 1.0)):
    sensor_faults, structural_damage = [], []
    for root, _, files in os.walk(fault_dir):
        for f in files:
            if f.endswith(".csv") and not f.startswith("deviation_stats"):
                path = os.path.join(root, f)
                if "sensor_faults" in root.replace("\\", "/"):
                    windows = get_windows_from_file(path, mean, std, seq_len, frac_range)
                    if windows is not None:
                        sensor_faults.append(windows)
                elif "structural_faults" in root.replace("\\", "/"):
                    windows = get_windows_from_file(path, mean, std, seq_len, frac_range)
                    if windows is not None:
                        structural_damage.append(windows)
    return sensor_faults, structural_damage

def main():
    vae, vae_mean, vae_std, device = load_vae_and_stats()
    cnn, cnn_device = load_cnn()
    with open(THRESHOLD_JSON) as f:
        mse_threshold = float(json.load(f)["threshold"])
    print(f"Loaded MSE threshold: {mse_threshold:.5f}")

    y_true, y_pred = [], []

    # --- Healthy Data (Normal: label 0) ---
    healthy_windows_list = get_healthy_windows(HEALTHY_DIR, vae_mean, vae_std, SEQ_LEN)
    for windows in healthy_windows_list:
        with torch.no_grad():
            for win in windows:
                x = torch.tensor(win[np.newaxis], dtype=torch.float32).to(device)
                recon = vae(x)[0].cpu().numpy().squeeze()
                mse = np.mean((win - recon) ** 2)
                if mse <= mse_threshold:
                    y_pred.append(0)  # Normal
                else:
                    recon_error = (win - recon) ** 2
                    stacked = np.stack([win, recon_error], axis=0)
                    x_cnn = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0).to(cnn_device)
                    pred_cnn = cnn(x_cnn).argmax(1).item()
                    y_pred.append(pred_cnn + 1)  # CNN: 0=Sensor Fault, 1=Structural Damage, so +1
                y_true.append(0)

    # --- Fault Data (Sensor Fault: label 1, Structural Damage: label 2) ---
    sensor_faults_list, structural_damage_list = get_fault_windows(FAULT_DIR, vae_mean, vae_std, SEQ_LEN)

    # Sensor Faults
    for windows in sensor_faults_list:
        with torch.no_grad():
            for win in windows:
                x = torch.tensor(win[np.newaxis], dtype=torch.float32).to(device)
                recon = vae(x)[0].cpu().numpy().squeeze()
                mse = np.mean((win - recon) ** 2)
                if mse <= mse_threshold:
                    y_pred.append(0)  # Incorrectly as Normal (FN)
                else:
                    recon_error = (win - recon) ** 2
                    stacked = np.stack([win, recon_error], axis=0)
                    x_cnn = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0).to(cnn_device)
                    pred_cnn = cnn(x_cnn).argmax(1).item()
                    y_pred.append(pred_cnn + 1)
                y_true.append(1)

    # Structural Damage
    for windows in structural_damage_list:
        with torch.no_grad():
            for win in windows:
                x = torch.tensor(win[np.newaxis], dtype=torch.float32).to(device)
                recon = vae(x)[0].cpu().numpy().squeeze()
                mse = np.mean((win - recon) ** 2)
                if mse <= mse_threshold:
                    y_pred.append(0)  # Incorrectly as Normal (FN)
                else:
                    recon_error = (win - recon) ** 2
                    stacked = np.stack([win, recon_error], axis=0)
                    x_cnn = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0).to(cnn_device)
                    pred_cnn = cnn(x_cnn).argmax(1).item()
                    y_pred.append(pred_cnn + 1)
                y_true.append(2)

    # --- Results & Visualization ---
    print("\n--- FINAL PIPELINE TEST RESULTS ---")
    report_str = classification_report(
        y_true, y_pred,
        target_names=class_names()
    )
    print(report_str)
    with open(os.path.join(PLOT_DIR, "final_test_report.txt"), "w") as f:
        f.write(report_str)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    # Remove colorbar by setting cbar=False
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=class_names(), yticklabels=class_names()
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Full Pipeline Test)")
    plt.tight_layout()
    cm_path = os.path.join(PLOT_DIR, "final_test_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.show()
    print(f"Final test confusion matrix saved to: {cm_path}")

if __name__ == "__main__":
    main()

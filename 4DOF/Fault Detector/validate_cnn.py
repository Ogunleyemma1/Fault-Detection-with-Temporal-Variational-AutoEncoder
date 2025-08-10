import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from cnn_model import CNN, SEQ_LEN, NUM_FEATURES
from temporal_vae import VAE
import os
import random

# --- Config ---
VAL_START, VAL_END = 0.4, 0.7
BATCH_SIZE = 500
DROPOUT = 0.5
SEED = 42
MODEL_PARAMS = dict(input_dim=12, latent_dim=16, hidden_dim=128, num_layers=2, dropout=0.3)
FAULT_DIR = "data_generation/faults"
PLOT_DIR = "CNN_Validation_Plots"
MODEL_PATH = "cnn_model.pt"    # Changed to match training script

os.makedirs(PLOT_DIR, exist_ok=True)

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_vae_and_stats(mean_path="vae_mean.npy", std_path="vae_std.npy", model_path="temporal_vae_model.pt"):
    mean, std = np.load(mean_path), np.load(std_path)
    std[std == 0] = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(**MODEL_PARAMS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, mean, std, device

def preprocess_with_recon(filepath, vae, mean, std, device, start_frac, end_frac):
    df = pd.read_csv(filepath)
    N = len(df); start = int(N * start_frac); end = int(N * end_frac)
    if end - start < SEQ_LEN: return None
    data = df.iloc[start:end].values.astype(np.float32)
    data_norm = (data - mean) / std
    windows = np.stack([data_norm[i:i + SEQ_LEN] for i in range(len(data_norm) - SEQ_LEN + 1)])
    recon_errors = []
    with torch.no_grad():
        for win in windows:
            x = torch.tensor(win[np.newaxis], dtype=torch.float32).to(device)
            recon = vae(x)[0].cpu().numpy().squeeze()
            error = (win - recon) ** 2
            recon_errors.append(error)
    return np.stack([windows, np.stack(recon_errors)], axis=1)

def list_fault_files(fault_dir):
    sensor_faults, structural_damage = [], []
    for root, _, files in os.walk(fault_dir):
        for f in files:
            if f.endswith(".csv") and not f.startswith("deviation_stats"):
                path = os.path.join(root, f)
                if "sensor_faults" in root.replace("\\", "/"):
                    sensor_faults.append(path)
                elif "structural_faults" in root.replace("\\", "/"):
                    structural_damage.append(path)
    return sensor_faults, structural_damage

def main():
    vae, mean, std, device = load_vae_and_stats()
    X_val, y_val = [], []

    sensor_faults, structural_damage = list_fault_files(FAULT_DIR)
    print(f"Found {len(sensor_faults)} sensor fault files and {len(structural_damage)} structural damage files.")

    for fpath in sensor_faults:
        stacked = preprocess_with_recon(fpath, vae, mean, std, device, VAL_START, VAL_END)
        if stacked is not None:
            X_val.append(stacked)
            y_val.append(np.zeros(len(stacked), dtype=int))
    for fpath in structural_damage:
        stacked = preprocess_with_recon(fpath, vae, mean, std, device, VAL_START, VAL_END)
        if stacked is not None:
            X_val.append(stacked)
            y_val.append(np.ones(len(stacked), dtype=int))

    if not X_val:
        print("ERROR: No data found for validation!")
        return

    X_val, y_val = np.concatenate(X_val), np.concatenate(y_val)
    print(f"Validation dataset: {X_val.shape}")

    dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Load the trained CNN model ---
    model = CNN(input_channels=2, dropout_rate=DROPOUT).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            preds = model(xb).argmax(1).cpu().numpy()
            y_true.extend(yb.numpy())
            y_pred.extend(preds)

    # Classification report
    report_str = classification_report(
        y_true, y_pred,
        target_names=["Sensor Fault", "Structural Damage"],
        zero_division=0
    )
    print("\nCNN Validation Set Classification Report:")
    print(report_str)
    with open(os.path.join(PLOT_DIR, "cnn_validation_report.txt"), "w") as f:
        f.write(report_str)

    # Confusion matrix plot (no color bar)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Sensor Fault", "Structural Damage"],
        yticklabels=["Sensor Fault", "Structural Damage"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (CNN Validation)")
    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, "cnn_validation_confusion_matrix.png")
    plt.savefig(plot_path)
    plt.show()

if __name__ == "__main__":
    main()

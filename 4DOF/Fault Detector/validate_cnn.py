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

VAL_START = 0.4  # Start at 40%
VAL_END = 0.7    # End at 70%
BATCH_SIZE = 500
DROPOUT = 0.3
SEED = 42

MODEL_PARAMS = dict(input_dim=12, latent_dim=16, hidden_dim=128, num_layers=2, dropout=0.3)
FAULT_DIR = "data_generation/faults"
PLOT_DIR = "CNN_Validation_Plots"

os.makedirs(PLOT_DIR, exist_ok=True)

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
    N = len(df)
    start = int(N * start_frac)
    end = int(N * end_frac)
    if end - start < SEQ_LEN:
        print(f"WARNING: {filepath} too short for required window.")
        return None
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
    recon_errors = np.stack(recon_errors)
    stacked = np.stack([windows, recon_errors], axis=1)  # (N, 2, SEQ_LEN, FEATURES)
    return stacked

def list_fault_files(fault_dir):
    sensor_faults = []
    structural_faults = []
    for root, dirs, files in os.walk(fault_dir):
        for f in files:
            if f.endswith(".csv") and not f.startswith("deviation_stats"):
                path = os.path.join(root, f)
                if "sensor_faults" in root.replace("\\", "/"):
                    sensor_faults.append(path)
                elif "structural_faults" in root.replace("\\", "/"):
                    structural_faults.append(path)
    return sensor_faults, structural_faults

def main():
    vae, mean, std, device = load_vae_and_stats()
    X, y = [], []

    # Find all fault files (sensor/structural)
    sensor_faults, structural_faults = list_fault_files(FAULT_DIR)

    print(f"Found {len(sensor_faults)} sensor fault files and {len(structural_faults)} structural fault files.")

    # Sensor faults: label = 0
    for fpath in sensor_faults:
        print(f"Processing sensor fault: {fpath}")
        stacked = preprocess_with_recon(fpath, vae, mean, std, device, VAL_START, VAL_END)
        if stacked is not None:
            X.append(stacked)
            y.append(np.zeros(len(stacked), dtype=int))

    # Structural faults: label = 1
    for fpath in structural_faults:
        print(f"Processing structural fault: {fpath}")
        stacked = preprocess_with_recon(fpath, vae, mean, std, device, VAL_START, VAL_END)
        if stacked is not None:
            X.append(stacked)
            y.append(np.ones(len(stacked), dtype=int))

    if not X:
        print("ERROR: No data found for validation!")
        return
    X, y = np.concatenate(X), np.concatenate(y)
    print(f"Validation dataset: {X.shape} (X), {y.shape} (y)")
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)  # shuffle=False for determinism
    model = CNN(input_channels=2, dropout=DROPOUT).to(device)
    model.load_state_dict(torch.load("cnn_model.pt", map_location=device))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            outputs = model(xb)
            preds = outputs.argmax(1).cpu().numpy()
            y_true.extend(yb.numpy())
            y_pred.extend(preds)

    # --- Save and print classification report ---
    report_str = classification_report(y_true, y_pred, target_names=["Sensor Fault", "Structural Fault"], zero_division=0)
    print("\nCNN Validation Classification Report:")
    print(report_str)
    with open(os.path.join(PLOT_DIR, "cnn_validation_classification_report.txt"), "w") as f:
        f.write(report_str)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Sensor Fault", "Structural Fault"],
                yticklabels=["Sensor Fault", "Structural Fault"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (CNN Validation)")
    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, "cnn_validation_confusion_matrix.png")
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    print(f"Confusion matrix plot saved to {plot_path}")

if __name__ == "__main__":
    main()

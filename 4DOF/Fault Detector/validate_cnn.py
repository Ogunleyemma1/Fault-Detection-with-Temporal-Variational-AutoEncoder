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

VAL_START = 0.55
VAL_END = 0.75
BATCH_SIZE = 500
DROPOUT = 0.3
SEED = 42

# !!! Ensure these VAE params MATCH what you used for training !!!
MODEL_PARAMS = dict(input_dim=12, latent_dim=8, hidden_dim=64, num_layers=2, dropout=0.3)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Actual fault files in your directory
FAULT_DIR = "data_generation/faults"
FAULT_DATASETS = {
    "Structural": [
        "structural_fault_c1_reduced_10.csv",
        "structural_fault_c1_reduced_30.csv",
        "structural_fault_c1_reduced_50.csv",
        "structural_fault_k2_reduced_50.csv",
        "structural_fault_k2_reduced_70.csv",
        "structural_fault_k2_reduced_80.csv",
    ],
    "Sensor": [
        "sensor_fault_v3_noisy.csv",
        "sensor_fault_x1_zero.csv",
    ]
}

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

def main():
    vae, mean, std, device = load_vae_and_stats()
    X, y = [], []
    for label, files in FAULT_DATASETS.items():
        for fname in files:
            fpath = os.path.join(FAULT_DIR, fname)
            print(f"Processing: {fpath}")
            if not os.path.isfile(fpath):
                print(f"WARNING: File not found: {fpath}")
                continue
            stacked = preprocess_with_recon(fpath, vae, mean, std, device, VAL_START, VAL_END)
            if stacked is not None:
                X.append(stacked)
                label_val = 1 if label == "Structural" else 0
                y.append(np.full(len(stacked), label_val))
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
    print("\nCNN Validation Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Sensor Fault", "Structural Fault"], zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Sensor Fault", "Structural Fault"],
                yticklabels=["Sensor Fault", "Structural Fault"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (CNN Validation)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

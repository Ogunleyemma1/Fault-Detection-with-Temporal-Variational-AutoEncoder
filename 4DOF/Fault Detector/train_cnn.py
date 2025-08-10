import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- Import your models (ensure these exist in your directory!) ---
from cnn_model import CNN, SEQ_LEN, NUM_FEATURES
from temporal_vae import VAE

# ========== CONFIGURATION ==========
# File/folder locations
FAULT_DIR = "data_generation/faults"
PLOT_DIR = "CNN_Training_Plots"
BEST_MODEL_PATH = "cnn_model.pt"   # Will save in the working directory

# Training and model parameters
CNN_EPOCHS = 50
BATCH_SIZE = 100
LEARNING_RATE = 0.0001
DROPOUT = 0.5          # Dropout rate for CNN model
WEIGHT_DECAY = 5e-5    # L2 regularization
EARLY_STOPPING_PATIENCE = 15
SEED = 42

# VAE model configuration (update if you change your VAE!)
MODEL_PARAMS = dict(input_dim=12, latent_dim=16, hidden_dim=128, num_layers=2, dropout=0.3)

# Data split fractions for training
TRAIN_START, TRAIN_END = 0.0, 0.4

# ================== SETUP FOR REPRODUCIBILITY ==================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Ensure plot directory exists
os.makedirs(PLOT_DIR, exist_ok=True)

# ========== FUNCTION DEFINITIONS ==========

def load_vae_and_stats(mean_path="vae_mean.npy", std_path="vae_std.npy", model_path="temporal_vae_model.pt"):
    """
    Loads a trained VAE model and normalization statistics.
    """
    mean, std = np.load(mean_path), np.load(std_path)
    std[std == 0] = 1e-6  # Avoid divide-by-zero
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(**MODEL_PARAMS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, mean, std, device

def preprocess_with_recon(filepath, vae, mean, std, device, start_frac, end_frac):
    """
    Loads a CSV file, applies normalization, runs through the VAE, and stacks input+reconstruction error.
    Returns: ndarray of shape [num_windows, 2, SEQ_LEN, NUM_FEATURES]
    """
    df = pd.read_csv(filepath)
    N = len(df)
    start = int(N * start_frac)
    end = int(N * end_frac)
    if end - start < SEQ_LEN:
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
    # Stack shape: [num_windows, 2, SEQ_LEN, NUM_FEATURES]
    return np.stack([windows, np.stack(recon_errors)], axis=1)

def list_fault_files(fault_dir):
    """
    Traverses the fault_dir to return lists of sensor and structural fault csv files.
    """
    sensor_faults, structural_faults = [], []
    for root, _, files in os.walk(fault_dir):
        for f in files:
            if f.endswith(".csv") and not f.startswith("deviation_stats"):
                path = os.path.join(root, f)
                root_slash = root.replace("\\", "/")
                if "sensor_faults" in root_slash:
                    sensor_faults.append(path)
                elif "structural_faults" in root_slash:
                    structural_faults.append(path)
    return sensor_faults, structural_faults

# ========== MAIN TRAINING SCRIPT ==========

def main():
    # --- Load VAE model and normalization statistics ---
    vae, mean, std, device = load_vae_and_stats()
    
    # --- Gather and preprocess all data windows ---
    X_pool, y_pool = [], []
    sensor_faults, structural_faults = list_fault_files(FAULT_DIR)
    print(f"[INFO] Found {len(sensor_faults)} sensor fault files and {len(structural_faults)} structural fault files.")

    for fpath in sensor_faults:
        stacked = preprocess_with_recon(fpath, vae, mean, std, device, TRAIN_START, TRAIN_END)
        if stacked is not None:
            X_pool.append(stacked)
            y_pool.append(np.zeros(len(stacked), dtype=int))  # Label: 0 = sensor fault

    for fpath in structural_faults:
        stacked = preprocess_with_recon(fpath, vae, mean, std, device, TRAIN_START, TRAIN_END)
        if stacked is not None:
            X_pool.append(stacked)
            y_pool.append(np.ones(len(stacked), dtype=int))   # Label: 1 = structural fault

    if not X_pool:
        print("[ERROR] No dataset windows were loaded. Exiting.")
        return

    X_all = np.concatenate(X_pool)
    y_all = np.concatenate(y_pool)
    print(f"[INFO] Training dataset shape: {X_all.shape} (windows, input+error, seq, features)")

    # --- Split dataset for training and validation ---
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all
    )

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                               torch.tensor(y_val, dtype=torch.long))
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)

    # --- Initialize and configure the CNN classifier ---
    model = CNN(input_channels=2, dropout_rate=DROPOUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)  # Balance if needed
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # --- Training loop with early stopping ---
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print("[INFO] Starting CNN training...")

    for epoch in range(CNN_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = loss_fn(output, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        history['train_loss'].append(train_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                output = model(xb)
                loss = loss_fn(output, yb)
                val_loss += loss.item()
                val_correct += (output.argmax(1) == yb).sum().item()
                val_total += yb.size(0)
        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = val_correct / val_total
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)

        print(f"Epoch {epoch+1:02d}: Train Loss={history['train_loss'][-1]:.4f} | "
              f"Val Loss={val_epoch_loss:.4f} | Val Acc={val_epoch_acc:.4f}")

        # Early stopping & model checkpointing
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  [INFO] Validation loss improved. Model saved to '{BEST_MODEL_PATH}'")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"[INFO] Early stopping at epoch {epoch+1}.")
            break

    print(f"[INFO] Training finished. Best model saved to '{BEST_MODEL_PATH}' with val loss: {best_val_loss:.4f}")

    # --- Plot loss curves ---
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CNN Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(PLOT_DIR, "cnn_training_curves.png")
    plt.savefig(plot_path)
    print(f"[INFO] Training/validation loss plot saved to '{plot_path}'.")
    plt.close()

if __name__ == "__main__":
    main()

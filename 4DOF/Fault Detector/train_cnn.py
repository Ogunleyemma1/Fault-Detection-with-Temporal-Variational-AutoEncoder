import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt
from cnn_model import CNN, SEQ_LEN, NUM_FEATURES
from temporal_vae import VAE
import os

# --- Config ---
CNN_EPOCHS = 50
BATCH_SIZE = 100
LR = 0.0001
DROPOUT = 0.2
TRAIN_START, TRAIN_END = 0.0, 0.4  # Use first 40% of each file for training
MODEL_PARAMS = dict(input_dim=12, latent_dim=16, hidden_dim=128, num_layers=2, dropout=0.3)
SEED = 42
PLOT_DIR = "CNN_Training_Plots"

FAULT_DIR = "data_generation/faults"

os.makedirs(PLOT_DIR, exist_ok=True)

# --- Random seeds for reproducibility ---
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
                # Decide label based on immediate parent folder
                parent = os.path.basename(os.path.dirname(path)).lower()
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
        stacked = preprocess_with_recon(fpath, vae, mean, std, device, TRAIN_START, TRAIN_END)
        if stacked is not None:
            X.append(stacked)
            y.append(np.zeros(len(stacked), dtype=int))

    # Structural faults: label = 1
    for fpath in structural_faults:
        print(f"Processing structural fault: {fpath}")
        stacked = preprocess_with_recon(fpath, vae, mean, std, device, TRAIN_START, TRAIN_END)
        if stacked is not None:
            X.append(stacked)
            y.append(np.ones(len(stacked), dtype=int))

    if not X:
        print("ERROR: No data found!")
        return
    X, y = np.concatenate(X), np.concatenate(y)
    print(f"Final dataset: {X.shape} (X), {y.shape} (y)")
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True)
    model = CNN(input_channels=2, dropout=DROPOUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(CNN_EPOCHS * 0.6), gamma=0.2)
    
    train_losses = []
    train_accs = []

    for epoch in range(CNN_EPOCHS):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = loss_fn(output, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            correct += (output.argmax(1) == yb).sum().item()
            total += yb.size(0)
        acc = correct / total
        train_losses.append(epoch_loss / len(dataloader))
        train_accs.append(acc)
        scheduler.step()
        print(f"Epoch {epoch + 1}/{CNN_EPOCHS}: Loss = {train_losses[-1]:.4f}, Accuracy = {acc:.3f}")

    torch.save(model.state_dict(), "cnn_model.pt")
    print("Saved CNN model to cnn_model.pt")

    # --- Training curve plot ---
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid()
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Curve')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, 'cnn_training_curves.png')
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    print(f"Training plot saved to {plot_path}")

if __name__ == "__main__":
    main()

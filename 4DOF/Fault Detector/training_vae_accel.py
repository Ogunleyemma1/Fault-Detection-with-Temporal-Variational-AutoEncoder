import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# === DATA LOADING ===
HEALTHY_DIR = "data_generation/healthy_runs"
accel_cols = ['a1', 'a2', 'a3', 'a4']

healthy_files = [os.path.join(HEALTHY_DIR, f) for f in os.listdir(HEALTHY_DIR)
                 if f.startswith("healthy_seed") and f.endswith(".csv")]
dfs = [pd.read_csv(f) for f in healthy_files]
all_healthy = pd.concat(dfs, axis=0, ignore_index=True)

accel_data = all_healthy[accel_cols].values.astype(np.float32)
print(f"Loaded {accel_data.shape[0]} samples from {len(healthy_files)} files (only a1–a4).")

# Plot raw acceleration
plt.figure(figsize=(12,4))
for i, c in enumerate(accel_cols):
    plt.plot(accel_data[:, i], label=c, alpha=0.7)
plt.legend()
plt.title("Raw Acceleration Time Series (all data)")
plt.savefig("plot_raw_accelerations.png")
plt.close()

# === NORMALIZATION ===
accel_mean = accel_data.mean(axis=0)
accel_std = accel_data.std(axis=0)
np.save("vae_accel_mean.npy", accel_mean)
np.save("vae_accel_std.npy", accel_std)
print("Mean:", accel_mean)
print("Std:", accel_std)
accel_data_norm = (accel_data - accel_mean) / (accel_std + 1e-8)

# === WINDOWING ===
WINDOW = 100
windows = []
for i in range(accel_data_norm.shape[0] - WINDOW + 1):
    windows.append(accel_data_norm[i:i+WINDOW])
windows = np.stack(windows)
print("Accel windows shape:", windows.shape)

# Plot first 3 normalized windows
for k in range(3):
    plt.figure(figsize=(8,2))
    for i, c in enumerate(accel_cols):
        plt.plot(windows[k,:,i], label=c)
    plt.legend()
    plt.title(f"Normalized Window {k}")
    plt.savefig(f"plot_window_{k}.png")
    plt.close()

# === TRAIN/VAL SPLIT ===
N = len(windows)
np.random.seed(42)
perm = np.random.permutation(N)
windows = windows[perm]
train_N = int(0.75 * N)
train_windows = windows[:train_N]
val_windows = windows[train_N:]
print(f"Train: {train_windows.shape}, Val: {val_windows.shape}")

# === TORCH DATA ===
train_tensor = torch.tensor(train_windows, dtype=torch.float32)
val_tensor = torch.tensor(val_windows, dtype=torch.float32)

# === MODEL ===
class VAE(nn.Module):
    def __init__(self, input_dim=4*WINDOW, latent_dim=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def decode(self, z):
        return self.decoder(z)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# === LOSS ===
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss

# === TRAINING ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = VAE().to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)
BATCH = 64
EPOCHS = 50

train_losses, train_recon, train_kl = [], [], []
val_losses, val_recon, val_kl = [], [], []

for epoch in range(1, EPOCHS+1):
    vae.train()
    idx = np.random.permutation(train_tensor.shape[0])
    batches = [idx[i:i+BATCH] for i in range(0, len(idx), BATCH)]
    tloss = trecon = tkl = 0.0
    for b in batches:
        x = train_tensor[b].reshape(len(b), -1).to(device)
        optimizer.zero_grad()
        recon, mu, logvar = vae(x)
        loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        tloss += loss.item() * len(b)
        trecon += recon_loss.item() * len(b)
        tkl += kl_loss.item() * len(b)
    tloss /= len(idx); trecon /= len(idx); tkl /= len(idx)
    train_losses.append(tloss)
    train_recon.append(trecon)
    train_kl.append(tkl)
    # --- VAL ---
    vae.eval()
    with torch.no_grad():
        x = val_tensor.reshape(val_tensor.shape[0], -1).to(device)
        recon, mu, logvar = vae(x)
        loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar)
        val_losses.append(loss.item())
        val_recon.append(recon_loss.item())
        val_kl.append(kl_loss.item())
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | Train: {tloss:.4f} (Recon: {trecon:.4f}, KL: {tkl:.4f}) | Val: {val_losses[-1]:.4f} (Recon: {val_recon[-1]:.4f}, KL: {val_kl[-1]:.4f})")

# === SAVE MODEL ===
torch.save(vae.state_dict(), "temporal_vae_model_accel.pt")
print("Model saved to temporal_vae_model_accel.pt")

# === PLOT TRAINING CURVES ===
plt.figure(figsize=(12,6))
plt.plot(train_losses, label="Train Total Loss")
plt.plot(train_recon, label="Train Recon Loss")
plt.plot(train_kl, label="Train KL Loss")
plt.plot(val_losses, "--", label="Val Total Loss")
plt.plot(val_recon, "--", label="Val Recon Loss")
plt.plot(val_kl, "--", label="Val KL Loss")
plt.title("VAE Training/Validation Loss (Accelerations Only)")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("vae_accel_loss_curves.png")
plt.close()

# === SAMPLE RECONSTRUCTION ===
vae.eval()
with torch.no_grad():
    for i in range(3):
        sample = val_tensor[i]
        recon, mu, logvar = vae(sample.reshape(1, -1).to(device))   # <-- FIXED: use reshape!
        plt.figure(figsize=(12,2))
        for j in range(4):
            plt.plot(sample[:,j].cpu(), alpha=0.4)
            plt.plot(recon[0].cpu().numpy().reshape(WINDOW, 4)[:,j], '--', alpha=0.7)
        plt.title(f"Val Sample {i}: Input vs Recon\nmu={np.round(mu.cpu().numpy()[0],4)}, logvar={np.round(logvar.cpu().numpy()[0],4)}")
        plt.legend(['Input a1', 'Recon a1', 'Input a2', 'Recon a2', 'Input a3', 'Recon a3', 'Input a4', 'Recon a4'])
        plt.tight_layout()
        plt.savefig(f"vae_accel_recon_sample_{i}.png")
        plt.close()

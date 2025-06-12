import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from temporal_vae import VAE

def test_vae(seq_len=100):
    df = pd.read_csv("vae_input_data.csv")
    data = df.values.astype(np.float32)
    mean = np.load("vae_mean.npy")
    std = np.load("vae_std.npy")
    data_norm = (data - mean) / std

    sequences = [data_norm[i:i + seq_len] for i in range(len(data_norm) - seq_len + 1)]
    sequences = np.stack(sequences)
    x = torch.tensor(sequences, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE(input_dim=12, latent_dim=16, hidden_dim=128, num_layers=2, dropout=0.3).to(device)
    model.load_state_dict(torch.load("temporal_vae_model.pt", map_location=device))
    model.eval()

    with torch.no_grad():
        x = x.to(device)
        recon, mu, logvar = model(x)
        recon_np = recon.cpu().numpy()

    full_len = len(data_norm)
    full_recon = np.zeros_like(data_norm)
    count = np.zeros((full_len, 1))

    for i in range(len(recon_np)):
        full_recon[i:i + seq_len] += recon_np[i]
        count[i:i + seq_len] += 1

    count[count == 0] = 1
    full_recon /= count

    recon_denorm = (full_recon * std) + mean
    pd.DataFrame(recon_denorm, columns=df.columns).to_csv("vae_reconstruction.csv", index=False)
    print("Reconstruction saved to vae_reconstruction.csv")

    time = np.arange(len(data_norm))
    num_dofs = data.shape[1] // 3
    var_labels = ['x', 'v', 'a']
    colors = ['tab:blue', 'tab:orange']

    for i, var in enumerate(var_labels):
        fig, axs = plt.subplots(num_dofs, 1, figsize=(14, 2.5 * num_dofs), sharex=True)
        if num_dofs == 1:
            axs = [axs]
        for dof in range(num_dofs):
            idx = i * num_dofs + dof
            axs[dof].plot(time, data_norm[:, idx], label="Normalized Input", color=colors[0])
            axs[dof].plot(time, full_recon[:, idx], label="Normalized Recon", color=colors[1], alpha=0.8)
            axs[dof].set_title(f"DOF {dof+1} - Normalized Comparison - {var}{dof+1}")
            axs[dof].legend()
            axs[dof].grid(True)
        axs[-1].set_xlabel("Time Step")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    test_vae(seq_len=100)

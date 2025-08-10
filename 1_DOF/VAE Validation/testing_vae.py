import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from temporal_vae import VAE

def canonical_variant(name):
    name = name.lower()
    if "drift" in name:
        return "Drifted"
    elif "amplitude" in name:
        return "Amplitude"
    elif "lowfreq" in name:  # Only lowfreq!
        return "LowFreq"
    elif "original" in name:
        return "Original"
    else:
        return None

def test_vae_all(seq_len=80, segment_len=100):
    print("🧪 Testing Temporal VAE on 1DOF variants: comparison + MSE stats...")

    df = pd.read_csv("1DOF_signal_clean_variants.csv")
    df = df.iloc[int(0.5 * len(df)):]  # use second half
    time = df["time"].values
    data = df.drop(columns=["time"]).values.astype(np.float32)

    # Normalize
    mean = np.load("vae_mean.npy")
    std = np.load("vae_std.npy")
    data_norm = (data - mean) / std

    # Build overlapping sequences
    sequences = [data_norm[i:i + seq_len] for i in range(len(data_norm) - seq_len + 1)]
    sequences = np.stack(sequences)
    x = torch.tensor(sequences, dtype=torch.float32)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dim=12, latent_dim=5, hidden_dim=32, num_layers=2, dropout=0.2).to(device)
    model.load_state_dict(torch.load("temporal_vae_model.pt", map_location=device))
    model.eval()

    # Inference
    with torch.no_grad():
        x = x.to(device)
        recon, mu, logvar = model(x)
        recon_np = recon.cpu().numpy()
        mu_np = mu.cpu().numpy()

    # Reconstruct full signal from overlapping segments
    full_len = len(data_norm)
    full_recon = np.zeros_like(data_norm)
    count = np.zeros((full_len, 1))
    for i in range(len(recon_np)):
        full_recon[i:i + seq_len] += recon_np[i]
        count[i:i + seq_len] += 1
    count[count == 0] = 1
    full_recon /= count

    # Denormalize
    recon_denorm = (full_recon * std) + mean
    recon_df = pd.DataFrame(recon_denorm, columns=df.columns[1:])
    recon_df.to_csv("vae_reconstruction.csv", index=False)
    print("✅ Reconstruction saved to vae_reconstruction.csv")

    # === Create plot folder ===
    os.makedirs("VAE Test Plot", exist_ok=True)

    col_names = df.columns[1:]
    groups = {
        "Displacement": [col for col in col_names if col.startswith("x_")],
        "Velocity": [col for col in col_names if col.startswith("v_")],
        "Acceleration": [col for col in col_names if col.startswith("a_")]
    }

    mse_by_variant = {"Original": [], "Drifted": [], "Amplitude": [], "LowFreq": []}
    colors = {"Original": "#1f77b4", "Drifted": "#ff7f0e", "Amplitude": "#2ca02c", "LowFreq": "#d62728"}

    # === Plot: Original vs Reconstructed ===
    for group_name, cols in groups.items():
        fig, axs = plt.subplots(len(cols), 1, figsize=(12, 8), sharex=True)
        for i, col in enumerate(cols):
            idx = df.columns.get_loc(col) - 1
            axs[i].plot(time, data[:, idx], label="Original", color="blue")
            axs[i].plot(time, recon_denorm[:, idx], label="Reconstructed", color="orange", alpha=0.7)
            axs[i].set_title(f"{group_name} - {col}")
            axs[i].set_ylabel("Value")
            axs[i].legend()
            axs[i].grid(True)

            variant = canonical_variant(col)
            if variant is None:
                continue  # skip if mapping not found

            original = data[:, idx]
            reconstructed = recon_denorm[:, idx]
            num_segments = len(original) // segment_len
            mse_segments = [
                np.mean((original[j*segment_len:(j+1)*segment_len] -
                         reconstructed[j*segment_len:(j+1)*segment_len]) ** 2)
                for j in range(num_segments)
            ]
            mse_by_variant[variant].extend(mse_segments)

        axs[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig(f"VAE Test Plot/{group_name.lower()}_comparison.png")
        plt.show()
        plt.close()

    # === Always plot all four variants ===
    variant_names = ["Original", "Drifted", "Amplitude", "LowFreq"]
    bar_colors = [colors[v] for v in variant_names]
    means = [np.mean(mse_by_variant[v]) if len(mse_by_variant[v]) > 0 else 0 for v in variant_names]

    plt.figure(figsize=(9, 6))
    plt.bar(variant_names, means, color=bar_colors)
    plt.xlabel("Signal Variant")
    plt.ylabel("Mean Segment-wise MSE")
    plt.title("Mean Segment-wise MSE by Signal Variant (Seen Signals)")
    plt.tight_layout()
    plt.savefig("VAE Test Plot/segment_mse_barplot_seen.png", dpi=150)
    plt.show()
    print("🎉 All reconstruction plots and MSE distribution saved in 'VAE Test Plot/'.")

    # === PCA Latent Space Visualization ===
    variant_labels = []
    for seq in sequences:
        col_idx = np.argmax(np.abs(seq[0]))
        col_name = col_names[col_idx]
        variant = canonical_variant(col_name)
        variant_labels.append(variant if variant is not None else "Unknown")
    variant_labels = np.array(variant_labels)
    keep_mask = np.isin(variant_labels, variant_names)
    mu_np_plot = mu_np[keep_mask]
    variant_labels_plot = variant_labels[keep_mask]

    pca = PCA(n_components=2)
    mu_pca = pca.fit_transform(mu_np_plot)

    plt.figure(figsize=(8, 6))
    for v in variant_names:
        idx = variant_labels_plot == v
        plt.scatter(mu_pca[idx, 0], mu_pca[idx, 1], label=v, color=colors[v], s=14, alpha=0.75)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Latent Space PCA by Signal Type (Seen 1DOF)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("VAE Test Plot/latent_space_pca_1dof_seen.png", dpi=150)
    plt.show()
    print("✅ Latent space PCA plot saved to VAE Test Plot/latent_space_pca_1dof_seen.png")

# === Run ===
if __name__ == "__main__":
    test_vae_all(seq_len=80, segment_len=100)

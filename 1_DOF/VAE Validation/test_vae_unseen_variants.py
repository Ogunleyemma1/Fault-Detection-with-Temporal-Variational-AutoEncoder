import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from temporal_vae import VAE

def test_vae_unseen(seq_len=80, segment_len=100):
    print("🧪 Testing Temporal VAE on unseen signals...")

    df = pd.read_csv("1DOF_signal_unseen_variants.csv")
    time = df["time"].values
    data = df.drop(columns=["time"]).values.astype(np.float32)

    # Normalize using training stats
    mean = np.load("vae_mean.npy")
    std = np.load("vae_std.npy")
    data_norm = (data - mean) / std

    # Create overlapping sequences
    sequences = [data_norm[i:i + seq_len] for i in range(len(data_norm) - seq_len + 1)]
    sequences = np.stack(sequences)
    x = torch.tensor(sequences, dtype=torch.float32)

    # Load trained VAE model
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

    # Stitch overlapping sequences
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
    pd.DataFrame(recon_denorm, columns=df.columns[1:]).to_csv("vae_reconstruction_unseen.csv", index=False)
    print("✅ Unseen signal reconstruction saved to vae_reconstruction_unseen.csv")

    os.makedirs("VAE Unseen Test Plot", exist_ok=True)

    col_names = df.columns[1:]
    groups = {
        "Displacement": [col for col in col_names if col.startswith("x_")],
        "Velocity": [col for col in col_names if col.startswith("v_")],
        "Acceleration": [col for col in col_names if col.startswith("a_")]
    }

    mse_stats = []
    mse_by_variant = {}

    # List of variant names (should match column suffixes after "_")
    variants = ["original", "envelope", "triangle", "square"]
    variant_colors = {
        "original": "#1f77b4", "envelope": "#ff7f0e",
        "triangle": "#2ca02c", "square": "#d62728"
    }

    # === Plot original vs reconstructed and collect MSE stats ===
    for group_name, cols in groups.items():
        fig, axs = plt.subplots(len(cols), 1, figsize=(12, 8), sharex=True)
        for i, col in enumerate(cols):
            idx = df.columns.get_loc(col) - 1
            original = data[:, idx]
            reconstructed = recon_denorm[:, idx]

            axs[i].plot(time, original, label="Original", color="blue")
            axs[i].plot(time, reconstructed, label="Reconstructed", color="orange", alpha=0.7)
            axs[i].set_title(f"{group_name} - {col}")
            axs[i].set_ylabel("Value")
            axs[i].legend()
            axs[i].grid(True)

            # Segment-wise MSE computation, separated by variant
            variant = col.split("_")[-1]
            num_segments = len(original) // segment_len
            mse_segments = [
                np.mean((original[j*segment_len:(j+1)*segment_len] -
                         reconstructed[j*segment_len:(j+1)*segment_len])**2)
                for j in range(num_segments)
            ]
            mse_stats.append({
                "Signal": col,
                "Group": group_name,
                "Variant": variant,
                "Min_MSE": np.min(mse_segments),
                "Max_MSE": np.max(mse_segments),
                "Mean_MSE": np.mean(mse_segments),
                "Std_MSE": np.std(mse_segments)
            })
            # Store for barplot
            if variant not in mse_by_variant:
                mse_by_variant[variant] = []
            mse_by_variant[variant].extend(mse_segments)

        axs[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig(f"VAE Unseen Test Plot/{group_name.lower()}_comparison_unseen.png")
        plt.show()
        plt.close()

    # Save MSE statistics
    pd.DataFrame(mse_stats).to_csv("VAE Unseen Test Plot/vae_segment_mse_stats_unseen.csv", index=False)
    print("📄 MSE stats saved to VAE Unseen Test Plot/vae_segment_mse_stats_unseen.csv")

    # --- Clean bar chart: mean MSE for each variant (log y scale, no error bars, no grid) ---
    means = []
    for variant in variants:
        mse_vals = mse_by_variant.get(variant, [])
        means.append(np.mean(mse_vals) if len(mse_vals) else np.nan)
    x = np.arange(len(variants))

    plt.figure(figsize=(8, 5))
    bars = plt.bar(x, means, color=[variant_colors[v] for v in variants], alpha=0.85, tick_label=[v.capitalize() for v in variants])
    plt.ylabel("Mean Segment-wise MSE", fontsize=13)
    plt.yscale("log")
    plt.xlabel("Signal Variant", fontsize=13)
    plt.title("Mean Segment-wise MSE by Signal Variant (Unseen Signals, Log Scale)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.1)
    plt.gca().spines['bottom'].set_linewidth(1.1)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("VAE Unseen Test Plot/segment_mse_barplot_unseen_log.png", dpi=150)
    plt.show()
    print("🎉 Clean unseen signal MSE bar chart saved in 'VAE Unseen Test Plot/'")

    # === Latent Space PCA Visualization ===
    # Assign variant label for each window based on max-abs col in first row
    variant_labels = []
    for seq in sequences:
        col_idx = np.argmax(np.abs(seq[0]))
        col_name = col_names[col_idx]
        variant = col_name.split("_")[-1]
        variant_labels.append(variant)
    variant_labels = np.array(variant_labels)
    keep_mask = np.isin(variant_labels, variants)
    mu_np_plot = mu_np[keep_mask]
    variant_labels_plot = variant_labels[keep_mask]

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    mu_pca = pca.fit_transform(mu_np_plot)

    plt.figure(figsize=(8, 6))
    for v in variants:
        idx = variant_labels_plot == v
        plt.scatter(mu_pca[idx, 0], mu_pca[idx, 1], label=v.capitalize(), color=variant_colors[v], s=14, alpha=0.75)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Latent Space PCA by Signal Type (Unseen 1DOF)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("VAE Unseen Test Plot/latent_space_pca_1dof_unseen.png", dpi=150)
    plt.show()
    print("✅ Latent space PCA plot saved to VAE Unseen Test Plot/latent_space_pca_1dof_unseen.png")

# === Run ===
if __name__ == "__main__":
    test_vae_unseen(seq_len=80, segment_len=100)

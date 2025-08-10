import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from temporal_vae import VAE

def test_vae_unseen_2dof(seq_len=100, segment_len=100):
    print("🧪 Testing Temporal VAE on 2DOF unseen signals...")

    # === Load data ===
    df = pd.read_csv("2DOF_signal_unseen_variants.csv")
    time = df["time"].values
    data = df.drop(columns=["time"]).values.astype(np.float32)

    # Load normalization from training
    mean = np.load("vae_mean.npy")
    std = np.load("vae_std.npy")
    data_norm = (data - mean) / std

    # Build overlapping sequences
    sequences = [data_norm[i:i + seq_len] for i in range(len(data_norm) - seq_len + 1)]
    sequences = np.stack(sequences)
    x = torch.tensor(sequences, dtype=torch.float32)

    # Load trained VAE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dim=data.shape[1], latent_dim=5, hidden_dim=128, num_layers=2, dropout=0.3).to(device)
    model.load_state_dict(torch.load("temporal_vae_model.pt", map_location=device))
    model.eval()

    # Inference
    with torch.no_grad():
        x = x.to(device)
        recon, mu, logvar = model(x)
        recon_np = recon.cpu().numpy()

    # === Reconstruct full signal ===
    full_len = len(data_norm)
    full_recon = np.zeros_like(data_norm)
    count = np.zeros((full_len, 1))
    for i in range(len(recon_np)):
        full_recon[i:i + seq_len] += recon_np[i]
        count[i:i + seq_len] += 1
    count[count == 0] = 1
    full_recon /= count

    # Denormalize and save
    recon_denorm = (full_recon * std) + mean
    pd.DataFrame(recon_denorm, columns=df.columns[1:]).to_csv("vae_reconstruction_unseen_2dof.csv", index=False)
    print("✅ Unseen 2DOF signal reconstruction saved to vae_reconstruction_unseen_2dof.csv")

    os.makedirs("VAE Unseen Test Plot 2DOF", exist_ok=True)

    # === Column groupings by variable and DOF ===
    col_names = df.columns[1:]  # Exclude time
    dofs = [1, 2]
    groups = {}
    for dof in dofs:
        groups[f"Displacement DOF{dof}"] = [col for col in col_names if col.startswith(f"x{dof}_")]
        groups[f"Velocity DOF{dof}"] = [col for col in col_names if col.startswith(f"v{dof}_")]
        groups[f"Acceleration DOF{dof}"] = [col for col in col_names if col.startswith(f"a{dof}_")]

    # === MSE analysis ===
    mse_stats = []
    mse_by_variant = {}

    variants = ["original", "envelope", "triangle", "square"]
    variant_colors = {"original": "#1f77b4", "envelope": "#ff7f0e", "triangle": "#2ca02c", "square": "#d62728"}

    # === Original vs reconstructed plots + collect MSE ===
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

            # Compute segment-wise MSE by variant
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
            if variant not in mse_by_variant:
                mse_by_variant[variant] = []
            mse_by_variant[variant].extend(mse_segments)
        axs[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig(f"VAE Unseen Test Plot 2DOF/{group_name.lower().replace(' ', '_')}_comparison_unseen.png")
        plt.show()
        plt.close()

    # Save MSE stats
    pd.DataFrame(mse_stats).to_csv("VAE Unseen Test Plot 2DOF/vae_segment_mse_stats_unseen_2dof.csv", index=False)
    print("📄 MSE stats saved to VAE Unseen Test Plot 2DOF/vae_segment_mse_stats_unseen_2dof.csv")

    # --- Barplot: mean MSE for each variant (log y) ---
    means = [np.mean(mse_by_variant[v]) if v in mse_by_variant else np.nan for v in variants]
    bar_colors = [variant_colors[v] for v in variants]
    xlabels = [v.capitalize() for v in variants]

    plt.figure(figsize=(8, 5))
    plt.bar(xlabels, means, color=bar_colors, alpha=0.85)
    plt.ylabel("Mean Segment-wise MSE", fontsize=13)
    plt.yscale("log")
    plt.xlabel("Signal Variant", fontsize=13)
    plt.title("Mean Segment-wise MSE by Signal Variant (2DOF Unseen, Log Scale)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.1)
    plt.gca().spines['bottom'].set_linewidth(1.1)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("VAE Unseen Test Plot 2DOF/segment_mse_barplot_unseen_2dof_log.png", dpi=150)
    plt.show()
    print("🎉 Clean unseen signal MSE bar chart saved in 'VAE Unseen Test Plot 2DOF/'")

    # --- Latent space PCA visualization ---
    latent_all = []
    label_all = []
    for i, variant in enumerate(variants):
        # For each variant, mask out other columns
        variant_data = np.zeros_like(data_norm)
        # Find columns containing this variant for all variables/DOFs
        variant_cols = [j for j, name in enumerate(df.columns[1:]) if name.endswith(f"_{variant}")]
        variant_data[:, variant_cols] = data_norm[:, variant_cols]
        # Build sequences
        sub_seqs = [variant_data[i:i + seq_len] for i in range(min(len(variant_data) - seq_len + 1, 500))]
        sub_seqs = torch.tensor(np.stack(sub_seqs), dtype=torch.float32).to(device)
        with torch.no_grad():
            mu, _ = model.encode(sub_seqs)
        latent_all.append(mu.cpu().numpy())
        label_all.extend([variant.capitalize()] * len(mu))
    latent_all = np.concatenate(latent_all, axis=0)
    label_all = np.array(label_all)

    # PCA 2D plot
    latent_2d = PCA(n_components=2).fit_transform(latent_all)
    plt.figure(figsize=(8, 6))
    for i, variant in enumerate(variants):
        idx = label_all == variant.capitalize()
        plt.scatter(latent_2d[idx, 0], latent_2d[idx, 1],
                    label=variant.capitalize(), alpha=0.7, s=13, c=variant_colors[variant])
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Latent Space PCA by Signal Type (Unseen 2DOF)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("VAE Unseen Test Plot 2DOF/latent_space_pca_unseen_2dof.png", dpi=150)
    plt.show()
    print("📊 Latent space PCA plot saved.")

# === Run ===
if __name__ == "__main__":
    test_vae_unseen_2dof(seq_len=100, segment_len=100)

# compare_initial_windows.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_reconstruction_window_comparison():
    # Load original and reconstructed data
    input_df = pd.read_csv("vae_input_data.csv")
    recon_df = pd.read_csv("vae_reconstruction.csv")

    # Time window lengths to analyze
    time_windows = [50, 100, 200]
    dof_labels = input_df.columns.tolist()

    fig, axs = plt.subplots(len(time_windows), 1, figsize=(12, 12), sharex=True)
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    for j, window in enumerate(time_windows):
        true_vals = input_df.iloc[:window].values
        pred_vals = recon_df.iloc[:window].values

        # Compute NMSE: normalized by variance of each DOF
        mse = np.mean((true_vals - pred_vals) ** 2, axis=0)
        var = np.var(true_vals, axis=0) + 1e-8  # avoid division by zero
        nmse = mse / var

        # Standard deviation of normalized squared error
        squared_errors = (true_vals - pred_vals) ** 2
        normalized_errors = squared_errors / var
        std_error = np.std(normalized_errors, axis=0)

        x_indices = np.arange(len(dof_labels))
        axs[j].bar(x_indices, nmse, yerr=std_error, capsize=5, color=colors[j], alpha=0.7)
        axs[j].set_title(f"Normalized MSE (mean ± std) - First {window} Time Steps")
        axs[j].set_ylabel("NMSE")
        axs[j].set_xticks(x_indices)
        axs[j].set_xticklabels(dof_labels, rotation=45)
        axs[j].grid(True)

    axs[-1].set_xlabel("DOF Variables")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_reconstruction_window_comparison()

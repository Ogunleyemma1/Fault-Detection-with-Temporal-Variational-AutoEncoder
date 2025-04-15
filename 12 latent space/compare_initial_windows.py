# compare_initial_windows.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_reconstruction_window_comparison():
    # Load data
    input_df = pd.read_csv("vae_input_data.csv")
    recon_df = pd.read_csv("vae_reconstruction.csv")

    # Time windows to compare
    time_windows = [50, 100, 200]
    dofs = ['x1', 'x2', 'x3', 'x4', 'v1', 'v2', 'v3', 'v4', 'a1', 'a2', 'a3', 'a4']

    # Set up the plot
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    for j, window in enumerate(time_windows):
        errors = input_df.iloc[:window].values - recon_df.iloc[:window].values
        mean_error = np.mean(errors, axis=0)
        std_error = np.std(errors, axis=0)

        x_indices = np.arange(len(dofs))
        axs[j].bar(x_indices, mean_error, yerr=std_error, capsize=5, color=colors[j], alpha=0.7)
        axs[j].set_title(f"Reconstruction Error (mean ± std) - First {window} Time Steps")
        axs[j].set_ylabel("Error")
        axs[j].set_xticks(x_indices)
        axs[j].set_xticklabels(dofs, rotation=45)
        axs[j].grid(True)

    axs[-1].set_xlabel("DOF Variables")
    plt.tight_layout()
    plt.show()

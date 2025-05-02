# compare_initial_windows.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_reconstruction_window_comparison():
    input_df = pd.read_csv("vae_input_data.csv")
    recon_df = pd.read_csv("vae_reconstruction.csv")

    time_windows = [50, 100, 200]
    df = pd.read_csv("vae_reconstruction.csv")
    dof_labels = df.columns.tolist()


    fig, axs = plt.subplots(len(time_windows), 1, figsize=(12, 12), sharex=True)
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    for j, window in enumerate(time_windows):
        errors = input_df.iloc[:window].values - recon_df.iloc[:window].values
        mean_error = np.mean(errors, axis=0)
        std_error = np.std(errors, axis=0)

        x_indices = np.arange(len(dof_labels))
        axs[j].bar(x_indices, mean_error, yerr=std_error, capsize=5, color=colors[j], alpha=0.7)
        axs[j].set_title(f"Reconstruction Error (mean ± std) - First {window} Time Steps")
        axs[j].set_ylabel("Error")
        axs[j].set_xticks(x_indices)
        axs[j].set_xticklabels(dof_labels, rotation=45)
        axs[j].grid(True)

    axs[-1].set_xlabel("DOF Variables")
    plt.tight_layout()
    plt.show()

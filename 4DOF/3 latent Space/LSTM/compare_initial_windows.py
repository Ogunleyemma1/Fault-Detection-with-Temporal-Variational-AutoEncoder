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
    # Explicitly define DOF labels for a 4DOF system
    dof_labels = ['x1', 'x2', 'x3', 'x4', 'v1', 'v2', 'v3', 'v4', 'a1', 'a2', 'a3', 'a4']

    # Create a subplot for each time window
    fig, axs = plt.subplots(len(time_windows), 1, figsize=(12, 12), sharex=True)
    colors = ['tab:blue', 'tab:orange', 'tab:green'] # Using colors from the first snippet

    for j, window in enumerate(time_windows):
        # Extract the corresponding window of data
        # Ensure the columns match the dof_labels order
        true_vals = input_df[dof_labels].iloc[:window].values
        pred_vals = recon_df[dof_labels].iloc[:window].values

        # Compute Mean Squared Error (MSE)
        mse = np.mean((true_vals - pred_vals) ** 2, axis=0)
        
        # Compute Variance of True Values
        # Add a small epsilon to prevent division by zero for NMSE
        var = np.var(true_vals, axis=0) + 1e-8 
        
        # Compute Normalized Mean Squared Error (NMSE)
        nmse = mse / var

        # Compute standard deviation of normalized squared errors
        # This is for the error bars, showing std of the normalized squared error, not just the mean error
        squared_errors = (true_vals - pred_vals) ** 2
        normalized_squared_errors = squared_errors / var
        std_error = np.std(normalized_squared_errors, axis=0) # This is the standard deviation for the error bars

        # Plot bar chart for this window
        x_indices = np.arange(len(dof_labels))
        axs[j].bar(x_indices, nmse, yerr=std_error, capsize=5, color=colors[j], alpha=0.7)
        axs[j].set_title(f"Normalized MSE (mean ± std) - First {window} Time Steps") # Title changed
        axs[j].set_ylabel("NMSE") # Y-label changed
        axs[j].set_xticks(x_indices)
        axs[j].set_xticklabels(dof_labels, rotation=45, ha='right') # Improved label rotation
        axs[j].grid(True)

    axs[-1].set_xlabel("DOF Variables")
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.show()

if __name__ == "__main__":
    plot_reconstruction_window_comparison()
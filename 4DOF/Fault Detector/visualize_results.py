# visualize_results.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None
    print("Warning: scikit-learn not installed. PCA plots will be skipped.")

# --- Configuration (should match testing_vae.py if applicable, e.g., SEQ_LEN) ---
SEQ_LEN = 100 # Used for plotting sample reconstruction time axis
RESULTS_CSV_PATH = "fault_detection_results_per_window.csv"
SAMPLE_RECON_JSON_PATH = "sample_reconstruction_data.json"
# --- End of Configuration ---

def plot_sample_reconstructions_from_json(json_filepath, seq_len):
    """Plots sample reconstructions loaded from a JSON file."""
    if not os.path.exists(json_filepath):
        print(f"Error: Sample reconstruction data file '{json_filepath}' not found.")
        return

    with open(json_filepath, 'r') as f:
        sample_data = json.load(f)

    for dataset_label, data in sample_data.items():
        original_window = np.array(data.get("original"))
        reconstructed_window = np.array(data.get("reconstructed"))

        if original_window is None or reconstructed_window is None or original_window.ndim < 2:
            print(f"Skipping sample reconstruction plot for {dataset_label} due to missing/invalid data.")
            continue
        
        num_channels_to_plot = min(3, original_window.shape[1]) # Plot first few channels
        
        fig, axs = plt.subplots(num_channels_to_plot, 1, figsize=(12, 2 * num_channels_to_plot), sharex=True)
        if num_channels_to_plot == 1: axs = [axs] # Make it iterable

        time_axis_window = np.arange(seq_len) # Assumes seq_len matches window length
        if original_window.shape[0] != seq_len: # Quick check
             time_axis_window = np.arange(original_window.shape[0])


        for ch_idx in range(num_channels_to_plot):
            axs[ch_idx].plot(time_axis_window, original_window[:, ch_idx], label=f'Original Ch {ch_idx+1}')
            axs[ch_idx].plot(time_axis_window, reconstructed_window[:, ch_idx], label=f'Reconstructed Ch {ch_idx+1}', linestyle='--')
            axs[ch_idx].set_ylabel(f'Ch {ch_idx+1}')
            axs[ch_idx].legend(); axs[ch_idx].grid(True)
        axs[-1].set_xlabel('Time step in window')
        fig.suptitle(f'Sample Reconstruction - Dataset: {dataset_label}', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_filename = f"sample_reconstruction_plot_{dataset_label.replace(' ', '_').replace('/', '_')}.png"
        plt.savefig(output_filename)
        print(f"Saved sample reconstruction plot to {output_filename}")
        plt.show()


def plot_latent_space_pca_from_csv(csv_filepath, plot_title_suffix, color_by_key, filename_suffix):
    """Generates and saves a PCA plot of the latent space from a CSV file."""
    if not os.path.exists(csv_filepath):
        print(f"Error: Results CSV file '{csv_filepath}' not found.")
        return
    if PCA is None: return # Skip if sklearn not available

    df_results = pd.read_csv(csv_filepath)
    if df_results.empty or 'mu_latent' not in df_results.columns or color_by_key not in df_results.columns:
        print(f"Skipping PCA plot: '{csv_filepath}' is empty or missing 'mu_latent' or '{color_by_key}' column.")
        return

    # Convert 'mu_latent' string representation of list back to numpy array
    try:
        all_mu_vectors_list = df_results['mu_latent'].apply(lambda x: json.loads(x) if isinstance(x, str) else x).tolist()
        all_mu_vectors = np.array(all_mu_vectors_list, dtype=float)
    except Exception as e:
        print(f"Error converting 'mu_latent' to numpy array: {e}. Skipping PCA plot.")
        return
        
    labels_for_coloring = df_results[color_by_key].tolist()

    if all_mu_vectors.ndim == 1: all_mu_vectors = all_mu_vectors.reshape(-1, 1)
    if all_mu_vectors.shape[0] <= 1 or all_mu_vectors.shape[1] == 0:
        print("Not enough data or latent dimensions for PCA plot.")
        return
    
    if all_mu_vectors.shape[1] > 2:
        pca_obj = PCA(n_components=2)
        mu_proj = pca_obj.fit_transform(all_mu_vectors)
    elif all_mu_vectors.shape[1] == 2:
        mu_proj = all_mu_vectors
    else: # shape[1] == 1
        mu_proj = np.hstack([all_mu_vectors, np.zeros_like(all_mu_vectors)])

    plt.figure(figsize=(12, 8))
    unique_labels = sorted(list(set(labels_for_coloring)))
    colors = plt.cm.get_cmap('viridis', len(unique_labels)) if len(unique_labels) > 10 else plt.cm.get_cmap('tab10', len(unique_labels))

    for i, label in enumerate(unique_labels):
        indices = [idx for idx, l_val in enumerate(labels_for_coloring) if l_val == label]
        if not indices: continue # Skip if no data points for this label
        plt.scatter(mu_proj[indices, 0], mu_proj[indices, 1], color=colors(i), label=f'{color_by_key.replace("_", " ").title()}: {label}', alpha=0.7, s=30)
    
    plt.title(f'Latent Space (PCA Projection) - {plot_title_suffix}')
    plt.xlabel('Principal Component 1'); plt.ylabel('Principal Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.80, 1])
    output_filename = f"latent_space_pca_plot_{filename_suffix}.png"
    plt.savefig(output_filename)
    print(f"Saved PCA plot to {output_filename}")
    plt.show()


def main_visualize():
    print("--- Visualization of Fault Detection Results ---")
    plot_sample_reconstructions_from_json(SAMPLE_RECON_JSON_PATH, SEQ_LEN)
    plot_latent_space_pca_from_csv(RESULTS_CSV_PATH, "Colored by True Dataset Label", "dataset", "true_labels")
    plot_latent_space_pca_from_csv(RESULTS_CSV_PATH, "Colored by Predicted Fault Type", "predicted_fault", "predicted_labels")
    print("--- Visualization complete ---")

if __name__ == "__main__":
    main_visualize()
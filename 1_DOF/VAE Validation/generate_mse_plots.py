import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set Seaborn theme for cleaner visuals
sns.set(style="whitegrid", font_scale=1.2)

# Define relative paths
test_stats_file = os.path.join("VAE Test Plot", "vae_segment_rmse_stats.csv")
unseen_stats_file = os.path.join("VAE Unseen Test Plot", "vae_segment_rmse_stats_unseen.csv")

# Check file existence
if not os.path.exists(test_stats_file):
    raise FileNotFoundError(f"Missing file: {test_stats_file}")
if not os.path.exists(unseen_stats_file):
    raise FileNotFoundError(f"Missing file: {unseen_stats_file}")

# Load CSVs
df_test = pd.read_csv(test_stats_file)
df_unseen = pd.read_csv(unseen_stats_file)

# Add labels
df_test["Source"] = "Clean"
df_unseen["Source"] = "Unseen"

# Combine into one DataFrame
df_all = pd.concat([df_test, df_unseen], ignore_index=True)

# Output folder for comparison plots
output_folder = "VAE Comparison Plots"
os.makedirs(output_folder, exist_ok=True)

# Define function for RMSE boxplot
def plot_boxplot(metric):
    plt.figure(figsize=(16, 9))
    sns.boxplot(data=df_all, x="Signal", y=metric, hue="Source")
    plt.title(f"{metric.replace('_', ' ')} Distribution: Clean vs Unseen", fontsize=16)
    plt.xlabel("Signal")
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Data Source")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{metric.lower()}_boxplot_comparison.png"))
    plt.show()
    plt.close()

# Plot all RMSE metrics as boxplots
for metric in ["Min_RMSE", "Max_RMSE", "Mean_RMSE", "Std_RMSE"]:
    plot_boxplot(metric)

# Save merged DataFrame for reference
df_all.to_csv(os.path.join(output_folder, "vae_combined_rmse_stats.csv"), index=False)

print("📊 Boxplot comparisons saved to 'VAE Comparison Plots'")

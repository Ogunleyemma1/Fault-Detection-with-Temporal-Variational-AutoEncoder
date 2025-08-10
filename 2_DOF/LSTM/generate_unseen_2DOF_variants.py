import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import sawtooth, square
import os

# Time vector
T_total = 30.0
dt = 0.01
t = np.arange(0, T_total + dt, dt)

# Signal parameters
A = 0.01   # Amplitude for both DOFs
f1 = 0.33  # Frequency for DOF 1
f2 = 0.5   # Different freq for DOF 2 (can choose as needed, or same as f1)

# === DOF 1 Signals ===
x1_ori = A * np.sin(2 * np.pi * f1 * t)
x1_env = (1 + 0.5 * np.cos(0.1 * np.pi * t)) * A * np.sin(2 * np.pi * f1 * t)
x1_tri = A * sawtooth(2 * np.pi * f1 * t, width=0.5)
x1_sqr = A * square(2 * np.pi * f1 * t)
v1_ori = np.gradient(x1_ori, t)
v1_env = np.gradient(x1_env, t)
v1_tri = np.gradient(x1_tri, t)
v1_sqr = np.gradient(x1_sqr, t)
a1_ori = np.gradient(v1_ori, t)
a1_env = np.gradient(v1_env, t)
a1_tri = np.gradient(v1_tri, t)
a1_sqr = np.gradient(v1_sqr, t)

# === DOF 2 Signals ===
x2_ori = A * np.sin(2 * np.pi * f2 * t + np.pi/4)   # Optional phase offset
x2_env = (1 + 0.5 * np.cos(0.13 * np.pi * t)) * A * np.sin(2 * np.pi * f2 * t + np.pi/4)
x2_tri = A * sawtooth(2 * np.pi * f2 * t, width=0.5)
x2_sqr = A * square(2 * np.pi * f2 * t)
v2_ori = np.gradient(x2_ori, t)
v2_env = np.gradient(x2_env, t)
v2_tri = np.gradient(x2_tri, t)
v2_sqr = np.gradient(x2_sqr, t)
a2_ori = np.gradient(v2_ori, t)
a2_env = np.gradient(v2_env, t)
a2_tri = np.gradient(v2_tri, t)
a2_sqr = np.gradient(v2_sqr, t)

# Save all to CSV
df = pd.DataFrame({
    "time": t,
    "x1_original": x1_ori,        "x1_envelope": x1_env,        "x1_triangle": x1_tri,        "x1_square": x1_sqr,
    "x2_original": x2_ori,        "x2_envelope": x2_env,        "x2_triangle": x2_tri,        "x2_square": x2_sqr,
    "v1_original": v1_ori,        "v1_envelope": v1_env,        "v1_triangle": v1_tri,        "v1_square": v1_sqr,
    "v2_original": v2_ori,        "v2_envelope": v2_env,        "v2_triangle": v2_tri,        "v2_square": v2_sqr,
    "a1_original": a1_ori,        "a1_envelope": a1_env,        "a1_triangle": a1_tri,        "a1_square": a1_sqr,
    "a2_original": a2_ori,        "a2_envelope": a2_env,        "a2_triangle": a2_tri,        "a2_square": a2_sqr,
})
csv_path = "2DOF_signal_unseen_variants.csv"
df.to_csv(csv_path, index=False)
print(f"✅ Saved to: {csv_path}")

# Output directory
output_dir = "VAE Validation/Plots_2DOF_Unseen"
os.makedirs(output_dir, exist_ok=True)

# Plotting function with fixed y-axis limits
def plot_variant_set(t, signals, ylabel, title_prefix, filename, ylim, labels):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for i, (data, label) in enumerate(zip(signals, labels)):
        axs[i].plot(t, data, color=colors[i])
        axs[i].set_title(f'{title_prefix} - {label}', fontsize=11)
        axs[i].set_ylabel(ylabel)
        axs[i].set_ylim(ylim)
        axs[i].grid(True)
    axs[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.show()

# Labels for plot
variant_labels = ['Original Sinusoid', 'Envelope-Modulated', 'Triangle Wave', 'Square Wave']

# Plot all for DOF 1
plot_variant_set(t, [x1_ori, x1_env, x1_tri, x1_sqr], 'Displacement (m)', 'Displacement Variants DOF1', 'Displacement_Unseen_DOF1.png', [-0.02, 0.02], variant_labels)
plot_variant_set(t, [v1_ori, v1_env, v1_tri, v1_sqr], 'Velocity (m/s)', 'Velocity Variants DOF1', 'Velocity_Unseen_DOF1.png', [-0.05, 0.05], variant_labels)
plot_variant_set(t, [a1_ori, a1_env, a1_tri, a1_sqr], 'Acceleration (m/s²)', 'Acceleration Variants DOF1', 'Acceleration_Unseen_DOF1.png', [-0.15, 0.15], variant_labels)

# Plot all for DOF 2
plot_variant_set(t, [x2_ori, x2_env, x2_tri, x2_sqr], 'Displacement (m)', 'Displacement Variants DOF2', 'Displacement_Unseen_DOF2.png', [-0.02, 0.02], variant_labels)
plot_variant_set(t, [v2_ori, v2_env, v2_tri, v2_sqr], 'Velocity (m/s)', 'Velocity Variants DOF2', 'Velocity_Unseen_DOF2.png', [-0.05, 0.05], variant_labels)
plot_variant_set(t, [a2_ori, a2_env, a2_tri, a2_sqr], 'Acceleration (m/s²)', 'Acceleration Variants DOF2', 'Acceleration_Unseen_DOF2.png', [-0.15, 0.15], variant_labels)

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
A = 0.01  # consistent amplitude with training data
f = 0.33  # base frequency (Hz)

# --- Original sinusoidal signal (used during training) ---
x_ori = A * np.sin(2 * np.pi * f * t)
v_ori = np.gradient(x_ori, t)
a_ori = np.gradient(v_ori, t)

# --- Envelope-modulated signal ---
x_env = (1 + 0.5 * np.cos(0.1 * np.pi * t)) * A * np.sin(2 * np.pi * f * t)
v_env = np.gradient(x_env, t)
a_env = np.gradient(v_env, t)

# --- Triangle wave signal ---
x_tri = A * sawtooth(2 * np.pi * f * t, width=0.5)
v_tri = np.gradient(x_tri, t)
a_tri = np.gradient(v_tri, t)

# --- Square wave signal ---
x_sqr = A * square(2 * np.pi * f * t)
v_sqr = np.gradient(x_sqr, t)
a_sqr = np.gradient(v_sqr, t)

# Save all to CSV
df = pd.DataFrame({
    "time": t,
    "x_original": x_ori,
    "x_envelope": x_env,
    "x_triangle": x_tri,
    "x_square": x_sqr,
    "v_original": v_ori,
    "v_envelope": v_env,
    "v_triangle": v_tri,
    "v_square": v_sqr,
    "a_original": a_ori,
    "a_envelope": a_env,
    "a_triangle": a_tri,
    "a_square": a_sqr
})
csv_path = "1DOF_signal_unseen_variants.csv"
df.to_csv(csv_path, index=False)
print(f"Saved to: {csv_path}")

# Output directory
output_dir = "VAE Validation/Plots"
os.makedirs(output_dir, exist_ok=True)

# Plotting function with fixed y-axis limits
def plot_variant_set(t, signals, ylabel, title_prefix, filename, ylim):
    labels = ['Original Sinusoid', 'Envelope-Modulated', 'Triangle Wave', 'Square Wave']
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

# Plot all with consistent y-axis ranges
plot_variant_set(t, [x_ori, x_env, x_tri, x_sqr], 'Displacement (m)', 'Displacement Variants', 'Displacement_Unseen.png', [-0.02, 0.02])
plot_variant_set(t, [v_ori, v_env, v_tri, v_sqr], 'Velocity (m/s)', 'Velocity Variants', 'Velocity_Unseen.png', [-0.05, 0.05])
plot_variant_set(t, [a_ori, a_env, a_tri, a_sqr], 'Acceleration (m/s²)', 'Acceleration Variants', 'Acceleration_Unseen.png', [-0.15, 0.15])

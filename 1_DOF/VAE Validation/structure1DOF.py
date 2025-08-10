import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# --- Simulation Parameters ---
m = 100.0          # mass (kg)
k = 1000.0         # stiffness (N/m)
c = 0.0            # damping coefficient
x0 = 0.01          # initial displacement (m)
v0 = 0.0           # initial velocity (m/s)
T_total = 30.0     # total simulation time (s)
dt = 0.01          # time step (s)
num_steps = int(T_total / dt) + 1
time = np.linspace(0, T_total, num_steps)

# --- Newmark-Beta Parameters ---
beta = 0.25
gamma = 0.5

# --- Initialize Arrays ---
x = np.zeros(num_steps)
v = np.zeros(num_steps)
a = np.zeros(num_steps)

# Initial Conditions
x[0] = x0
v[0] = v0
a[0] = (-k * x[0]) / m

# Effective Stiffness
K_eff = m / (beta * dt**2) + gamma * c / (beta * dt) + k

# --- Time Integration using Newmark-Beta ---
for i in range(1, num_steps):
    b = (
        m * ((1/(beta*dt**2)) * x[i-1] + (1/(beta*dt)) * v[i-1] + ((1/(2*beta)) - 1) * a[i-1])
        - c * (v[i-1] + (1 - gamma) * dt * a[i-1])
    )
    x[i] = b / K_eff
    a[i] = (1/(beta*dt**2)) * (x[i] - x[i-1]) - (1/(beta*dt)) * v[i-1] - ((1/(2*beta)) - 1) * a[i-1]
    v[i] = v[i-1] + dt * ((1 - gamma) * a[i-1] + gamma * a[i])

# --- Signal Variants ---
drift_rate = 0.001
x_drift = x + drift_rate * time
v_drift = v + drift_rate * time
a_drift = a + drift_rate * time

amp_scale = 1.5
x_amp = x * amp_scale
v_amp = v * amp_scale
a_amp = a * amp_scale

# Low-frequency variant: downsample and interpolate
xp = np.linspace(0, T_total, num_steps // 2)
x_half = x[:num_steps // 2]
v_half = v[:num_steps // 2]
a_half = a[:num_steps // 2]

x_lowfreq = np.interp(time, xp, x_half)
v_lowfreq = np.interp(time, xp, v_half)
a_lowfreq = np.interp(time, xp, a_half)

# --- Save to CSV ---
df = pd.DataFrame({
    "time": time,
    "x_original": x,
    "x_drifted": x_drift,
    "x_amplitude_scaled": x_amp,
    "x_lowfreq": x_lowfreq,
    "v_original": v,
    "v_drifted": v_drift,
    "v_amplitude_scaled": v_amp,
    "v_lowfreq": v_lowfreq,
    "a_original": a,
    "a_drifted": a_drift,
    "a_amplitude_scaled": a_amp,
    "a_lowfreq": a_lowfreq
})
df.to_csv("1DOF_signal_clean_variants.csv", index=False)
print("✅ Saved signal variants to '1DOF_signal_clean_variants.csv'")

# --- Save and Visualize Plots ---
save_dir = "VAE Validation/Plots"
os.makedirs(save_dir, exist_ok=True)

def plot_variants(y_data, labels, ylabel, title_prefix, colors, filename):
    fig, axs = plt.subplots(len(y_data), 1, figsize=(14, 10), sharex=True)

    y_min = min([np.min(data) for data in y_data])
    y_max = max([np.max(data) for data in y_data])

    for i, (data, label) in enumerate(zip(y_data, labels)):
        axs[i].plot(time, data, label=label, color=colors[i], linewidth=1.8)
        axs[i].set_title(f"{title_prefix} - {label}", fontsize=12)
        axs[i].set_ylabel(ylabel, fontsize=11)
        axs[i].set_ylim(y_min, y_max)
        axs[i].grid(True)
        axs[i].tick_params(axis='both', labelsize=10)

    axs[-1].set_xlabel("Time (s)", fontsize=11)
    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.show()

# --- Signal Data and Labels ---
labels = ["Original", "Drifted (Shift)", "Amplitude-Scaled", "Low-Frequency"]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

# --- Plot and Save ---
plot_variants([x, x_drift, x_amp, x_lowfreq], labels, "Displacement (m)", "Displacement Variants", colors, "Displacement_Variants.png")
plot_variants([v, v_drift, v_amp, v_lowfreq], labels, "Velocity (m/s)", "Velocity Variants", colors, "Velocity_Variants.png")
plot_variants([a, a_drift, a_amp, a_lowfreq], labels, "Acceleration (m/s²)", "Acceleration Variants", colors, "Acceleration_Variants.png")

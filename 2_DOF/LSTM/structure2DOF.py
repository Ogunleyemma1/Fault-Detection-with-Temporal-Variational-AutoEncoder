import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- System Parameters ---
m1, m2 = 100.0, 100.0
k1, k2 = 1000.0, 1000.0
c1, c2 = 0.0, 0.0
x0 = [0.01, 0.0]    # initial displacements
v0 = [0.0, 0.0]     # initial velocities

T_total = 30.0
dt = 0.01
num_steps = int(T_total / dt) + 1
time = np.linspace(0, T_total, num_steps)

beta = 0.25
gamma = 0.5

# --- Matrices ---
M = np.array([[m1, 0], [0, m2]])
K = np.array([[k1 + k2, -k2], [-k2, k2]])
C = np.array([[c1 + c2, -c2], [-c2, c2]])

# --- Initialize arrays ---
x = np.zeros((2, num_steps))
v = np.zeros((2, num_steps))
a = np.zeros((2, num_steps))
x[:, 0] = x0
v[:, 0] = v0
a[:, 0] = np.linalg.solve(M, -K @ x0 - C @ v0)

# --- Newmark-Beta Integration ---
K_eff = K + (gamma/(beta*dt))*C + (1/(beta*dt**2))*M
K_eff_inv = np.linalg.inv(K_eff)

for i in range(1, num_steps):
    # Predictors
    x_pred = x[:, i-1] + dt*v[:, i-1] + (dt**2)/2*(1-2*beta)*a[:, i-1]
    v_pred = v[:, i-1] + dt*(1-gamma)*a[:, i-1]
    F_ext = np.zeros(2)
    b = (F_ext
         + M @ (x_pred/(beta*dt**2))
         + C @ (v_pred - (gamma/(beta*dt))*x_pred)
    )
    x[:, i] = K_eff_inv @ b
    a[:, i] = (x[:, i] - x_pred)/(beta*dt**2)
    v[:, i] = v_pred + gamma*dt*a[:, i]

# --- Signal Variants ---
drift_rate = 0.001
x_drift = x + drift_rate * time
v_drift = v + drift_rate * time
a_drift = a + drift_rate * time

amp_scale = 1.5
x_amp = x * amp_scale
v_amp = v * amp_scale
a_amp = a * amp_scale

# Low-frequency variant: downsample & interpolate
xp = np.linspace(0, T_total, num_steps // 2)
x_half = x[:, :num_steps // 2]
v_half = v[:, :num_steps // 2]
a_half = a[:, :num_steps // 2]
x_lowfreq = np.vstack([np.interp(time, xp, x_half[dof]) for dof in range(2)])
v_lowfreq = np.vstack([np.interp(time, xp, v_half[dof]) for dof in range(2)])
a_lowfreq = np.vstack([np.interp(time, xp, a_half[dof]) for dof in range(2)])

# --- Save to CSV ---
df = pd.DataFrame({
    "time": time,
    "x1_original": x[0], "x1_drifted": x_drift[0], "x1_amplitude_scaled": x_amp[0], "x1_lowfreq": x_lowfreq[0],
    "x2_original": x[1], "x2_drifted": x_drift[1], "x2_amplitude_scaled": x_amp[1], "x2_lowfreq": x_lowfreq[1],
    "v1_original": v[0], "v1_drifted": v_drift[0], "v1_amplitude_scaled": v_amp[0], "v1_lowfreq": v_lowfreq[0],
    "v2_original": v[1], "v2_drifted": v_drift[1], "v2_amplitude_scaled": v_amp[1], "v2_lowfreq": v_lowfreq[1],
    "a1_original": a[0], "a1_drifted": a_drift[0], "a1_amplitude_scaled": a_amp[0], "a1_lowfreq": a_lowfreq[0],
    "a2_original": a[1], "a2_drifted": a_drift[1], "a2_amplitude_scaled": a_amp[1], "a2_lowfreq": a_lowfreq[1],
})
df.to_csv("2DOF_signal_clean_variants.csv", index=False)
print("✅ Saved 2DOF signal variants to '2DOF_signal_clean_variants.csv'")

# --- Save and Visualize Plots ---
save_dir = "VAE Validation/Plots_2DOF"
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

labels = ["Original", "Drifted (Shift)", "Amplitude-Scaled", "Low-Frequency"]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for dof in range(2):
    plot_variants([x[dof], x_drift[dof], x_amp[dof], x_lowfreq[dof]], labels, f"Displacement x{dof+1} (m)", f"Displacement Variants DOF{dof+1}", colors, f"Displacement_Variants_DOF{dof+1}.png")
    plot_variants([v[dof], v_drift[dof], v_amp[dof], v_lowfreq[dof]], labels, f"Velocity v{dof+1} (m/s)", f"Velocity Variants DOF{dof+1}", colors, f"Velocity_Variants_DOF{dof+1}.png")
    plot_variants([a[dof], a_drift[dof], a_amp[dof], a_lowfreq[dof]], labels, f"Acceleration a{dof+1} (m/s²)", f"Acceleration Variants DOF{dof+1}", colors, f"Acceleration_Variants_DOF{dof+1}.png")

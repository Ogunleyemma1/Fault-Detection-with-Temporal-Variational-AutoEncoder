import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------
# System Parameters
# ----------------------------
m = 100.0         # Mass (kg)
k = 1000.0        # Stiffness (N/m)
x0 = 0.01         # Initial displacement (m)
v0 = 0.0          # Initial velocity (m/s)
omega_n = np.sqrt(k / m)  # Natural frequency (rad/s)

# ----------------------------
# Time Configuration
# ----------------------------
T_total = 30.0
dt = 0.01
t = np.arange(0, T_total + dt, dt)

# ----------------------------
# Correct Analytical Solutions (Undamped, x(t) = x0 * cos(omega_n * t))
# ----------------------------
x = x0 * np.cos(omega_n * t)
v = -x0 * omega_n * np.sin(omega_n * t)
a = -x0 * omega_n**2 * np.cos(omega_n * t)

# ----------------------------
# Save to CSV for VAE Input
# ----------------------------
data_for_vae = np.vstack((x, v, a)).T
df_vae = pd.DataFrame(data_for_vae, columns=['x1', 'v1', 'a1'])
df_vae.to_csv("vae_input_analytical_data.csv", index=False)
print("Saved analytical solution to 'vae_input_analytical_data.csv'")

# ----------------------------
# Plotting Function
# ----------------------------
def plot_analytical_solution():
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(t, x, label='Displacement (x)', color='tab:blue')
    axs[1].plot(t, v, label='Velocity (v)', color='tab:orange')
    axs[2].plot(t, a, label='Acceleration (a)', color='tab:green')

    axs[0].set_ylabel("Displacement (m)")
    axs[1].set_ylabel("Velocity (m/s)")
    axs[2].set_ylabel("Acceleration (m/s²)")
    axs[2].set_xlabel("Time (s)")

    for ax in axs:
        ax.legend()
        ax.grid(True)

    plt.suptitle("1DOF Undamped Free Vibration (Analytical Solution)")
    plt.tight_layout()
    plt.show()

# ----------------------------
# Utility for Convergence Comparison
# ----------------------------
def get_analytical_displacement(T_total=30.0, dt=0.01):
    omega_n = np.sqrt(k / m)
    t = np.arange(0, T_total + dt, dt)
    x = x0 * np.cos(omega_n * t)
    return t, x

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    plot_analytical_solution()

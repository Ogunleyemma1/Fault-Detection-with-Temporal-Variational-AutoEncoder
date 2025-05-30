import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------
# System Parameters
# ----------------------------
m1, m2 = 100.0, 100.0
k1, k2 = 1000.0, 1000.0
T_total = 10.0
dt = 0.005
t = np.arange(0, T_total + dt, dt)

# ----------------------------
# Shared matrices for simulation
# ----------------------------
M = np.array([[m1, 0], [0, m2]])
K = np.array([[k1 + k2, -k2], [-k2, k2]])
eigvals, eigvecs = np.linalg.eig(np.linalg.inv(M) @ K)
omega = np.sqrt(eigvals)
phi = eigvecs

# ----------------------------
# Initial Conditions
# ----------------------------
x0 = np.array([0.01, 0.0])
v0 = np.array([0.0, 0.0])
A = np.linalg.inv(phi) @ x0
B = np.linalg.inv(phi) @ v0 / omega

# ----------------------------
# Solve Analytical Response
# ----------------------------
x_t, v_t, a_t = [], [], []

for ti in t:
    q = A * np.cos(omega * ti) + B * np.sin(omega * ti)
    dq = -A * omega * np.sin(omega * ti) + B * omega * np.cos(omega * ti)
    ddq = -A * omega**2 * np.cos(omega * ti) - B * omega**2 * np.sin(omega * ti)

    x = phi @ q
    v = phi @ dq
    a = phi @ ddq

    x_t.append(x)
    v_t.append(v)
    a_t.append(a)

x_t = np.array(x_t).T  # shape (2, N)
v_t = np.array(v_t).T
a_t = np.array(a_t).T

# ----------------------------
# Save to CSV for VAE Input
# ----------------------------
data = np.vstack((x_t, v_t, a_t)).T
df = pd.DataFrame(data, columns=['x1', 'x2', 'v1', 'v2', 'a1', 'a2'])
df.to_csv("vae_input_analytical_data_2dof.csv", index=False)
print("Saved analytical solution to 'vae_input_analytical_data_2dof.csv'")

# ----------------------------
# Plotting Function
# ----------------------------
def plot_analytical_solution_by_dof(t, x_t, v_t, a_t, dt):
    """
    Plots displacement, velocity, and acceleration for each DOF,
    including dt in plot titles and a main title per DOF.
    """
    indices = np.linspace(0, len(t) - 1, min(len(t), 1000), dtype=int)
    time = t[indices]
    x_plot = x_t[:, indices]
    v_plot = v_t[:, indices]
    a_plot = a_t[:, indices]

    for dof in range(x_t.shape[0]):
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f"Analytical Solution DOF[{dof+1}]")  # <- Main title as requested!

        axs[0].plot(time, x_plot[dof], label=f'x{dof+1} (Displacement)', color='tab:blue')
        axs[0].set_title(f"Displacement vs Time (dt = {dt})")

        axs[1].plot(time, v_plot[dof], label=f'v{dof+1} (Velocity)', color='tab:orange')
        axs[1].set_title(f"Velocity vs Time (dt = {dt})")

        axs[2].plot(time, a_plot[dof], label=f'a{dof+1} (Acceleration)', color='tab:green')
        axs[2].set_title(f"Acceleration vs Time (dt = {dt})")

        axs[2].set_xlabel("Time (s)")
        for ax in axs:
            ax.set_ylabel(ax.get_legend_handles_labels()[1][0].split()[0])
            ax.legend()
            ax.grid()

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leaves space for suptitle
        plt.show()

# ----------------------------
# Utility for Convergence Study
# ----------------------------
def get_analytical_displacement(T_total=10.0, dt=0.005):
    t = np.arange(0, T_total + dt, dt)
    # (Note: M, K, phi, omega, x0, v0 must be globally available)
    A = np.linalg.inv(phi) @ x0
    B = np.linalg.inv(phi) @ v0 / omega

    x1_vals, x2_vals = [], []
    for ti in t:
        q = A * np.cos(omega * ti) + B * np.sin(omega * ti)
        x = phi @ q
        x1_vals.append(x[0])
        x2_vals.append(x[1])

    return t, np.array(x1_vals), np.array(x2_vals)

# ----------------------------
# MAIN: Run when script is called directly
# ----------------------------
if __name__ == "__main__":
    plot_analytical_solution_by_dof(t, x_t, v_t, a_t, dt)

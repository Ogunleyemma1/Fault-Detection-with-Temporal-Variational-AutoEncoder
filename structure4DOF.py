# ---------------------------------------------
# This script simulates a 4-DOF damped spring-mass system
# using the Newmark-Beta integration method.
# Outputs:
# - A CSV file (vae_input_data.csv) containing displacement,
#   velocity, and acceleration for use in VAE training.
# - Plots of the time evolution of x, v, a per DOF in 3 subplots per figure.
# ---------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def run_simulation():
    # ----------------------------
    #  Define System Parameters
    # ----------------------------
    m = np.array([1.0, 1.0, 1.0, 1.0])
    k = np.array([100.0, 100.0, 100.0, 100.0])
    c = np.array([1.0, 1.0, 1.0, 1.0])

    # Time settings
    T_total = 20
    dt = 0.001
    safe_dt = max(dt, 1e-6)
    t_eval = np.arange(0, T_total, safe_dt)

    # Newmark-Beta integration parameters
    gamma = 0.5  # Controls velocity integration
    beta = 0.25   # Controls displacement integration

    # System matrices
    M = np.diag(m)
    K = np.diag(k) + np.diag(-k[1:], 1) + np.diag(-k[:-1], -1)
    C = np.diag(c) + np.diag(-c[1:], 1) + np.diag(-c[:-1], -1)

    M_inv = np.linalg.pinv(M) if np.linalg.cond(M) > 1e10 else np.linalg.inv(M)

    # ----------------------------
    #  Initial Conditions
    # ----------------------------
    x0 = np.zeros(4)
    v0 = np.zeros(4)
    a0 = np.nan_to_num(M_inv @ (-C @ v0 - K @ x0), nan=0.0)

    x = np.zeros((4, len(t_eval)))
    v = np.zeros((4, len(t_eval)))
    a = np.zeros((4, len(t_eval)))
    x[:, 0], v[:, 0], a[:, 0] = x0, v0, a0

    # Effective stiffness matrix for implicit integration
    K_eff = M / (beta * safe_dt**2) + gamma * C / (beta * safe_dt) + K
    K_inv = np.linalg.pinv(K_eff) if np.linalg.cond(K_eff) > 1e10 else np.linalg.inv(K_eff)

    # External sinusoidal force applied to the last mass
    def F_ext(t):
        return np.array([0.0, 0.0, 0.0, 1.0 * np.sin(2*np.pi * 0.5 * t)])

    # ----------------------------
    #  Time-Stepping Loop
    # ----------------------------
    for i in range(1, len(t_eval)):
        t = t_eval[i]
        F_t = F_ext(t)
        denom = max(beta * safe_dt**2, 1e-6)

        b = np.nan_to_num(
            F_t + M @ (x[:, i-1] / denom + v[:, i-1] / (beta * safe_dt) + (0.5 - beta) * a[:, i-1]),
            nan=0.0, posinf=1e5, neginf=-1e5
        )
        b -= C @ (v[:, i-1] + (1 - gamma) * safe_dt * a[:, i-1])

        x[:, i] = np.clip(K_inv @ b, -1e5, 1e5)
        a[:, i] = np.clip((x[:, i] - x[:, i-1]) / denom - v[:, i-1] / (beta * safe_dt) - (0.5 - beta) * a[:, i-1], -1e5, 1e5)
        v[:, i] = np.clip(v[:, i-1] + safe_dt * ((1 - gamma) * a[:, i-1] + gamma * a[:, i]), -1e5, 1e5)

    print("Simulation completed successfully.")

    # Stack data and save to CSV
    data_for_vae = np.vstack((x, v, a)).T
    df_vae = pd.DataFrame(data_for_vae, columns=['x1', 'x2', 'x3', 'x4', 'v1', 'v2', 'v3', 'v4', 'a1', 'a2', 'a3', 'a4'])
    df_vae.to_csv("vae_input_data.csv", index=False)

    # ----------------------------
    #  Plot DOF-wise Subplots (x, v, a) for each DOF separately
    # ----------------------------
    plot_indices = np.linspace(0, len(t_eval) - 1, 500, dtype=int)
    time = t_eval[plot_indices]
    x_plot = x[:, plot_indices]
    v_plot = v[:, plot_indices]
    a_plot = a[:, plot_indices]

    for dof in range(4):
        fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

        axs[0].plot(time, x_plot[dof], color='tab:blue', label=f"x{dof+1} (Displacement)")
        axs[0].set_ylabel("Displacement (m)")
        axs[0].set_title(f"DOF {dof+1} - Displacement vs Time")
        axs[0].grid(True)
        axs[0].legend()

        axs[1].plot(time, v_plot[dof], color='tab:orange', label=f"v{dof+1} (Velocity)")
        axs[1].set_ylabel("Velocity (m/s)")
        axs[1].set_title(f"DOF {dof+1} - Velocity vs Time")
        axs[1].grid(True)
        axs[1].legend()

        axs[2].plot(time, a_plot[dof], color='tab:green', label=f"a{dof+1} (Acceleration)")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Acceleration (m/s²)")
        axs[2].set_title(f"DOF {dof+1} - Acceleration vs Time")
        axs[2].grid(True)
        axs[2].legend()

        plt.tight_layout()
        plt.show()

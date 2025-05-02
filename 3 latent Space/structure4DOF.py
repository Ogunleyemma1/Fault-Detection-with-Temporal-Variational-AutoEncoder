import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# ----------------------------
#  Global System Configuration
# ----------------------------
system_config = {
    "mass": [1.0, 1.0, 1.0, 1.0],
    "stiffness": [100.0, 100.0, 100.0, 100.0],
    "damping": [0.1, 0.1, 0.1, 0.1],
    "T_total": 10,
    "dt": 0.01,
    "beta": 0.25,
    "gamma": 0.5,
    "force_function": lambda t: torch.zeros((len(t), 4))  # Default zero force (batch of zeros)
}


# ----------------------------
#  System Matrix Generator
# ----------------------------
def compute_matrices(m, k, c):
    m = np.array(m)
    k = np.array(k)
    c = np.array(c)

    M = np.diag(m)
    K = np.zeros((len(m), len(m)))
    for i in range(len(m)):
        if i == 0:
            K[i, i] += k[i]
        else:
            K[i, i] += k[i] + k[i - 1]
            K[i, i - 1] = -k[i - 1]
            K[i - 1, i] = -k[i - 1]

    C = np.diag(c) + np.diag(-c[1:], 1) + np.diag(-c[:-1], -1)
    return M, C, K


# ----------------------------
#  Torch-Formatted Matrix Access
# ----------------------------
def get_system_matrices(device):
    m = system_config["mass"]
    k = system_config["stiffness"]
    c = system_config["damping"]
    M, C, K = compute_matrices(m, k, c)
    return (
        torch.tensor(M, dtype=torch.float32, device=device),
        torch.tensor(C, dtype=torch.float32, device=device),
        torch.tensor(K, dtype=torch.float32, device=device)
    )


def get_force_function(device):
    return lambda t: system_config["force_function"](t).to(device)


# ----------------------------
#  Main Simulation Runner
# ----------------------------
def run_simulation():
    m = np.array(system_config["mass"])
    k = np.array(system_config["stiffness"])
    c = np.array(system_config["damping"])
    T_total = system_config["T_total"]
    dt = system_config["dt"]
    beta = system_config["beta"]
    gamma = system_config["gamma"]

    num_steps = int(T_total / dt) + 1  # Include final time
    t_eval = np.linspace(0, T_total, num_steps)
    M, C, K = compute_matrices(m, k, c)

    M_inv = np.linalg.pinv(M) if np.linalg.cond(M) > 1e10 else np.linalg.inv(M)

    x0 = np.array([0.05, 0.0, 0.0, 0.0])
    v0 = np.zeros(4)
    a0 = np.nan_to_num(M_inv @ (-C @ v0 - K @ x0), nan=0.0)

    x = np.zeros((4, len(t_eval)))
    v = np.zeros((4, len(t_eval)))
    a = np.zeros((4, len(t_eval)))
    x[:, 0], v[:, 0], a[:, 0] = x0, v0, a0

    K_eff = M / (beta * dt**2) + gamma * C / (beta * dt) + K
    K_inv = np.linalg.pinv(K_eff) if np.linalg.cond(K_eff) > 1e10 else np.linalg.inv(K_eff)

    F_ext = system_config["force_function"]

    for i in range(1, len(t_eval)):
        t = t_eval[i]
        F_t = F_ext(torch.tensor([t]))[0].numpy()
        denom = max(beta * dt**2, 1e-6)
        b = np.nan_to_num(
            F_t + M @ (x[:, i - 1] / denom + v[:, i - 1] / (beta * dt) + (0.5 - beta) * a[:, i - 1]),
            nan=0.0, posinf=1e5, neginf=-1e5
        )
        b -= C @ (v[:, i - 1] + (1 - gamma) * dt * a[:, i - 1])
        x[:, i] = np.clip(K_inv @ b, -1e5, 1e5)
        a[:, i] = np.clip((x[:, i] - x[:, i - 1]) / denom - v[:, i - 1] / (beta * dt) - (0.5 - beta) * a[:, i - 1], -1e5, 1e5)
        v[:, i] = np.clip(v[:, i - 1] + dt * ((1 - gamma) * a[:, i - 1] + gamma * a[:, i]), -1e5, 1e5)

    print("Simulation completed successfully.")

    data_for_vae = np.vstack((x, v, a)).T
    df_vae = pd.DataFrame(data_for_vae, columns=['x1', 'x2', 'x3', 'x4', 'v1', 'v2', 'v3', 'v4', 'a1', 'a2', 'a3', 'a4'])
    df_vae.to_csv("vae_input_data.csv", index=False)

    plot_indices = np.linspace(0, num_steps - 1, min(num_steps, 1000), dtype=int)
    time = t_eval[plot_indices]
    x_plot = x[:, plot_indices]
    v_plot = v[:, plot_indices]
    a_plot = a[:, plot_indices]

    for dof in range(4):
        fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        axs[0].plot(time, x_plot[dof], label=f"x{dof+1} (Displacement)", color='tab:blue')
        axs[1].plot(time, v_plot[dof], label=f"v{dof+1} (Velocity)", color='tab:orange')
        axs[2].plot(time, a_plot[dof], label=f"a{dof+1} (Acceleration)", color='tab:green')

        axs[0].set_ylabel("Displacement (m)")
        axs[1].set_ylabel("Velocity (m/s)")
        axs[2].set_ylabel("Acceleration (m/s²)")
        axs[2].set_xlabel("Time (s)")

        for ax in axs:
            ax.grid(True)
            ax.legend()

        plt.suptitle(f"DOF {dof+1} Response Over Time")
        plt.tight_layout()
        plt.show()

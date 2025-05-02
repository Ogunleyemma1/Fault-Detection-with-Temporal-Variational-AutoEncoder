import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# ----------------------------
#  System Configuration (2DOF Free Vibration)
# ----------------------------
system_config = {
    "mass": [100.0, 100.0],              # Masses (kg)
    "stiffness": [1000.0, 1000.0],       # Spring constants (N/m)
    "damping": [25.0, 25.0],            # Damping coefficients (Ns/m)
    "T_total": 30.0,                    # Total simulation time (s)
    "dt": 0.01,                         # Timestep size
    "beta": 0.25,                       # Newmark-Beta parameter
    "gamma": 0.5,                       # Newmark-Gamma parameter
    "force_function": lambda t: torch.zeros((len(t), 2))  # No external forcing
}


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


# ----------------------------
#  Matrix Assembly Functions
# ----------------------------
def compute_matrices(m, k, c):
    M = np.diag(m)
    K = np.array([
        [k[0] + k[1], -k[1]],
        [-k[1],       k[1]]
    ])
    C = np.array([
        [c[0] + c[1], -c[1]],
        [-c[1],       c[1]]
    ])
    return M, C, K

# ----------------------------
#  Main Simulation Function
# ----------------------------
def run_simulation():
    m = np.array(system_config["mass"])
    k = np.array(system_config["stiffness"])
    c = np.array(system_config["damping"])
    T_total = system_config["T_total"]
    dt = system_config["dt"]
    beta = system_config["beta"]
    gamma = system_config["gamma"]

    num_steps = int(T_total / dt) + 1
    t_eval = np.linspace(0, T_total, num_steps)

    M, C, K = compute_matrices(m, k, c)
    M_inv = np.linalg.inv(M)

    x0 = np.array([0.01, 0.000])
    v0 = np.zeros(2)
    a0 = np.nan_to_num(M_inv @ (-C @ v0 - K @ x0), nan=0.0)

    x = np.zeros((2, len(t_eval)))
    v = np.zeros((2, len(t_eval)))
    a = np.zeros((2, len(t_eval)))
    x[:, 0], v[:, 0], a[:, 0] = x0, v0, a0

    K_eff = M / (beta * dt**2) + gamma * C / (beta * dt) + K
    K_inv = np.linalg.inv(K_eff)

    F_ext = system_config["force_function"]

    for i in range(1, len(t_eval)):
        t = t_eval[i]
        F_t = F_ext(torch.tensor([t]))[0].numpy()
        denom = beta * dt**2

        b = F_t + M @ (x[:, i-1] / denom + v[:, i-1] / (beta * dt) + (0.5 - beta) * a[:, i-1])
        b -= C @ (v[:, i-1] + (1 - gamma) * dt * a[:, i-1])

        x[:, i] = np.clip(K_inv @ b, -1e5, 1e5)
        a[:, i] = np.clip(
            (x[:, i] - x[:, i-1]) / denom - v[:, i-1] / (beta * dt) - (0.5 - beta) * a[:, i-1],
            -1e5, 1e5
        )
        v[:, i] = np.clip(
            v[:, i-1] + dt * ((1 - gamma) * a[:, i-1] + gamma * a[:, i]),
            -1e5, 1e5
        )

    print("2DOF Free Vibration Simulation completed successfully.")

    data_for_vae = np.vstack((x, v, a)).T
    df_vae = pd.DataFrame(data_for_vae, columns=['x1', 'x2', 'v1', 'v2', 'a1', 'a2'])
    df_vae.to_csv("vae_input_data.csv", index=False)

    indices = np.linspace(0, num_steps - 1, min(num_steps, 1000), dtype=int)
    time = t_eval[indices]
    x_plot = x[:, indices]
    v_plot = v[:, indices]
    a_plot = a[:, indices]

    for dof in range(x.shape[0]):
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axs[0].plot(time, x_plot[dof], label=f'x{dof+1} (Displacement)', color='tab:blue')
        axs[0].set_title(f"DOF {dof+1} - Displacement vs Time")
        axs[1].plot(time, v_plot[dof], label=f'v{dof+1} (Velocity)', color='tab:orange')
        axs[1].set_title(f"DOF {dof+1} - Velocity vs Time")
        axs[2].plot(time, a_plot[dof], label=f'a{dof+1} (Acceleration)', color='tab:green')
        axs[2].set_title(f"DOF {dof+1} - Acceleration vs Time")
        axs[2].set_xlabel("Time (s)")
        for ax in axs:
            ax.set_ylabel(ax.get_legend_handles_labels()[1][0].split()[0])
            ax.legend()
            ax.grid()

        plt.tight_layout()
        plt.show(block=True)

# ----------------------------
#  Optional Force Getter
# ----------------------------
def get_force_function():
    return system_config["force_function"]

if __name__ == "__main__":
    run_simulation()

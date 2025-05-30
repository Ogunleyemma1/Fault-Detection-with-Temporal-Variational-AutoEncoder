import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# ----------------------------
#  System Configuration (1DOF Free Vibration)
# ----------------------------
system_config = {
    "mass": [100.0],          # mass m = 100 kg
    "stiffness": [1000.0],    # stiffness k = 1000 N/m
    "damping": [0.0],         # damping c = 0 kg/s for undamped
    "T_total": 30.0,          # total simulation time (s)
    "dt": 0.01,               # timestep size
    "beta": 0.25,             # Newmark-Beta parameters (constant average acceleration)
    "gamma": 0.5,
    "force_function": lambda t: torch.zeros((len(t), 1)),  # No external forcing (free vibration)
}

# ----------------------------
#  Matrix Assembly Functions
# ----------------------------
def compute_matrices(m, k, c):
    M = np.diag(m)
    K = np.array([[k[0]]])
    C = np.array([[c[0]]])
    return M, C, K

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
#  Main Simulation Function
# ----------------------------
def run_simulation():
    # Unpack configuration
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
    M_inv = np.linalg.inv(M)

    # Initial conditions
    x0 = np.array([0.01])   # Initial displacement 0.01 m
    v0 = np.zeros(1)        # Initial velocity 0 m/s
    # a0 = M^{-1} * (F_0 - C*v0 - K*x0) -- for undamped/unforced: a0 = -K*x0/M
    a0 = np.nan_to_num(M_inv @ (-C @ v0 - K @ x0), nan=0.0)

    # Storage arrays
    x = np.zeros((1, len(t_eval)))
    v = np.zeros((1, len(t_eval)))
    a = np.zeros((1, len(t_eval)))
    x[:, 0], v[:, 0], a[:, 0] = x0, v0, a0

    # Effective stiffness matrix
    K_eff = M / (beta * dt**2) + gamma * C / (beta * dt) + K
    K_inv = np.linalg.inv(K_eff)

    F_ext = system_config["force_function"]

    # Newmark-Beta coefficients
    a0_coef = 1.0 / (beta * dt**2)
    a1_coef = 1.0 / (beta * dt)
    a2_coef = (1.0 / (2 * beta)) - 1.0

    # Time-Stepping Loop (Newmark-Beta method)
    for i in range(1, len(t_eval)):
        t = t_eval[i]
        F_t = F_ext(torch.tensor([t]))[0].numpy()

        # RHS vector with correct coefficients
        b = (
            F_t
            + M @ (
                x[:, i-1] * a0_coef
                + v[:, i-1] * a1_coef
                + a[:, i-1] * a2_coef
            )
            - C @ (
                v[:, i-1] + (1.0 - gamma) * dt * a[:, i-1]
            )
        )

        # Solve for new displacement
        x[:, i] = K_inv @ b

        # Update acceleration and velocity
        a[:, i] = (
            a0_coef * (x[:, i] - x[:, i-1])
            - a1_coef * v[:, i-1]
            - a2_coef * a[:, i-1]
        )
        v[:, i] = v[:, i-1] + dt * ((1.0 - gamma) * a[:, i-1] + gamma * a[:, i])

    print("1DOF Free Vibration Simulation completed successfully.")

    # Save data
    data_for_vae = np.vstack((x, v, a)).T
    df_vae = pd.DataFrame(data_for_vae, columns=['x1', 'v1', 'a1'])
    df_vae.to_csv("vae_input_data.csv", index=False)

    # Plot (for quick comparison with analytical)
    indices = np.linspace(0, num_steps - 1, min(num_steps, 1000), dtype=int)
    time = t_eval[indices]
    x_plot = x[0, indices]
    v_plot = v[0, indices]
    a_plot = a[0, indices]

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(time, x_plot, color='tab:blue', label='Displacement x1')
    axs[1].plot(time, v_plot, color='tab:orange', label='Velocity v1')
    axs[2].plot(time, a_plot, color='tab:green', label='Acceleration a1')

    axs[0].set_ylabel("Displacement (m)")
    axs[1].set_ylabel("Velocity (m/s)")
    axs[2].set_ylabel("Acceleration (m/s²)")
    axs[2].set_xlabel("Time (s)")

    for ax in axs:
        ax.grid(True)
        ax.legend()

    plt.suptitle(f"1DOF Undamped Free Vibration (Numerical Solution)\nTime Step Size: Δt = {dt:.7f} s")
    plt.tight_layout()
    plt.show()

# Run directly
if __name__ == "__main__":
    run_simulation()

def get_force_function():
    return system_config["force_function"]

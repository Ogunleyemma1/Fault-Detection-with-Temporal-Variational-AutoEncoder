import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# ----------------------------
#  System Configuration (Illustrative Example 4.5 - Metric Units)
# ----------------------------

# Given values (in imperial units)
m_lb_s2_in = 10       # Mass
k_lb_per_in = 10000   # Stiffness

# Conversion factors
lb_to_N = 4.44822
in_to_m = 0.0254
lb_per_in_to_N_per_m = 175.1268
mass_conversion_factor = 175.1268  # 1 lb*s^2/in = 175.1268 kg

# Metric conversions
mass_kg = m_lb_s2_in * mass_conversion_factor
k_N_m = k_lb_per_in * lb_per_in_to_N_per_m

# Critical damping and actual damping (10% of critical)
c_cr = 2 * np.sqrt(mass_kg * k_N_m)
c_actual = 0.10 * c_cr

# Force ramp parameters
F_max_N = 5000 * lb_to_N  # Final force value in Newtons
t_ramp_end = 0.1          # Time at which ramp ends (sec)

# Define force function
def force_function(t):
    force_values = []
    for ti in t:
        if ti <= t_ramp_end:
            force = (F_max_N / t_ramp_end) * ti  # Linear ramp
        else:
            force = F_max_N
        force_values.append([force])
    return torch.tensor(force_values, dtype=torch.float32)

# Final system config
system_config = {
    "mass": [mass_kg],
    "stiffness": [k_N_m],
    "damping": [c_actual],
    "T_total": 0.3,           # Simulate for 2 seconds
    "dt": 0.02,              # Fine timestep for impulsive behavior
    "beta": 0.25,             # Newmark-Beta parameters
    "gamma": 0.5,
    "force_function": force_function,
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

    num_steps = int(T_total / dt) + 1
    t_eval = np.linspace(0, T_total, num_steps)

    M, C, K = compute_matrices(m, k, c)
    M_inv = np.linalg.inv(M)

    # Initial conditions
    x0 = np.array([0.0])   # Start at rest
    v0 = np.zeros(1)
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

    # Time-Stepping Loop (Newmark-Beta method)
    for i in range(1, len(t_eval)):
        t = t_eval[i]
        F_t = F_ext(torch.tensor([t]))[0].numpy()
        denom = beta * dt**2

        b = F_t + M @ (x[:, i-1] / denom + v[:, i-1] / (beta * dt) + (0.5 - beta) * a[:, i-1])
        b -= C @ (v[:, i-1] + (1 - gamma) * dt * a[:, i-1])

        x[:, i] = np.clip(K_inv @ b, -1e5, 1e5)
        a[:, i] = np.clip((x[:, i] - x[:, i-1]) / denom - v[:, i-1] / (beta * dt) - (0.5 - beta) * a[:, i-1], -1e5, 1e5)
        v[:, i] = np.clip(v[:, i-1] + dt * ((1 - gamma) * a[:, i-1] + gamma * a[:, i]), -1e5, 1e5)

    print("Illustrative Example 4.5 - Impulse Load Simulation completed successfully.")

    # Save data
    data_for_vae = np.vstack((x, v, a)).T
    df_vae = pd.DataFrame(data_for_vae, columns=['x1', 'v1', 'a1'])
    df_vae.to_csv("vae_input_data.csv", index=False)

    # Plot
    indices = np.linspace(0, num_steps - 1, min(num_steps, 2000), dtype=int)
    time = t_eval[indices]
    x_plot = x[0, indices]
    v_plot = v[0, indices]
    a_plot = a[0, indices]

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

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

    plt.suptitle("1DOF System Response - Illustrative Example 4.5 (Ramp Force)")
    plt.tight_layout()
    plt.show()

# Run directly
if __name__ == "__main__":
    run_simulation()

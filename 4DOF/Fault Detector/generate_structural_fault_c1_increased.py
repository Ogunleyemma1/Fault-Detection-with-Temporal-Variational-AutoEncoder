# generate_structural_fault_c1_increased.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# ... (original_system_config and faulty_system_config definitions remain the same) ...
original_system_config = {
    "mass": [100.0, 100.0, 100.0, 100.0],
    "stiffness": [1000.0, 1000.0, 1000.0, 1000.0],
    "damping": [25.0, 25.0, 25.0, 25.0],
    "T_total": 10.0,
    "dt": 0.01,
    "beta": 0.25,
    "gamma": 0.5,
    "force_function": lambda t: torch.zeros((len(t), 4))
}
faulty_system_config = original_system_config.copy()
faulty_system_config["damping"] = list(original_system_config["damping"])
faulty_system_config["damping"][0] *= 1.5
print(f"Simulating structural fault: c[0] increased by 50%. New damping: {faulty_system_config['damping']}")

# ... (compute_matrices function remains the same) ...
def compute_matrices(m, k, c):
    M = np.diag(m)
    n = len(m)
    K = np.zeros((n, n))
    C = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            K[i, i] += k[i]; K[i, i - 1] -= k[i - 1]; K[i - 1, i] -= k[i - 1]; K[i - 1, i - 1] += k[i - 1]
            C[i, i] += c[i]; C[i, i - 1] -= c[i - 1]; C[i - 1, i] -= c[i - 1]; C[i - 1, i - 1] += c[i - 1]
        else:
            K[i, i] += k[i]; C[i, i] += c[i]
    return M, C, K

# ----------------------------
#  Main Simulation Function
# ----------------------------
def run_simulation(current_system_config, output_filename="structural_fault_c1_increased.csv", normal_data_ref_file="vae_input_data.csv"):
    # ... (simulation setup: m_vals, k_vals, c_vals, etc. remains the same) ...
    m_vals = np.array(current_system_config["mass"])
    k_vals = np.array(current_system_config["stiffness"])
    c_vals = np.array(current_system_config["damping"])
    T_total = current_system_config["T_total"]
    dt = current_system_config["dt"]
    beta = current_system_config["beta"]
    gamma = current_system_config["gamma"]
    num_dof = len(m_vals)
    num_steps = int(T_total / dt) + 1
    t_eval = np.linspace(0, T_total, num_steps)
    M, C, K = compute_matrices(m_vals, k_vals, c_vals)
    M_inv = np.linalg.inv(M)
    x0 = np.zeros(num_dof); x0[num_dof-1] = 0.01; v0 = np.zeros(num_dof)
    F_at_t0 = current_system_config["force_function"](torch.tensor([0.0]))[0].numpy()
    a0 = np.nan_to_num(M_inv @ (F_at_t0 - C @ v0 - K @ x0), nan=0.0)
    x = np.zeros((num_dof, len(t_eval))); v = np.zeros((num_dof, len(t_eval))); a = np.zeros((num_dof, len(t_eval)))
    x[:, 0], v[:, 0], a[:, 0] = x0, v0, a0
    a0_nb = 1/(beta*dt**2); a1_nb = gamma/(beta*dt); a2_nb = 1/(beta*dt); a3_nb = (1/(2*beta))-1; a4_nb = (gamma/beta)-1; a5_nb = (dt/2)*((gamma/beta)-2)
    K_eff = a0_nb * M + a1_nb * C + K
    K_eff_inv = np.linalg.inv(K_eff)
    F_ext_func = current_system_config["force_function"]

    for i in range(1, len(t_eval)):
        # ... (Newmark-Beta loop remains the same) ...
        t_current = t_eval[i]; F_t = F_ext_func(torch.tensor([t_current]))[0].numpy()
        x_prev = x[:, i-1]; v_prev = v[:, i-1]; a_prev = a[:, i-1]
        P_eff_M_contrib = M @ (a0_nb*x_prev + a2_nb*v_prev + a3_nb*a_prev)
        P_eff_C_contrib = C @ (a1_nb*x_prev + a4_nb*v_prev + a5_nb*a_prev)
        P_eff = F_t + P_eff_M_contrib + P_eff_C_contrib
        x_curr = K_eff_inv @ P_eff; x[:, i] = np.clip(x_curr, -1e5, 1e5)
        a_curr = a0_nb*(x_curr-x_prev) - a2_nb*v_prev - a3_nb*a_prev
        v_curr = v_prev + dt*((1-gamma)*a_prev + gamma*a_curr)
        a[:, i] = np.clip(a_curr, -1e5, 1e5); v[:, i] = np.clip(v_curr, -1e5, 1e5)

    print(f"Faulty ({output_filename}) 4DOF Vibration Simulation completed successfully.")

    data_for_vae_faulty = np.vstack((x, v, a)).T
    labels = [f"x{j+1}" for j in range(num_dof)] + \
             [f"v{j+1}" for j in range(num_dof)] + \
             [f"a{j+1}" for j in range(num_dof)]
    df_vae_faulty = pd.DataFrame(data_for_vae_faulty, columns=labels)
    df_vae_faulty.to_csv(output_filename, index=False)
    print(f"Faulty data saved to {output_filename}")

    # --- Plotting Comparison ---
    try:
        df_normal_ref = pd.read_csv(normal_data_ref_file)
        plot_comparison = True
    except FileNotFoundError:
        print(f"Warning: Normal data reference file '{normal_data_ref_file}' not found. Skipping comparison plot.")
        plot_comparison = False

    if plot_comparison:
        # Plot for a representative DOF, e.g., x1, as c[0] affects DOF1
        dof_to_plot_idx = 0 # Index for x1 (0-based)
        channel_to_plot = f'x{dof_to_plot_idx+1}' # e.g., 'x1'

        plt.figure(figsize=(12, 6))
        plt.plot(t_eval, df_normal_ref[channel_to_plot], label=f'Normal - {channel_to_plot}', alpha=0.7)
        plt.plot(t_eval, df_vae_faulty[channel_to_plot], label=f'Faulty (c1 increased) - {channel_to_plot}', linestyle='--')
        plt.title(f'Comparison: Normal vs. Structural Fault (c[0] increased) - {channel_to_plot}')
        plt.xlabel('Time (s)')
        plt.ylabel('Displacement (m)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    # --- End of Plotting ---
    return t_eval, x

# ... (get_force_function and get_system_matrices remain the same) ...
def get_force_function(current_system_config, device=None): return lambda t: current_system_config["force_function"](t).to(device) if device else current_system_config["force_function"](t)
def get_system_matrices(current_system_config, device):
    m=current_system_config["mass"]; k_s=current_system_config["stiffness"]; c_s=current_system_config["damping"]
    M_mat, C_mat, K_mat = compute_matrices(m,k_s,c_s)
    return torch.tensor(M_mat,dtype=torch.float32,device=device), torch.tensor(C_mat,dtype=torch.float32,device=device), torch.tensor(K_mat,dtype=torch.float32,device=device)

if __name__ == "__main__":
    run_simulation(faulty_system_config,
                   output_filename="structural_fault_c1_increased.csv",
                   normal_data_ref_file="vae_input_data.csv")
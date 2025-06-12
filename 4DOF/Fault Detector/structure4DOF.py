import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# ----------------------------
#  System Configuration (4DOF Free Vibration)
# ----------------------------
system_config = {
    "mass": [100.0, 100.0, 100.0, 100.0],
    "stiffness": [1000.0, 1000.0, 1000.0, 1000.0],
    "damping": [25.0, 25.0, 25.0, 25.0],
    "T_total": 10.0,
    "dt": 0.01,
    "beta": 0.25,
    "gamma": 0.5,
    "force_function": lambda t: torch.zeros((len(t) if hasattr(t, '__len__') else 1, 4))
}

def compute_matrices(m_list, k_list, c_list):
    M_diag = np.array(m_list)
    k = np.array(k_list)
    c = np.array(c_list)
    M = np.diag(M_diag)
    n = len(M_diag)
    K = np.zeros((n, n))
    C = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            K[i, i] += k[i]
            K[i, i - 1] -= k[i - 1]
            K[i - 1, i] -= k[i - 1]
            K[i - 1, i - 1] += k[i - 1]
            C[i, i] += c[i]
            C[i, i - 1] -= c[i - 1]
            C[i - 1, i] -= c[i - 1]
            C[i - 1, i - 1] += c[i - 1]
        else:
            K[i, i] += k[i]
            C[i, i] += c[i]
    return M, C, K

def run_simulation(initial_displacement_x4, output_filename):
    m_vals = np.array(system_config["mass"])
    k_vals = np.array(system_config["stiffness"])
    c_vals = np.array(system_config["damping"])
    T_total = system_config["T_total"]
    dt = system_config["dt"]
    beta = system_config["beta"]
    gamma = system_config["gamma"]

    num_dof = len(m_vals)
    num_steps = int(T_total / dt) + 1
    t_eval = np.linspace(0, T_total, num_steps)

    M, C, K = compute_matrices(m_vals, k_vals, c_vals)
    M_inv = np.linalg.inv(M)

    x0 = np.zeros(num_dof)
    x0[num_dof-1] = initial_displacement_x4
    v0 = np.zeros(num_dof)
    F_at_t0 = system_config["force_function"](torch.tensor([0.0]))[0].numpy()
    a0 = np.nan_to_num(M_inv @ (F_at_t0 - C @ v0 - K @ x0), nan=0.0)

    x = np.zeros((num_dof, len(t_eval)))
    v = np.zeros((num_dof, len(t_eval)))
    a = np.zeros((num_dof, len(t_eval)))
    x[:, 0], v[:, 0], a[:, 0] = x0, v0, a0

    a0_nb = 1 / (beta * dt**2)
    a1_nb = gamma / (beta * dt)
    a2_nb = 1 / (beta * dt)
    a3_nb = (1 / (2 * beta)) - 1
    a4_nb = (gamma / beta) - 1
    a5_nb = (dt / 2) * ((gamma / beta) - 2)
    K_eff = a0_nb * M + a1_nb * C + K
    K_eff_inv = np.linalg.inv(K_eff)
    F_ext_func = system_config["force_function"]

    for i in range(1, len(t_eval)):
        t_current = t_eval[i]
        F_t = F_ext_func(torch.tensor([t_current]))[0].numpy()
        x_prev = x[:, i-1]
        v_prev = v[:, i-1]
        a_prev = a[:, i-1]
        P_eff_M_contrib = M @ (a0_nb * x_prev + a2_nb * v_prev + a3_nb * a_prev)
        P_eff_C_contrib = C @ (a1_nb * x_prev + a4_nb * v_prev + a5_nb * a_prev)
        P_eff = F_t + P_eff_M_contrib + P_eff_C_contrib
        x_curr = K_eff_inv @ P_eff
        x[:, i] = np.clip(x_curr, -1e5, 1e5)
        a_curr = a0_nb * (x_curr - x_prev) - a2_nb * v_prev - a3_nb * a_prev
        v_curr = v_prev + dt * ((1 - gamma) * a_prev + gamma * a_curr)
        a[:, i] = np.clip(a_curr, -1e5, 1e5)
        v[:, i] = np.clip(v_curr, -1e5, 1e5)

    # Save results
    data_for_output = np.vstack((x, v, a)).T
    labels = [f"x{j+1}" for j in range(num_dof)] + [f"v{j+1}" for j in range(num_dof)] + [f"a{j+1}" for j in range(num_dof)]
    df_output = pd.DataFrame(data_for_output, columns=labels)
    df_output.to_csv(output_filename, index=False)
    print(f"Data saved to {output_filename}")

    return t_eval, x, v, a

def generate_default_datasets(plot_training=True):
    print("\n--- Generating Training Data ---")
    t_train, x_train, v_train, a_train = run_simulation(initial_displacement_x4=0.01, output_filename="vae_input_data.csv")
    print("\n--- Generating Validation Data ---")
    run_simulation(initial_displacement_x4=0.02, output_filename="validation_data.csv")
    print("\n--- Generating Test Data Scenario ---")
    run_simulation(initial_displacement_x4=0.04, output_filename="test_data_scenario.csv")

    if plot_training:
        num_steps = len(t_train)
        indices = np.linspace(0, num_steps - 1, min(num_steps, 1000), dtype=int)
        time_plot = t_train[indices]
        x_plot = x_train[:, indices]
        v_plot = v_train[:, indices]
        a_plot = a_train[:, indices]

        for dof in range(x_train.shape[0]):
            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            axs[0].plot(time_plot, x_plot[dof], color='tab:blue', label=f'Displacement x{dof+1}')
            axs[0].set_ylabel('Displacement (m)')
            axs[0].legend()
            axs[0].grid(True)

            axs[1].plot(time_plot, v_plot[dof], color='tab:orange', label=f'Velocity v{dof+1}')
            axs[1].set_ylabel('Velocity (m/s)')
            axs[1].legend()
            axs[1].grid(True)

            axs[2].plot(time_plot, a_plot[dof], color='tab:green', label=f'Acceleration a{dof+1}')
            axs[2].set_ylabel('Acceleration (m/s²)')
            axs[2].set_xlabel('Time (s)')
            axs[2].legend()
            axs[2].grid(True)

            fig.suptitle(f"Training Data: DOF {dof+1} - Displacement, Velocity, Acceleration vs Time")
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.show()

        print("\nAll simulations completed. Only the training dataset's DOFs were plotted (x, v, a per DOF).")

# --- ONLY executes if run directly! ---
if __name__ == "__main__":
    generate_default_datasets(plot_training=True)

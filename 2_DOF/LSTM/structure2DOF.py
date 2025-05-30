import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
#  System Configuration (2DOF Free Vibration)
# ----------------------------
system_config = {
    "mass": [100.0, 100.0],
    "stiffness": [1000.0, 1000.0], # These are k1, k2 for the springs
    "damping": [0.0, 0.0],        # These are c1, c2 for the dampers (if any)
    "T_total": 10.0,
    "beta": 0.25,
    "gamma": 0.5,
    "dt": 0.01,  
}

def compute_matrices(m_vals, k_vals, c_vals):
    M = np.diag(m_vals)
    # For a 2DOF system: spring k_vals[0] (k1) between ground and m1, spring k_vals[1] (k2) between m1 and m2
    K = np.array([[k_vals[0] + k_vals[1], -k_vals[1]], 
                  [-k_vals[1], k_vals[1]]])
    # Similar structure for damping matrix C based on c_vals[0] (c1) and c_vals[1] (c2)
    if np.all(np.array(c_vals) == 0):
        C = np.zeros_like(K)
    else:
        C = np.array([[c_vals[0] + c_vals[1], -c_vals[1]], 
                      [-c_vals[1], c_vals[1]]])
    return M, C, K

def run_simulation(dt=None, save_plot=False):
    # Allow override of dt for convergence studies
    if dt is not None:
        current_dt = dt
    else:
        current_dt = system_config["dt"]

    m_config = np.array(system_config["mass"])
    k_config = np.array(system_config["stiffness"])
    c_config = np.array(system_config["damping"])
    T_total = system_config["T_total"]
    beta = system_config["beta"]
    gamma = system_config["gamma"]

    t = np.arange(0, T_total + current_dt, current_dt)
    num_steps = len(t)
    M, C, K = compute_matrices(m_config, k_config, c_config)
    
    x0 = np.array([0.01, 0.0])
    v0 = np.array([0.0, 0.0])

    x = np.zeros((2, num_steps))
    v = np.zeros((2, num_steps))
    a = np.zeros((2, num_steps))
    
    x[:, 0] = x0
    v[:, 0] = v0
    # Initial acceleration: M a0 + C v0 + K x0 = F0. Assuming F0 = 0 for free vibration.
    if np.all(M == 0): # Should not happen for valid mass matrix
        a[:, 0] = np.zeros_like(x0)
    else:
        a[:, 0] = np.linalg.solve(M, -K @ x0 - C @ v0)


    K_eff = K + (gamma / (beta * current_dt)) * C + (1 / (beta * current_dt**2)) * M
    try:
        K_eff_inv = np.linalg.inv(K_eff)
    except np.linalg.LinAlgError:
        print("Error: K_eff is singular. Check parameters.")
        return None, None # Or handle error appropriately

    for i in range(1, num_steps):
        # Predictor step
        x_pred = x[:, i-1] + current_dt * v[:, i-1] + (current_dt**2 / 2) * (1 - 2*beta) * a[:, i-1]
        v_pred = v[:, i-1] + current_dt * (1 - gamma) * a[:, i-1]
        
        F_ext = np.zeros(2)  # External force at time t_i, zero for free vibration
        
        term_M_contrib = M @ (x_pred / (beta * current_dt**2))
        term_C_contrib = C @ (v_pred - (gamma / (beta * current_dt)) * x_pred)
        
        b = F_ext + term_M_contrib + term_C_contrib
        
        x[:, i] = K_eff_inv @ b
        
        # Corrector step (update acceleration and velocity)
        a[:, i] = (x[:, i] - x_pred) / (beta * current_dt**2)
        v[:, i] = v_pred + gamma * current_dt * a[:, i]

    # Save to CSV for VAE and analysis
    data_for_vae = np.vstack((x, v, a)).T
    df_vae = pd.DataFrame(data_for_vae, columns=['x1', 'x2', 'v1', 'v2', 'a1', 'a2'])
    df_vae.to_csv("vae_input_data.csv", index=False)
    # print(f"2DOF Free Vibration Simulation (dt={current_dt}) completed successfully.") # Less verbose for convergence study

    # Optionally, plot for the last simulation
    if save_plot:
        indices = np.linspace(0, num_steps - 1, min(num_steps, 1000), dtype=int)
        time_plot = t[indices]
        x_plot = x[:, indices]
        v_plot = v[:, indices]
        a_plot = a[:, indices]
        for dof in range(x.shape[0]):
            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            fig.suptitle(f"Numerical Solution DOF[{dof+1}] (dt = {current_dt})")
            axs[0].plot(time_plot, x_plot[dof], label=f'x{dof+1} (Displacement)', color='tab:blue')
            axs[0].set_title(f"Displacement vs Time")
            axs[1].plot(time_plot, v_plot[dof], label=f'v{dof+1} (Velocity)', color='tab:orange')
            axs[1].set_title(f"Velocity vs Time")
            axs[2].plot(time_plot, a_plot[dof], label=f'a{dof+1} (Acceleration)', color='tab:green')
            axs[2].set_title(f"Acceleration vs Time")
            axs[2].set_xlabel("Time (s)")
            for j, ax_item in enumerate(axs):
                ax_item.set_ylabel(['x', 'v', 'a'][j] + str(dof+1))
                ax_item.legend()
                ax_item.grid()
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

    return t, x

if __name__ == "__main__":
    print("Running single simulation with default dt and plotting...")
    run_simulation(dt=system_config["dt"], save_plot=True)
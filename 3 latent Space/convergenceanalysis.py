import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from structure4DOF import run_simulation, system_config

def compute_l2_error(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def convergence_study():
    dt_values = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    errors = {f'dof{i+1}': [] for i in range(4)}

    # Use current dt in system_config as finest reference
    finest_dt = system_config["dt"]
    system_config["dt"] = finest_dt
    print(f"Running reference simulation with dt = {finest_dt}...")
    run_simulation()
    df_ref = pd.read_csv("vae_input_data.csv")
    ref_displacements = [df_ref[f'x{i+1}'].values for i in range(4)]

    for dt in dt_values[:-1]:  # exclude the last, which is the reference
        print(f"Running simulation with dt = {dt}...")
        system_config["dt"] = dt
        run_simulation()

        df = pd.read_csv("vae_input_data.csv")
        displacements = [df[f'x{i+1}'].values for i in range(4)]

        min_len = min(len(displacements[0]), len(ref_displacements[0]))
        for i in range(4):
            err = compute_l2_error(displacements[i][:min_len], ref_displacements[i][:min_len])
            errors[f'dof{i+1}'].append(err)

    # Plot convergence (log-log)
    dt_plot = dt_values[:-1]
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'd']
    for i in range(4):
        plt.loglog(dt_plot, errors[f'dof{i+1}'], marker=markers[i], label=f'DOF {i+1}')

    plt.xlabel("Time Step Size (dt)")
    plt.ylabel("L2 Error (vs Reference)")
    plt.title("Convergence Analysis - L2 Error vs Time Step")
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergence_plot_4DOF.png")
    plt.show()

if __name__ == "__main__":
    convergence_study()

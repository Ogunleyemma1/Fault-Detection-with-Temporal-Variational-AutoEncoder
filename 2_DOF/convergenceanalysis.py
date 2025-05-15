import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from structure2DOF import run_simulation, system_config


def compute_l2_error(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def convergence_study():
    dt_values = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    errors_dof1 = []
    errors_dof2 = []

    finest_dt = system_config["dt"]
    system_config["dt"] = finest_dt
    print(f"Running reference simulation with dt = {finest_dt}...")
    run_simulation()
    df_ref = pd.read_csv("vae_input_data.csv")
    ref_x1 = df_ref['x1'].values
    ref_x2 = df_ref['x2'].values

    for dt in dt_values[:-1]:
        print(f"Running simulation with dt = {dt}...")
        system_config["dt"] = dt
        run_simulation()

        df = pd.read_csv("vae_input_data.csv")
        x1 = df['x1'].values
        x2 = df['x2'].values

        min_len = min(len(x1), len(ref_x1))
        err1 = compute_l2_error(x1[:min_len], ref_x1[:min_len])
        err2 = compute_l2_error(x2[:min_len], ref_x2[:min_len])
        errors_dof1.append(err1)
        errors_dof2.append(err2)

    # Remove last dt (reference) for plotting
    dt_plot = dt_values[:-1]

    # Plot convergence (log-log)
    plt.figure(figsize=(10, 5))
    plt.loglog(dt_plot, errors_dof1, marker='o', label='DOF 1')
    plt.loglog(dt_plot, errors_dof2, marker='s', label='DOF 2')
    plt.xlabel("Time Step Size (dt)")
    plt.ylabel("L2 Error (vs Reference)")
    plt.title("Convergence Analysis - L2 Error vs Time Step")
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergence_plot.png")
    plt.show()


if __name__ == "__main__":
    convergence_study()

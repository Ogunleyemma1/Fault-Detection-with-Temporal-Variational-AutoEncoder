import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from structure1DOF import run_simulation, system_config
from undamped_1dof_analytical import get_analytical_displacement

def compute_normalized_l2_error(a, b):
    """
    Normalized discrete L2 error in percentage.
    ε = ||a - b|| / ||b|| * 100
    """
    return np.linalg.norm(a - b) / np.linalg.norm(b) * 100

def convergence_study():
    dt_values = [0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003125]
    T_total = system_config["T_total"]
    errors = []

    for dt in dt_values:
        print(f"Running simulation with dt = {dt}...")
        system_config["dt"] = dt
        run_simulation()

        # Load numerical solution
        df = pd.read_csv("vae_input_data.csv")
        x_num = df['x1'].values
        t_sim = np.linspace(0, T_total, len(x_num))

        # Compute analytical solution with same dt
        _, x_exact = get_analytical_displacement(T_total, dt=dt)

        # Ensure alignment in length
        min_len = min(len(x_num), len(x_exact))
        error = compute_normalized_l2_error(x_num[:min_len], x_exact[:min_len])
        errors.append(error)

    # Convert to arrays
    dt_plot = np.array(dt_values)
    errors = np.array(errors)

    # Compute convergence rate
    log_dt = np.log10(dt_plot)
    log_err = np.log10(errors)
    slope, _ = np.polyfit(log_dt, log_err, 1)
    beta = -slope

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.loglog(dt_plot, errors, marker='o', label=f"Normalized L2 Error (%), β ≈ {beta:.2f}")
    plt.xlabel("Time Step Size (dt)")
    plt.ylabel("Normalized L2 Error (%)")
    plt.title("Convergence Analysis - 1DOF (Analytical Reference)")
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergence_plot_1dof.png")
    plt.show()

    print(f"Estimated convergence rate β ≈ {beta:.2f}")

if __name__ == "__main__":
    convergence_study()

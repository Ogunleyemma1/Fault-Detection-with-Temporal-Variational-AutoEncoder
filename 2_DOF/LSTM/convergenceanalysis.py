import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from structure2DOF import run_simulation, system_config
from undamped_2dof_analytical import get_analytical_displacement

def compute_normalized_l2_error(a, b):
    """Normalized L2 error (%) between a and b"""
    return np.linalg.norm(a - b) / np.linalg.norm(b) * 100

def get_time_array(T_total, dt):
    """Robust time array generator: always same number of points for a given T_total and dt."""
    N = int(np.round(T_total / dt)) + 1
    return np.linspace(0, T_total, N)

def convergence_study():
    dt_values = [0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625]
    T_total = system_config["T_total"]
    errors_x1 = []
    errors_x2 = []

    for dt in dt_values:
        print(f"Running simulation with dt = {dt:.5f}")
        t = get_time_array(T_total, dt)

        # ---- Numerical solution (simulation and load) ----
        run_simulation(dt=dt, save_plot=False)
        df = pd.read_csv("vae_input_data.csv")
        x1_num = df['x1'].values
        x2_num = df['x2'].values

        # ---- Analytical solution (same t!) ----
        _, x1_exact, x2_exact = get_analytical_displacement(T_total, dt)

        # ---- Safety check: arrays must match in length ----
        assert len(x1_num) == len(x1_exact), f"Time arrays don't match for dt={dt}!"
        assert len(x2_num) == len(x2_exact), f"Time arrays don't match for dt={dt}!"

        # ---- Compute normalized L2 errors ----
        err1 = compute_normalized_l2_error(x1_num, x1_exact)
        err2 = compute_normalized_l2_error(x2_num, x2_exact)
        errors_x1.append(err1)
        errors_x2.append(err2)

    dt_plot = np.array(dt_values)
    errors_x1 = np.array(errors_x1)
    errors_x2 = np.array(errors_x2)

    # Estimate convergence rate (slope)
    log_dt = np.log10(dt_plot)
    slope1, _ = np.polyfit(log_dt, np.log10(errors_x1), 1)
    slope2, _ = np.polyfit(log_dt, np.log10(errors_x2), 1)

    # ---- Plotting ----
    plt.figure(figsize=(10, 5))
    plt.loglog(dt_plot, errors_x1, 'o-', label=f'x1 Error, β ≈ {-slope1:.2f}')
    plt.loglog(dt_plot, errors_x2, 's-', label=f'x2 Error, β ≈ {-slope2:.2f}')
    plt.xlabel("Time Step Size (dt)")
    plt.ylabel("Normalized L2 Error (%)")
    plt.title("Convergence Analysis - 2DOF vs Analytical Solution")
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergence_plot_2dof.png")
    plt.show()

    print(f"Convergence rate β for x1: {-slope1:.2f}")
    print(f"Convergence rate β for x2: {-slope2:.2f}")

if __name__ == "__main__":
    convergence_study()

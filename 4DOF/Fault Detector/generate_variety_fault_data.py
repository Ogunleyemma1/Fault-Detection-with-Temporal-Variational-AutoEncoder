import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ---- PHYSICAL PARAMETERS ----
BASE_MASS = 50.0
BASE_STIFFNESS = 200000.0
DAMPING_RATIO = 0.02
FORCE_RMS = 50.0
FORCE_SEED = 42
SIM_DURATION = 10.0
DT = 0.01

FAULT_DIR = "data_generation/faults"
PLOTS_DIR = os.path.join(FAULT_DIR, "plots")
NUM_DOFS = 4

def init_force(T_total, dt, num_dofs, rms, seed):
    np.random.seed(seed)
    steps = int(T_total / dt) + 1
    base = np.random.randn(steps, num_dofs) * rms
    window = int(0.5 / dt)
    for j in range(num_dofs):
        s = pd.Series(base[:, j])
        base[:, j] = s.rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    return torch.tensor(base, dtype=torch.float32)

def compute_matrices(m, k, zeta, num_dofs):
    M = np.diag(m)
    K = np.zeros((num_dofs, num_dofs))
    for i in range(num_dofs):
        if i == 0:
            K[i, i] = k[i] + k[i+1]
            K[i, i+1] = -k[i+1]
        elif i == num_dofs - 1:
            K[i, i] = k[i]
            K[i, i-1] = -k[i]
        else:
            K[i, i] = k[i] + k[i+1]
            K[i, i-1] = -k[i]
            K[i, i+1] = -k[i+1]
    eigvals = np.linalg.eigvals(np.linalg.inv(M) @ K)
    omegas = np.sqrt(np.sort(eigvals.real[eigvals.real > 0]))
    if len(omegas) < 2:
        alpha, beta = 0.1, 0.001
    else:
        o1, o2 = omegas[0], omegas[1]
        A = np.array([[1/(2*o1), o1/2], [1/(2*o2), o2/2]])
        z = np.array([zeta, zeta])
        alpha, beta = np.linalg.solve(A, z)
        alpha, beta = max(alpha, 0), max(beta, 0.0001)
    C = alpha * M + beta * K
    return M, C, K

def run_simulation(cfg, force_tensor, out_csv, duration, dt=0.01, zeta=None):
    m, k = np.array(cfg["mass"]), np.array(cfg["stiffness"])
    nd = cfg["num_dofs"]
    T_total, beta, gamma = duration, cfg["beta"], cfg["gamma"]
    t = np.linspace(0, T_total, int(T_total / dt) + 1)
    M, C, K = compute_matrices(m, k, zeta, nd)
    M_inv = np.linalg.inv(M)
    x0, v0 = np.zeros(nd), np.zeros(nd)
    F0 = force_tensor[0].numpy()
    a0 = np.nan_to_num(M_inv @ (F0 - C @ v0 - K @ x0), nan=0.0)
    x, v, a = np.zeros((nd, len(t))), np.zeros((nd, len(t))), np.zeros((nd, len(t)))
    x[:, 0], v[:, 0], a[:, 0] = x0, v0, a0
    a0_nb = 1 / (beta * dt ** 2)
    a1_nb = gamma / (beta * dt)
    a2_nb = 1 / (beta * dt)
    a3_nb = (1 / (2 * beta)) - 1
    a4_nb = (gamma / beta) - 1
    a5_nb = (dt / 2) * ((gamma / beta) - 2)
    K_eff = a0_nb * M + a1_nb * C + K
    K_eff_inv = np.linalg.inv(K_eff)
    for i in range(1, len(t)):
        F_t = force_tensor[i].numpy()
        x_prev, v_prev, a_prev = x[:, i-1], v[:, i-1], a[:, i-1]
        P_eff_M = M @ (a0_nb * x_prev + a2_nb * v_prev + a3_nb * a_prev)
        P_eff_C = C @ (a1_nb * x_prev + a4_nb * v_prev + a5_nb * a_prev)
        P_eff = F_t + P_eff_M + P_eff_C
        x_curr = K_eff_inv @ P_eff
        x[:, i] = np.clip(x_curr, -1e5, 1e5)
        a_curr = a0_nb * (x_curr - x_prev) - a2_nb * v_prev - a3_nb * a_prev
        v_curr = v_prev + dt * ((1 - gamma) * a_prev + gamma * a_curr)
        a[:, i] = np.clip(a_curr, -1e5, 1e5)
        v[:, i] = np.clip(v_curr, -1e5, 1e5)
    data = np.vstack((x, v, a)).T
    labels = [f"x{j+1}" for j in range(nd)] + [f"v{j+1}" for j in range(nd)] + [f"a{j+1}" for j in range(nd)]
    df = pd.DataFrame(data, columns=labels)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    return t, x, v, a

# ---- PLOTTING HELPERS ----
def plot_comparison_all_dofs(normal_file, fault_file, fault_label, plots_dir):
    df_n = pd.read_csv(normal_file)
    df_f = pd.read_csv(fault_file)
    t = np.linspace(0, SIM_DURATION, len(df_n))
    for dof in range(1, NUM_DOFS + 1):
        fig, axes = plt.subplots(3, 2, figsize=(18, 8))
        # Displacement
        axes[0, 0].plot(t, df_n[f'x{dof}'], label=f"Normal x{dof}")
        axes[0, 0].set_title(f"x{dof}: Normal")
        axes[0, 1].plot(t, df_f[f'x{dof}'], color='r', label=f"Fault x{dof} ({fault_label})")
        axes[0, 1].set_title(f"x{dof}: Fault")
        # Velocity
        axes[1, 0].plot(t, df_n[f'v{dof}'], label=f"Normal v{dof}", color='orange')
        axes[1, 0].set_title(f"v{dof}: Normal")
        axes[1, 1].plot(t, df_f[f'v{dof}'], color='brown', label=f"Fault v{dof} ({fault_label})")
        axes[1, 1].set_title(f"v{dof}: Fault")
        # Acceleration
        axes[2, 0].plot(t, df_n[f'a{dof}'], label=f"Normal a{dof}", color='green')
        axes[2, 0].set_title(f"a{dof}: Normal")
        axes[2, 1].plot(t, df_f[f'a{dof}'], color='darkgreen', label=f"Fault a{dof} ({fault_label})")
        axes[2, 1].set_title(f"a{dof}: Fault")
        # Labels and layout
        for i in range(3):
            axes[i, 0].legend()
            axes[i, 1].legend()
            axes[i, 0].set_ylabel(["Displacement (m)", "Velocity (m/s)", "Acceleration (m/s²)"][i])
        for j in range(2):
            axes[2, j].set_xlabel("Time (s)")
        plt.suptitle(f"DOF {dof}: Normal vs Fault ({fault_label}) — Displacement, Velocity, Acceleration")
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        os.makedirs(plots_dir, exist_ok=True)
        out_img = os.path.join(plots_dir, f"{os.path.splitext(os.path.basename(fault_file))[0]}_DOF{dof}.png")
        plt.savefig(out_img)
        plt.close()
        print(f"Plot saved to {out_img}")

# ---- STRUCTURAL FAULTS ----
def generate_structural_faults(normal_file):
    for k2_reduction in [0.5, 0.7, 0.8]:
        fault_cfg = {
            "mass": [BASE_MASS * 1.2, BASE_MASS, BASE_MASS, BASE_MASS * 0.8],
            "stiffness": [BASE_STIFFNESS * 1.5,
                          BASE_STIFFNESS * 1.2 * k2_reduction,
                          BASE_STIFFNESS,
                          BASE_STIFFNESS * 0.8],
            "beta": 0.25, "gamma": 0.5, "num_dofs": NUM_DOFS
        }
        force_tensor = init_force(SIM_DURATION, DT, NUM_DOFS, FORCE_RMS, FORCE_SEED)
        out_csv = os.path.join(FAULT_DIR, f"structural_fault_k2_reduced_{int(100*k2_reduction)}.csv")
        run_simulation(fault_cfg, force_tensor, out_csv, SIM_DURATION, dt=DT, zeta=DAMPING_RATIO)
        plot_comparison_all_dofs(normal_file, out_csv, f"k2 reduced {int(100*k2_reduction)}%", PLOTS_DIR)

    for damping_reduction in [0.5, 0.3, 0.1]:
        fault_cfg = {
            "mass": [BASE_MASS * 1.2, BASE_MASS, BASE_MASS, BASE_MASS * 0.8],
            "stiffness": [BASE_STIFFNESS * 1.5,
                          BASE_STIFFNESS * 1.2,
                          BASE_STIFFNESS,
                          BASE_STIFFNESS * 0.8],
            "beta": 0.25, "gamma": 0.5, "num_dofs": NUM_DOFS
        }
        force_tensor = init_force(SIM_DURATION, DT, NUM_DOFS, FORCE_RMS, FORCE_SEED)
        out_csv = os.path.join(FAULT_DIR, f"structural_fault_c1_reduced_{int(100*damping_reduction)}.csv")
        run_simulation(fault_cfg, force_tensor, out_csv, SIM_DURATION, dt=DT, zeta=DAMPING_RATIO * damping_reduction)
        plot_comparison_all_dofs(normal_file, out_csv, f"c1 reduced {int(100*damping_reduction)}%", PLOTS_DIR)

# ---- SENSOR FAULTS ----
def generate_sensor_faults(normal_file):
    df_normal = pd.read_csv(normal_file)
    # Sensor x1 zero
    df_x1_zero = df_normal.copy()
    df_x1_zero["x1"] = 0
    out_csv = os.path.join(FAULT_DIR, "sensor_fault_x1_zero.csv")
    df_x1_zero.to_csv(out_csv, index=False)
    plot_comparison_all_dofs(normal_file, out_csv, "x1 zero", PLOTS_DIR)
    print("Sensor fault (x1 zero) saved to", out_csv)
    # Sensor v3 noisy
    df_v3_noisy = df_normal.copy()
    np.random.seed(123)
    df_v3_noisy["v3"] += np.random.normal(0, 10, size=len(df_v3_noisy))
    out_csv = os.path.join(FAULT_DIR, "sensor_fault_v3_noisy.csv")
    df_v3_noisy.to_csv(out_csv, index=False)
    plot_comparison_all_dofs(normal_file, out_csv, "v3 noisy", PLOTS_DIR)
    print("Sensor fault (v3 noisy) saved to", out_csv)

# ---- MAIN ----
if __name__ == "__main__":
    os.makedirs(FAULT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    # Generate a healthy run as reference for plotting
    healthy_cfg = {
        "mass": [BASE_MASS * 1.2, BASE_MASS, BASE_MASS, BASE_MASS * 0.8],
        "stiffness": [BASE_STIFFNESS * 1.5,
                      BASE_STIFFNESS * 1.2,
                      BASE_STIFFNESS,
                      BASE_STIFFNESS * 0.8],
        "beta": 0.25, "gamma": 0.5, "num_dofs": NUM_DOFS
    }
    force_tensor = init_force(SIM_DURATION, DT, NUM_DOFS, FORCE_RMS, FORCE_SEED)
    healthy_file = os.path.join(FAULT_DIR, "healthy_base.csv")
    run_simulation(healthy_cfg, force_tensor, healthy_file, SIM_DURATION, dt=DT, zeta=DAMPING_RATIO)
    generate_structural_faults(healthy_file)
    generate_sensor_faults(healthy_file)
    print("\nAll fault data and all-DOF plots generated in data_generation/faults/plots/")

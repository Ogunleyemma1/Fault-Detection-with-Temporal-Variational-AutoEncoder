import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# === SET YOUR FOLDER HERE ===
FAULT_DIR = r"data_generation/faults"
STRUCTURAL_FAULT_DIR = os.path.join(FAULT_DIR, "structural_faults")
SENSOR_FAULT_DIR = os.path.join(FAULT_DIR, "sensor_faults")
PLOT_SUFFIX = "plots"
HEALTHY_FILE = os.path.join(FAULT_DIR, "healthy_base.csv")

# --- PHYSICAL PARAMETERS ---
BASE_MASS = 50.0
BASE_STIFFNESS = 200000.0
DAMPING_RATIO = 0.02
FORCE_RMS = 200.0
FORCE_SEED = 42
SIM_DURATION = 10.0
DT = 0.01
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

def run_simulation(cfg, force_tensor, duration, dt=0.01, zeta=None):
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
    return t, df

def plot_comparison_all_dofs(normal_df, fault_df, fault_label, plots_dir):
    t = np.linspace(0, SIM_DURATION, len(normal_df))
    stats = []
    for dof in range(1, NUM_DOFS + 1):
        fig, axes = plt.subplots(3, 2, figsize=(18, 8))
        for j, var in enumerate(['x', 'v', 'a']):
            normal = normal_df[f"{var}{dof}"]
            fault = fault_df[f"{var}{dof}"]
            axes[j, 0].plot(t, normal, label=f"Normal {var}{dof}", color=['b', 'orange', 'green'][j])
            axes[j, 0].set_title(f"{var}{dof}: Normal")
            axes[j, 1].plot(t, fault, label=f"Fault {var}{dof} ({fault_label})", color=['r', 'brown', 'darkgreen'][j])
            axes[j, 1].set_title(f"{var}{dof}: Fault")
            axes[j, 0].legend(); axes[j, 1].legend()
            axes[j, 0].set_ylabel(["Displacement (m)", "Velocity (m/s)", "Acceleration (m/s²)"][j])
            axes[2, j%2].set_xlabel("Time (s)")
            max_diff = np.max(np.abs(normal - fault))
            min_diff = np.min(normal - fault)
            stats.append([f"{var}{dof}", max_diff, min_diff])
        plt.suptitle(f"DOF {dof}: Normal vs Fault ({fault_label}) — Displacement, Velocity, Acceleration")
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        os.makedirs(plots_dir, exist_ok=True)
        out_img = os.path.join(plots_dir, f"{fault_label}_DOF{dof}.png")
        plt.savefig(out_img)
        plt.close()
    return stats

def log_table(stats, folder):
    df = pd.DataFrame(stats, columns=["Variable", "Max_Abs_Deviation", "Min_Deviation"])
    df.to_csv(os.path.join(folder, "deviation_stats.csv"), index=False)
    print(f"Deviation stats saved to {os.path.join(folder, 'deviation_stats.csv')}")
    print(df)

def make_structural_faults(normal_df, force_tensor, healthy_cfg):
    reductions = [0.7, 0.8, 0.9, 1.0]
    for perc in reductions:
        fault_cfg = healthy_cfg.copy()
        fault_cfg["stiffness"] = [s * perc for s in healthy_cfg["stiffness"]]
        label = f"k1_k2_k3_k4_reduced_{int(perc*100)}"
        f_dir = os.path.join(STRUCTURAL_FAULT_DIR, label)
        plot_dir = os.path.join(f_dir, PLOT_SUFFIX)
        _, fault_df = run_simulation(fault_cfg, force_tensor, SIM_DURATION, dt=DT, zeta=DAMPING_RATIO)
        fpath = os.path.join(f_dir, f"{label}.csv")
        os.makedirs(f_dir, exist_ok=True)
        fault_df.to_csv(fpath, index=False)
        stats = plot_comparison_all_dofs(normal_df, fault_df, label, plot_dir)
        log_table(stats, f_dir)

def add_sensor_noise(df, cols, std=0.2):
    noisy_df = df.copy()
    np.random.seed(123)
    for col in cols:
        noisy_df[col] += np.random.normal(0, std, size=len(df))
    return noisy_df

def add_spiky_fault(df, cols, mag=0.1, freq=0.1):
    spiky_df = df.copy()
    np.random.seed(123)
    n = len(df)
    for col in cols:
        spikes = np.zeros(n)
        spikes_idx = np.random.choice(n, int(n*freq), replace=False)
        spikes[spikes_idx] = np.random.normal(mag, mag/2, size=len(spikes_idx))
        spiky_df[col] += spikes
    return spiky_df

def add_drift_fault(df, cols, slope=0.005):
    drift_df = df.copy()
    n = len(df)
    drift = np.linspace(0, slope*n, n)
    for col in cols:
        drift_df[col] += drift
    return drift_df

def add_bias_fault(df, cols, bias=0.1):
    bias_df = df.copy()
    for col in cols:
        bias_df[col] += bias
    return bias_df

def make_sensor_faults(normal_df):
    faults = {
        "noisy": lambda df: add_sensor_noise(df, ["x4", "v4", "a4"], std=0.3),
        "spiky": lambda df: add_spiky_fault(df, ["x1", "v1", "a1"], mag=0.25, freq=0.08),
        "drift": lambda df: add_drift_fault(df, ["x2", "v2", "a2"], slope=0.003),
        "bias":  lambda df: add_bias_fault(df, ["x3", "v3", "a3"], bias=0.18)
    }
    for fault_name, fault_fn in faults.items():
        s_dir = os.path.join(SENSOR_FAULT_DIR, fault_name)
        plot_dir = os.path.join(s_dir, PLOT_SUFFIX)
        fault_df = fault_fn(normal_df)
        fpath = os.path.join(s_dir, f"{fault_name}.csv")
        os.makedirs(s_dir, exist_ok=True)
        fault_df.to_csv(fpath, index=False)
        stats = plot_comparison_all_dofs(normal_df, fault_df, fault_name, plot_dir)
        log_table(stats, s_dir)

if __name__ == "__main__":
    healthy_cfg = {
        "mass": [BASE_MASS * 1.2, BASE_MASS, BASE_MASS, BASE_MASS * 0.8],
        "stiffness": [BASE_STIFFNESS * 1.5,
                      BASE_STIFFNESS * 1.2,
                      BASE_STIFFNESS,
                      BASE_STIFFNESS * 0.8],
        "beta": 0.25, "gamma": 0.5, "num_dofs": NUM_DOFS
    }
    force_tensor = init_force(SIM_DURATION, DT, NUM_DOFS, FORCE_RMS, FORCE_SEED)

    if not os.path.isfile(HEALTHY_FILE):
        print(f"Healthy baseline file not found: {HEALTHY_FILE}")
        print("Generating healthy baseline data...")
        _, normal_df = run_simulation(healthy_cfg, force_tensor, SIM_DURATION, dt=DT, zeta=DAMPING_RATIO)
        os.makedirs(os.path.dirname(HEALTHY_FILE), exist_ok=True)
        normal_df.to_csv(HEALTHY_FILE, index=False)
        print(f"Healthy baseline data saved to {HEALTHY_FILE}")
    else:
        normal_df = pd.read_csv(HEALTHY_FILE)
        print(f"Healthy baseline loaded from {HEALTHY_FILE}")

    print("\n--- Generating Structural Faults ---")
    make_structural_faults(normal_df, force_tensor, healthy_cfg)

    print("\n--- Generating Sensor Faults ---")
    make_sensor_faults(normal_df)

    print("\nAll faults and plots have been generated. Check the faults folder.")

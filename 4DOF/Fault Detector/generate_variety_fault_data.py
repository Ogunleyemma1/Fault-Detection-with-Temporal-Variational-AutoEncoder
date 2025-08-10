import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import re

# ==== CONFIG ====
FAULT_DIR = r"data_generation/faults"
STRUCTURAL_FAULT_DIR = os.path.join(FAULT_DIR, "structural_faults")
SENSOR_FAULT_DIR = os.path.join(FAULT_DIR, "sensor_faults")
PLOT_SUFFIX = "plots"
HEALTHY_FILE = os.path.join(FAULT_DIR, "healthy_base.csv")

BASE_MASS = 50.0
BASE_STIFFNESS = 200000.0
DAMPING_RATIO = 0.02
FORCE_RMS = 200.0
FORCE_SEED = 42
SIM_DURATION = 10.0
DT = 0.01
NUM_DOFS = 4

def ensure_dir(path):
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path

def safe_name(text, maxlen=30):
    return re.sub(r'[^A-Za-z0-9_]', '', text)[:maxlen]

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
    a0_nb, a1_nb, a2_nb = 1/(beta*dt**2), gamma/(beta*dt), 1/(beta*dt)
    a3_nb, a4_nb, a5_nb = (1/(2*beta))-1, (gamma/beta)-1, (dt/2)*((gamma/beta)-2)
    K_eff = a0_nb * M + a1_nb * C + K
    K_eff_inv = np.linalg.inv(K_eff)
    for i in range(1, len(t)):
        F_t = force_tensor[i].numpy()
        x_prev, v_prev, a_prev = x[:, i-1], v[:, i-1], a[:, i-1]
        P_eff_M = M @ (a0_nb * x_prev + a2_nb * v_prev + a3_nb * a_prev)
        P_eff_C = C @ (a1_nb * x_prev + a4_nb * v_prev + a5_nb * a_prev)
        P_eff = F_t + P_eff_M + P_eff_C
        x_curr = K_eff_inv @ P_eff
        a_curr = a0_nb * (x_curr - x_prev) - a2_nb * v_prev - a3_nb * a_prev
        v_curr = v_prev + dt * ((1 - gamma) * a_prev + gamma * a_curr)
        x[:, i], v[:, i], a[:, i] = x_curr, v_curr, a_curr
    data = np.vstack((x, v, a)).T
    labels = [f"x{j+1}" for j in range(nd)] + [f"v{j+1}" for j in range(nd)] + [f"a{j+1}" for j in range(nd)]
    return t, pd.DataFrame(data, columns=labels)

def plot_comparison_all_dofs(normal_df, fault_df, fault_label, plots_dir):
    t = np.linspace(0, SIM_DURATION, len(normal_df))
    safe_label = safe_name(fault_label)
    ensure_dir(plots_dir)
    for dof in range(1, NUM_DOFS + 1):
        fig, axes = plt.subplots(3, 2, figsize=(18, 8), sharex=True)
        for j, var in enumerate(['x', 'v', 'a']):
            col_name = f"{var}{dof}"
            normal_data = normal_df[col_name]
            fault_data = fault_df[col_name]
            y_min, y_max = min(normal_data.min(), fault_data.min()), max(normal_data.max(), fault_data.max())
            padding = (y_max - y_min) * 0.05
            axes[j, 0].plot(t, normal_data, label=f"Normal {col_name}", color='navy')
            axes[j, 0].set_ylabel(["Displacement", "Velocity", "Acceleration"][j])
            axes[j, 0].legend(); axes[j, 0].grid(True, linestyle='--', alpha=0.6)
            axes[j, 1].plot(t, fault_data, label=f"Fault {col_name}", color='crimson')
            axes[j, 1].legend(); axes[j, 1].grid(True, linestyle='--', alpha=0.6)
            axes[j, 0].set_ylim(y_min - padding, y_max + padding)
            axes[j, 1].set_ylim(y_min - padding, y_max + padding)
        axes[-1, 0].set_xlabel("Time (s)"); axes[-1, 1].set_xlabel("Time (s)")
        plt.suptitle(f"DOF {dof} — Normal vs Fault: {safe_label}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(plots_dir, f"{safe_label}_DOF{dof}.png")
        plt.savefig(save_path); plt.close()

def make_structural_faults(normal_df, force_tensor, healthy_cfg):
    # This function remains the same as it correctly simulates a physical change.
    reductions = [0.9, 0.8, 0.7, 0.6]
    for perc in reductions:
        simple_label = f"Red{int(100-(perc*100))}pct"
        f_dir = os.path.join(STRUCTURAL_FAULT_DIR, simple_label); plot_dir = os.path.join(f_dir, PLOT_SUFFIX)
        ensure_dir(f_dir); ensure_dir(plot_dir)
        fault_cfg = healthy_cfg.copy()
        fault_cfg["stiffness"] = [s * perc for s in healthy_cfg["stiffness"]]
        _, fault_df = run_simulation(fault_cfg, force_tensor, SIM_DURATION, dt=DT, zeta=DAMPING_RATIO)
        fault_df.to_csv(os.path.join(f_dir, f"{simple_label}.csv"), index=False)
        plot_comparison_all_dofs(normal_df, fault_df, simple_label, plot_dir)

# --- [MODIFIED] Helper functions for targeted fault injection ---
def inject_noise(signal, magnitude):
    return signal + np.random.normal(0, magnitude, size=len(signal))

def inject_spikes(signal, magnitude, freq=0.01):
    spikes = np.zeros_like(signal)
    n = len(signal)
    spikes_idx = np.random.choice(n, int(n * freq), replace=False)
    spikes[spikes_idx] = np.random.normal(magnitude, magnitude / 4, size=len(spikes_idx))
    return signal + spikes

def inject_drift(signal, magnitude):
    return signal + np.linspace(0, magnitude, len(signal))

def inject_bias(signal, magnitude):
    return signal + magnitude
# ----------------------------------------------------------------

# --- [REWRITTEN] make_sensor_faults function for realistic simulation ---
def make_sensor_faults(normal_df):
    """
    Simulates realistic sensor faults where only one sensor measurement is corrupted
    at a time, while others (e.g., velocity, acceleration) remain healthy.
    """
    faults_config = {
        "noisy": {"func": inject_noise, "col": "x4", "rel_mag": 0.50},
        "spiky": {"func": inject_spikes, "col": "x1", "rel_mag": 5.0},
        "drift": {"func": inject_drift, "col": "x2", "rel_mag": 10.0},
        "bias":  {"func": inject_bias,  "col": "x3", "rel_mag": 2.0}
    }
    
    for fault_name, config in faults_config.items():
        s_dir = os.path.join(SENSOR_FAULT_DIR, fault_name)
        plot_dir = os.path.join(s_dir, PLOT_SUFFIX)
        ensure_dir(s_dir)
        ensure_dir(plot_dir)

        # Start with a fresh copy of the healthy data
        fault_df = normal_df.copy()
        
        # Identify the single column to corrupt
        target_col = config["col"]
        print(f"[INFO] Applying '{fault_name}' fault to sensor '{target_col}'")

        # Calculate the magnitude based on the healthy signal's standard deviation
        std_dev = normal_df[target_col].std()
        abs_magnitude = std_dev * config["rel_mag"]
        
        # Apply the fault function to ONLY the target column
        fault_df[target_col] = config["func"](normal_df[target_col], magnitude=abs_magnitude)
        
        # All other columns in fault_df remain as they were in normal_df (i.e., healthy)
        
        # Save the resulting dataframe
        fault_csv_path = os.path.join(s_dir, f"{fault_name}.csv")
        fault_df.to_csv(fault_csv_path, index=False)
        
        # Generate comparison plots
        plot_comparison_all_dofs(normal_df, fault_df, fault_name, plot_dir)

if __name__ == "__main__":
    healthy_cfg = {
        "mass": [BASE_MASS * 1.2, BASE_MASS, BASE_MASS, BASE_MASS * 0.8],
        "stiffness": [BASE_STIFFNESS * 1.5, BASE_STIFFNESS * 1.2, BASE_STIFFNESS, BASE_STIFFNESS * 0.8],
        "beta": 0.25, "gamma": 0.5, "num_dofs": NUM_DOFS
    }
    force_tensor = init_force(SIM_DURATION, DT, NUM_DOFS, FORCE_RMS, FORCE_SEED)
    
    if not os.path.isfile(HEALTHY_FILE):
        ensure_dir(os.path.dirname(HEALTHY_FILE))
        _, normal_df = run_simulation(healthy_cfg, force_tensor, SIM_DURATION, dt=DT, zeta=DAMPING_RATIO)
        normal_df.to_csv(HEALTHY_FILE, index=False)
        print(f"[INFO] Healthy baseline generated and saved to: {os.path.abspath(HEALTHY_FILE)}")
    else:
        normal_df = pd.read_csv(HEALTHY_FILE)
        print(f"[INFO] Healthy baseline loaded from: {os.path.abspath(HEALTHY_FILE)}")

    make_structural_faults(normal_df, force_tensor, healthy_cfg)
    print(f"[INFO] Structural fault datasets generated under: {os.path.abspath(STRUCTURAL_FAULT_DIR)}")

    make_sensor_faults(normal_df)
    print(f"[INFO] Sensor fault datasets generated under: {os.path.abspath(SENSOR_FAULT_DIR)}")

    print("\n[SUCCESS] All fault data and plots have been generated.")
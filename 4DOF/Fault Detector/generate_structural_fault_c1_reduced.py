import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# ---- Parameters for simulation ----
BASE_MASS = 50.0
BASE_STIFFNESS = 200000.0
DAMPING_RATIO_NORMAL = 0.02      # Use for healthy (not touched in this script)
DAMPING_RATIO_FAULT = 0.01       # 50% reduction for c1 in this script (faulty)
FORCE_RMS = 50.0
FORCE_SEED = 42
SIM_DURATION = 90.0
DT = 0.01

# ---- Structural Fault Output File ----
FAULT_FILE = "structural_fault_c1_reduced.csv"
NORMAL_FILE = "vae_input_data.csv"  # Used only for plotting/compare at end

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
    T_total, beta, gamma, nd = duration, cfg["beta"], cfg["gamma"], cfg["num_dofs"]
    t = np.linspace(0, T_total, int(T_total / dt) + 1)
    M, C, K = compute_matrices(m, k, zeta, nd)
    M_inv = np.linalg.inv(M)
    x0 = np.zeros(nd)
    v0 = np.zeros(nd)
    F0 = force_tensor[0].numpy()
    a0 = np.nan_to_num(M_inv @ (F0 - C @ v0 - K @ x0), nan=0.0)
    x = np.zeros((nd, len(t)))
    v = np.zeros((nd, len(t)))
    a = np.zeros((nd, len(t)))
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
    df.to_csv(out_csv, index=False)
    print(f"Fault data saved to {out_csv}")
    return t, x, v, a

def plot_dof1_normal_vs_fault_separated(normal_file, fault_file, dt=0.01):
    df_normal = pd.read_csv(normal_file)
    df_fault = pd.read_csv(fault_file)
    t = np.arange(len(df_normal)) * dt

    fig, axs = plt.subplots(3, 2, figsize=(14, 10), gridspec_kw={'width_ratios': [1, 1]}, sharex='col')

    # Displacement x1
    axs[0,0].plot(t, df_normal["x1"], label="Normal x1", color='tab:blue', linewidth=1)
    axs[0,0].set_ylabel("Displacement (m)")
    axs[0,0].legend()
    axs[0,0].grid(True)
    axs[0,0].set_title("x1: Normal")
    axs[0,1].plot(t, df_fault["x1"], label="Fault x1 (c1 reduced)", color='red', linewidth=1)
    axs[0,1].set_ylabel("Displacement (m)")
    axs[0,1].legend()
    axs[0,1].grid(True)
    axs[0,1].set_title("x1: Fault")

    # Velocity v1
    axs[1,0].plot(t, df_normal["v1"], label="Normal v1", color='tab:orange', linewidth=1)
    axs[1,0].set_ylabel("Velocity (m/s)")
    axs[1,0].legend()
    axs[1,0].grid(True)
    axs[1,0].set_title("v1: Normal")
    axs[1,1].plot(t, df_fault["v1"], label="Fault v1 (c1 reduced)", color='maroon', linewidth=1)
    axs[1,1].set_ylabel("Velocity (m/s)")
    axs[1,1].legend()
    axs[1,1].grid(True)
    axs[1,1].set_title("v1: Fault")

    # Acceleration a1
    axs[2,0].plot(t, df_normal["a1"], label="Normal a1", color='tab:green', linewidth=1)
    axs[2,0].set_ylabel("Acceleration (m/s²)")
    axs[2,0].legend()
    axs[2,0].grid(True)
    axs[2,0].set_title("a1: Normal")
    axs[2,0].set_xlabel("Time (s)")
    axs[2,1].plot(t, df_fault["a1"], label="Fault a1 (c1 reduced)", color='darkgreen', linewidth=1)
    axs[2,1].set_ylabel("Acceleration (m/s²)")
    axs[2,1].legend()
    axs[2,1].grid(True)
    axs[2,1].set_title("a1: Fault")
    axs[2,1].set_xlabel("Time (s)")

    plt.suptitle("DOF 1: Normal vs Fault (c1 Reduced by 50%) — Displacement, Velocity, Acceleration")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    fig2, axs2 = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axs2[0].plot(t, df_normal["x1"] - df_fault["x1"], color='purple', label='x1: Normal - Fault')
    axs2[0].set_ylabel("Δ Displacement (m)")
    axs2[0].set_title("Displacement Difference (x1)")
    axs2[0].legend()
    axs2[0].grid(True)
    axs2[1].plot(t, df_normal["v1"] - df_fault["v1"], color='brown', label='v1: Normal - Fault')
    axs2[1].set_ylabel("Δ Velocity (m/s)")
    axs2[1].set_title("Velocity Difference (v1)")
    axs2[1].legend()
    axs2[1].grid(True)
    axs2[2].plot(t, df_normal["a1"] - df_fault["a1"], color='darkgreen', label='a1: Normal - Fault')
    axs2[2].set_ylabel("Δ Acceleration (m/s²)")
    axs2[2].set_title("Acceleration Difference (a1)")
    axs2[2].set_xlabel("Time (s)")
    axs2[2].legend()
    axs2[2].grid(True)
    plt.suptitle("Difference (Normal - Fault) for DOF 1")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    # --- Fault config: everything same as normal except zeta (damping ratio) is reduced by 50% ---
    fault_config = {
        "mass": [BASE_MASS * 1.2, BASE_MASS, BASE_MASS, BASE_MASS * 0.8],
        "stiffness": [BASE_STIFFNESS * 1.5, BASE_STIFFNESS * 1.2, BASE_STIFFNESS, BASE_STIFFNESS * 0.8],
        "beta": 0.25,
        "gamma": 0.5,
        "num_dofs": 4
    }
    force_tensor = init_force(SIM_DURATION, DT, fault_config["num_dofs"], FORCE_RMS, FORCE_SEED)
    # Only run the fault case (healthy data should already exist as vae_input_data.csv)
    run_simulation(fault_config, force_tensor, FAULT_FILE, SIM_DURATION, dt=DT, zeta=DAMPING_RATIO_FAULT)
    # For comparison: plot x1, v1, a1 for DOF1 normal vs fault, plus difference
    plot_dof1_normal_vs_fault_separated(NORMAL_FILE, FAULT_FILE, dt=DT)

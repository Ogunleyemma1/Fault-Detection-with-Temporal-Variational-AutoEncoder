import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

BASE_MASS = 50.0
BASE_STIFFNESS = 200000.0
DAMPING_RATIO = 0.02
FORCE_RMS = 50.0
FORCE_SEED = 42
MASTER_DURATION = 10.0

system_config = {
    "mass": [BASE_MASS * 1.2, BASE_MASS, BASE_MASS, BASE_MASS * 0.8],
    "stiffness": [BASE_STIFFNESS * 1.5, BASE_STIFFNESS * 1.2, BASE_STIFFNESS, BASE_STIFFNESS * 0.8],
    "T_total": MASTER_DURATION,
    "dt": 0.01,
    "beta": 0.25,
    "gamma": 0.5,
    "num_dofs": 4,
    "damping_ratio": DAMPING_RATIO,
}

def init_force(T_total, dt, num_dofs, rms, seed):
    np.random.seed(seed)
    steps = int(T_total / dt) + 1
    base = np.random.randn(steps, num_dofs) * rms
    window = int(0.5 / dt)
    for j in range(num_dofs):
        s = pd.Series(base[:, j])
        base[:, j] = s.rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    return torch.tensor(base, dtype=torch.float32), steps, dt

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

def run_simulation(cfg, force_tensor, steps, dt, out_csv, duration, plot=False):
    m, k = np.array(cfg["mass"]), np.array(cfg["stiffness"])
    nd = cfg["num_dofs"]
    T_total, beta, gamma = duration, cfg["beta"], cfg["gamma"]
    t = np.linspace(0, T_total, int(T_total / cfg["dt"]) + 1)
    M, C, K = compute_matrices(m, k, cfg["damping_ratio"], nd)
    M_inv = np.linalg.inv(M)
    x0 = np.zeros(nd)
    v0 = np.zeros(nd)
    F0 = force_tensor[0].numpy()
    a0 = np.nan_to_num(M_inv @ (F0 - C @ v0 - K @ x0), nan=0.0)
    x = np.zeros((nd, len(t)))
    v = np.zeros((nd, len(t)))
    a = np.zeros((nd, len(t)))
    x[:, 0], v[:, 0], a[:, 0] = x0, v0, a0
    a0_nb = 1 / (beta * cfg["dt"] ** 2)
    a1_nb = gamma / (beta * cfg["dt"])
    a2_nb = 1 / (beta * cfg["dt"])
    a3_nb = (1 / (2 * beta)) - 1
    a4_nb = (gamma / beta) - 1
    a5_nb = (cfg["dt"] / 2) * ((gamma / beta) - 2)
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
        v_curr = v_prev + cfg["dt"] * ((1 - gamma) * a_prev + gamma * a_curr)
        a[:, i] = np.clip(a_curr, -1e5, 1e5)
        v[:, i] = np.clip(v_curr, -1e5, 1e5)
    data = np.vstack((x, v, a)).T
    labels = [f"x{j+1}" for j in range(nd)] + [f"v{j+1}" for j in range(nd)] + [f"a{j+1}" for j in range(nd)]
    df = pd.DataFrame(data, columns=labels)
    df.to_csv(out_csv, index=False)
    print(f"Data saved to {out_csv}")
    if plot:
        idx = np.linspace(0, len(t)-1, min(len(t), 3000), dtype=int)
        for j in range(nd):
            fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            axs[0].plot(t[idx], x[j, idx], color='tab:blue', label=f'x{j+1} (disp)')
            axs[0].set_ylabel('Displacement (m)')
            axs[0].legend()
            axs[0].grid(True)
            axs[1].plot(t[idx], v[j, idx], color='tab:orange', label=f'v{j+1} (vel)')
            axs[1].set_ylabel('Velocity (m/s)')
            axs[1].legend()
            axs[1].grid(True)
            axs[2].plot(t[idx], a[j, idx], color='tab:green', label=f'a{j+1} (acc)')
            axs[2].set_ylabel('Acceleration (m/s²)')
            axs[2].set_xlabel('Time (s)')
            axs[2].legend()
            axs[2].grid(True)
            fig.suptitle(f"DOF {j+1}: Displacement, Velocity, Acceleration vs Time")
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.show()
    return t, x, v, a

if __name__ == "__main__":
    cfg = system_config.copy()
    force_tensor, steps, dt = init_force(MASTER_DURATION, cfg["dt"], cfg["num_dofs"], FORCE_RMS, FORCE_SEED)
    run_simulation(cfg, force_tensor, steps, dt, out_csv="vae_input_data.csv", duration=MASTER_DURATION, plot=True)

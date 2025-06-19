import os
import numpy as np
from structure4DOF import system_config, init_force, run_simulation

# PARAMETERS
N_HEALTHY_RUNS = 10
OUT_DIR = "data_generation/healthy_runs"
FORCE_RMS = 50.0
BASE_SEED = 2025

os.makedirs(OUT_DIR, exist_ok=True)

for i in range(N_HEALTHY_RUNS):
    seed = BASE_SEED + i
    cfg = system_config.copy()
    # --- Add small, random deviations to mass/stiffness/damping for each run ---
    mass_jitter = np.random.uniform(0.98, 1.02, size=len(cfg['mass']))
    stiff_jitter = np.random.uniform(0.98, 1.02, size=len(cfg['stiffness']))
    cfg['mass'] = [m * mj for m, mj in zip(cfg['mass'], mass_jitter)]
    cfg['stiffness'] = [k * kj for k, kj in zip(cfg['stiffness'], stiff_jitter)]
    # Optional: Slight damping jitter
    cfg['damping_ratio'] = float(np.random.uniform(0.015, 0.025))
    force_tensor, steps, dt = init_force(
        cfg["T_total"], cfg["dt"], cfg["num_dofs"], FORCE_RMS, seed
    )
    out_csv = os.path.join(OUT_DIR, f"healthy_seed{seed}.csv")
    run_simulation(cfg, force_tensor, steps, dt, out_csv, duration=cfg["T_total"], plot=False)
    print(f"Generated: {out_csv}")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# System parameters
m = np.array([1.0, 1.0, 1.0, 1.0])  # Masses
k = np.array([100.0, 100.0, 100.0, 100.0])  # Spring constants
c = np.array([1.0, 1.0, 1.0, 1.0])  # Damping coefficients

# Time parameters
T_total = 3600  # Simulate for 1 hour
dt = 0.1  # Time step
safe_dt = max(dt, 1e-6)  # Ensure stability
t_eval = np.arange(0, T_total, safe_dt)

# Newmark-Beta parameters
gamma = 0.5
beta = 0.3

# System Matrices
M = np.diag(m)
K = np.diag(k) + np.diag(-k[1:], k=1) + np.diag(-k[:-1], k=-1)
C = np.diag(c) + np.diag(-c[1:], k=1) + np.diag(-c[:-1], k=-1)

# Check if M is singular
if np.linalg.cond(M) > 1e10:
    print("Warning: M is nearly singular, using pseudo-inverse")
    M_inv = np.linalg.pinv(M)
else:
    M_inv = np.linalg.inv(M)

# Initial Conditions
x0 = np.zeros(4)
v0 = np.zeros(4)
a0 = np.nan_to_num(M_inv @ (-C @ v0 - K @ x0), nan=0.0)

# Storage Arrays
x = np.zeros((4, len(t_eval)))
v = np.zeros((4, len(t_eval)))
a = np.zeros((4, len(t_eval)))

# Assign Initial Conditions
x[:, 0] = x0
v[:, 0] = v0
a[:, 0] = a0

# Compute Effective Stiffness Matrix
K_eff = M / (beta * safe_dt**2) + gamma * C / (beta * safe_dt) + K

# Check if K_eff is singular
if np.linalg.cond(K_eff) > 1e10:
    print("Warning: K_eff is nearly singular, using pseudo-inverse")
    K_inv = np.linalg.pinv(K_eff)
else:
    K_inv = np.linalg.inv(K_eff)

# External Force Function
def F_ext(t):
    return np.array([0.0, 0.0, 0.0, 5.0 * np.sin(2 * np.pi * t)])

# Time-Stepping Loop
for i in range(1, len(t_eval)):
    t = t_eval[i]
    F_t = F_ext(t)

    # Prevent division overflow by clamping denominators
    denom = max(beta * safe_dt**2, 1e-6)

    # Compute Right-Hand Side
    b = np.nan_to_num(
        F_t + M @ (x[:, i-1] / denom + v[:, i-1] / (beta * safe_dt) + (0.5 - beta) * a[:, i-1]),
        nan=0.0, posinf=1e5, neginf=-1e5
    )
    b -= C @ (v[:, i-1] + (1 - gamma) * safe_dt * a[:, i-1])

    # Solve for New Displacement
    x[:, i] = np.clip(K_inv @ b, -1e5, 1e5)

    # Compute Acceleration
    a[:, i] = np.clip(
        (x[:, i] - x[:, i-1]) / denom - v[:, i-1] / (beta * safe_dt) - (0.5 - beta) * a[:, i-1],
        -1e5, 1e5
    )

    # Compute Velocity
    v[:, i] = np.clip(
        v[:, i-1] + safe_dt * ((1 - gamma) * a[:, i-1] + gamma * a[:, i]),
        -1e5, 1e5
    )

print("Simulation completed successfully.")

data_for_vae = np.vstack((x, v, a)).T

# Create a Pandas DataFrame
df_vae = pd.DataFrame(data_for_vae, columns=['x1', 'x2', 'x3', 'x4', 
                                             'v1', 'v2', 'v3', 'v4', 
                                             'a1', 'a2', 'a3', 'a4'])

# Save the DataFrame to a CSV file
df_vae.to_csv("vae_input_data.csv", index=False)  # index=False prevents writing row numbers

# Select indices every 300 seconds (5 minutes)
five_min_indices = np.arange(0, len(t_eval), int(300 // safe_dt)).astype(int)

# Ensure indices are within bounds
five_min_indices = five_min_indices[five_min_indices < len(t_eval)]

# **Plot Results**
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t_eval[five_min_indices], x[:, five_min_indices].T, marker='o', linestyle='-')
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.title("Displacements (Every 5 Minutes)")

plt.subplot(3, 1, 2)
plt.plot(t_eval[five_min_indices], v[:, five_min_indices].T, marker='o', linestyle='-')
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocities (Every 5 Minutes)")

plt.subplot(3, 1, 3)
plt.plot(t_eval[five_min_indices], a[:, five_min_indices].T, marker='o', linestyle='-')
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Accelerations (Every 5 Minutes)")

plt.tight_layout()
plt.show()
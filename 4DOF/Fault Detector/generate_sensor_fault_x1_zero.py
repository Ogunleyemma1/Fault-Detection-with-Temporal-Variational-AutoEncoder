# generate_sensor_fault_x1_zero.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Added for plotting

def apply_x1_zero_fault(input_csv="vae_input_data.csv", output_csv="sensor_fault_x1_zero.csv"):
    try:
        df_normal = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Input file '{input_csv}' not found. Please generate normal data first.")
        return

    df_faulty = df_normal.copy()
    target_column = 'x1'

    if target_column not in df_faulty.columns:
        print(f"Error: Column '{target_column}' not found in '{input_csv}'. Check column names.")
        return

    original_signal_normal = df_normal[target_column].values # For plotting original
    df_faulty[target_column] = 0.0
    faulty_signal = df_faulty[target_column].values

    df_faulty.to_csv(output_csv, index=False)
    print(f"Sensor fault data ({target_column} reads zero) saved to '{output_csv}'")

    # --- Plotting Comparison ---
    # Assuming time steps are just row indices for simplicity in plotting
    # If you have a 'time' column in vae_input_data.csv, use that.
    # Otherwise, generate a time array based on dt from your simulation config.
    # For now, using index as time step for simplicity.
    time_steps = np.arange(len(original_signal_normal))
    # If you have dt and T_total, you can create a more accurate time axis:
    # T_total_sim = 10.0 # From your config
    # dt_sim = 0.01    # From your config
    # num_samples_sim = int(T_total_sim / dt_sim) + 1
    # time_eval_sim = np.linspace(0, T_total_sim, num_samples_sim)
    # if len(time_eval_sim) == len(original_signal_normal):
    #     time_steps = time_eval_sim
    # else:
    #     print("Warning: Time axis length mismatch for plotting. Using simple indices.")


    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, original_signal_normal, label=f'Normal - {target_column}', alpha=0.7)
    plt.plot(time_steps, faulty_signal, label=f'Faulty (Sensor Zero) - {target_column}', linestyle='--')
    plt.title(f'Comparison: Normal vs. Sensor Fault ({target_column} Zero)')
    plt.xlabel('Time Step / Time (s)')
    plt.ylabel('Signal Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # --- End of Plotting ---

if __name__ == "__main__":
    normal_data_file = "vae_input_data.csv"
    faulty_data_output_file = "sensor_fault_x1_zero.csv"
    apply_x1_zero_fault(input_csv=normal_data_file, output_csv=faulty_data_output_file)
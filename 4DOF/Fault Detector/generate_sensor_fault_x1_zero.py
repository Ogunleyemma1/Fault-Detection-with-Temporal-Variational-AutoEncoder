import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def apply_x1_zero_fault(input_csv="test_data_scenario.csv", output_csv="sensor_fault_x1_zero_TEST.csv"):
    try:
        df_normal = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Input file '{input_csv}' not found. Please generate test data first.")
        return

    df_faulty = df_normal.copy()
    target_column = 'x1'

    if target_column not in df_faulty.columns:
        print(f"Error: Column '{target_column}' not found in '{input_csv}'. Check column names.")
        return

    original_signal_normal = df_normal[target_column].values
    df_faulty[target_column] = 0.0
    faulty_signal = df_faulty[target_column].values

    df_faulty.to_csv(output_csv, index=False)
    print(f"Sensor fault data ({target_column} forced to zero) saved to '{output_csv}'")

    # --- Plotting Comparison ---
    time_steps = np.arange(len(original_signal_normal))

    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, original_signal_normal, label=f'Normal - {target_column}', alpha=0.7)
    plt.plot(time_steps, faulty_signal, label=f'Faulty (Sensor Zero) - {target_column}', linestyle='--')
    plt.title(f'Comparison: Normal vs. Sensor Fault ({target_column} Zero) on Test Data')
    plt.xlabel('Time Step / Time (s)')
    plt.ylabel('Signal Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    normal_data_file = "vae_input_data.csv"
    faulty_data_output_file = "sensor_fault_x1_zero.csv"
    apply_x1_zero_fault(input_csv=normal_data_file, output_csv=faulty_data_output_file)

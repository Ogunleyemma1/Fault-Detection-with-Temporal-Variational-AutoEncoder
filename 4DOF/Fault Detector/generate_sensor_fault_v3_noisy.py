import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def apply_v3_noisy_fault(input_csv="test_data_scenario.csv", output_csv="sensor_fault_v3_noisy_TEST.csv", noise_factor=1.5):
    try:
        df_normal = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Input file '{input_csv}' not found. Please generate normal test data first.")
        return

    df_faulty = df_normal.copy()
    target_column = 'v3'

    if target_column not in df_faulty.columns:
        print(f"Error: Column '{target_column}' not found in '{input_csv}'. Check column names.")
        return

    original_signal_normal = df_normal[target_column].values
    original_signal_to_corrupt = df_faulty[target_column].values

    signal_std = np.std(original_signal_to_corrupt)
    noise_std = signal_std * noise_factor
    noise = np.random.normal(loc=0.0, scale=noise_std, size=len(original_signal_to_corrupt))

    df_faulty[target_column] = original_signal_to_corrupt + noise
    faulty_signal = df_faulty[target_column].values

    df_faulty.to_csv(output_csv, index=False)
    print(f"Sensor fault data (noisy '{target_column}') saved to '{output_csv}'")

    # --- Plotting Comparison ---
    time_steps = np.arange(len(original_signal_normal))

    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, original_signal_normal, label=f'Normal - {target_column}', alpha=0.7)
    plt.plot(time_steps, faulty_signal, label=f'Faulty (Sensor Noisy) - {target_column}', linestyle='--', alpha=0.8)
    plt.title(f'Comparison: Normal vs. Sensor Fault (Noisy {target_column}) on Test Data')
    plt.xlabel('Time Step / Time (s)')
    plt.ylabel('Signal Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    normal_data_file = "vae_input_data.csv"
    faulty_data_output_file = "sensor_fault_v3_noisy.csv"
    noise_multiplier = 1.5

    apply_v3_noisy_fault(input_csv=normal_data_file,
                         output_csv=faulty_data_output_file,
                         noise_factor=noise_multiplier)

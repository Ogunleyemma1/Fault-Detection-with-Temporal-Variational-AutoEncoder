import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import structure4DOF
import training_vae
import testing_vae
import compare_initial_windows
import validation_vae

def main():
    print("Running structural simulation...")
    # Correct: Only this, never run_simulation()
    structure4DOF.generate_default_datasets(plot_training=False)

    print("Training Temporal VAE...")
    training_vae.train_vae()

    print("Validating and optimizing thresholds...")
    validation_vae.main()

    print("Testing Temporal VAE...")
    #testing_vae.main()

    print("Plotting reconstruction error comparison over time windows...")
    #compare_initial_windows.plot_reconstruction_window_comparison()

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()

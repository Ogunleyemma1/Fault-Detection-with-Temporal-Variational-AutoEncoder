# main.py

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import structure1DOF         # Simulation module to generate displacement, velocity, and acceleration data
import training_vae          # VAE training module
import testing_vae           # VAE testing module (reconstruction)
import compare_initial_windows  # Error comparison module

def main():
    print("Running structural simulation...")
    structure1DOF.run_simulation()

    print("Training Temporal VAE...")
    training_vae.train_vae()

    print("Testing Temporal VAE...")
    testing_vae.test_vae()

    print("Plotting reconstruction error comparison over time windows...")
    compare_initial_windows.plot_reconstruction_window_comparison()

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()

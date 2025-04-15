# main.py

# ---------------------------------------------
# This script orchestrates the entire VAE pipeline:
# 1. Runs the 4DOF structural simulation to generate input data.
# 2. Trains a Temporal Variational Autoencoder (VAE).
# 3. Tests the trained VAE and saves the reconstruction output.
# ---------------------------------------------
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import structure4DOF         # Simulation module to generate displacement, velocity, and acceleration data
import training_vae          # VAE training module
import testing_vae           # VAE testing module (reconstruction)

def main():
    print("Running structural simulation...")
    #structure4DOF.run_simulation()  # Step 1: Generates vae_input_data.csv

    print("Training Temporal VAE...")
    #training_vae.train_vae()        # Step 2: Trains and saves the VAE model

    print("Testing Temporal VAE...")
    testing_vae.test_vae()          # Step 3: Loads the trained model and reconstructs the input

    print("Pipeline completed successfully.")

# Entry point
if __name__ == "__main__":
    main()
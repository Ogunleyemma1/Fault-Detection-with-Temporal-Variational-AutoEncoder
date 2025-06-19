import os
import pandas as pd

# === CONFIGURATION ===
HEALTHY_DIR = os.path.join(
    os.getcwd(), "data_generation", "healthy_runs"
)

# === CHECK WORKING DIRECTORY ===
print("Current working directory:", os.getcwd())

# === LIST FILES IN HEALTHY_DIR ===
print("Files in healthy_runs directory:")
for f in os.listdir(HEALTHY_DIR):
    print("  ", f)

# === GATHER FILES ===
healthy_files = [os.path.join(HEALTHY_DIR, f)
                 for f in os.listdir(HEALTHY_DIR)
                 if f.startswith("healthy_seed") and f.endswith(".csv")]

print("\nFound healthy_seed CSV files:")
for f in healthy_files:
    print("  ", f)

if not healthy_files:
    print("\nERROR: No healthy_seed CSV files found in folder '{}'."
          " Are your files in the correct folder? Is the HEALTHY_DIR correct?".format(HEALTHY_DIR))
    import sys; sys.exit(1)

# === TRY TO LOAD FILES ===
dfs = []
for f in healthy_files:
    print(f"\nReading file: {f}")
    try:
        df = pd.read_csv(f)
        print(f"  Shape: {df.shape}")
        if df.empty:
            print(f"  WARNING: File '{f}' is empty!")
        else:
            print("  First 2 rows:\n", df.head(2))
        dfs.append(df)
    except Exception as e:
        print(f"  ERROR reading '{f}': {e}")

# === CONCATENATE DATAFRAMES ===
if not dfs:
    print("ERROR: All files were empty or failed to load.")
    import sys; sys.exit(1)

all_healthy = pd.concat(dfs, axis=0, ignore_index=True)
print(f"\n=== SUMMARY ===\nTotal samples loaded: {len(all_healthy)}")
print("First 5 rows of combined DataFrame:\n", all_healthy.head())

# Extra: Per-column stats
print("\nColumn means:\n", all_healthy.mean())
print("\nColumn std:\n", all_healthy.std())
print("\nColumn min:\n", all_healthy.min())
print("\nColumn max:\n", all_healthy.max())

# (stop here for inspection)

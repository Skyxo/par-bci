
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# PATHS
PRETRAIN_DB = "EEGnet/PRETRAIN_DATABASE_3CLASS.npz"
USER_DATA_DIR = "processed_data_3_class"

def print_stats(name, X):
    print(f"--- {name} ---")
    print(f"Shape: {X.shape}")
    print(f"Mean: {np.mean(X):.5e}")
    print(f"Std : {np.std(X):.5e}")
    print(f"Min : {np.min(X):.5e}")
    print(f"Max : {np.max(X):.5e}")
    
    # Check if likely uV or V
    max_val = np.max(np.abs(X))
    if max_val > 1.0:
        print("👉 Likely Scale: MICROVOLTS (uV) (or Raw integers)")
    elif max_val < 1e-3:
        print("👉 Likely Scale: VOLTS (V)")
    else:
        print("👉 Likely Scale: UNKNOWN / STANDARDIZED")
    print("")

def main():
    print("🔍 DATA INSPECTION REPORT\n")

    # 1. Inspect Physionet (Pre-train)
    if os.path.exists(PRETRAIN_DB):
        data = np.load(PRETRAIN_DB)
        X_physio = data['X']
        print_stats("Physionet (Pre-training Data)", X_physio)
    else:
        print(f"❌ Pretrain DB not found: {PRETRAIN_DB}")

    # 2. Inspect User Data
    X_files = glob.glob(os.path.join(USER_DATA_DIR, "X_*.npy"))
    if X_files:
        # Load first file
        f = X_files[0]
        print(f"Loading user file: {f}")
        X_user = np.load(f)
        print_stats("User Data (Validation)", X_user)
        
        # 3. Visual Comparison (Histogram)
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(X_physio.flatten(), bins=100, color='blue', alpha=0.7, label='Physionet')
        plt.title(f"Physionet Distribution\n(Std: {np.std(X_physio):.2e})")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.hist(X_user.flatten(), bins=100, color='orange', alpha=0.7, label='User')
        plt.title(f"User Data Distribution\n(Std: {np.std(X_user):.2e})")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("data_distribution_comparison.png")
        print("📊 Saved distribution plot to data_distribution_comparison.png")
        
    else:
        print(f"❌ No user data found in {USER_DATA_DIR}")

if __name__ == "__main__":
    main()

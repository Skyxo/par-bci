import os
import glob
import subprocess
import sys

# Define Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS_DIR = os.path.join(BASE_DIR, "tools")
PREPROCESS_SCRIPT = os.path.join(TOOLS_DIR, "preprocess_data_to_npy.py")
VISUALIZE_SCRIPT = os.path.join(TOOLS_DIR, "visualize_correlation", "visualize_envelope_correlation.py")
RESULTS_DIR = os.path.join(TOOLS_DIR, "visualize_correlation", "results")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")

def main():
    print("=== 🚀 STARTING BATCH ANALYSIS ===")
    
    # 1. Run Preprocessing (Ensures all CSVs are converted to NPY)
    # The preprocess_data_to_npy.py script processes ALL CSVs in the root
    print(f"Running Preprocessing Script: {PREPROCESS_SCRIPT}")
    subprocess.run([sys.executable, PREPROCESS_SCRIPT], cwd=BASE_DIR, check=True)
    
    # 2. Find all generated X_*.npy files
    search_pattern = os.path.join(PROCESSED_DATA_DIR, "X_EEG_Session_*.npy")
    npy_files = glob.glob(search_pattern)
    
    if not npy_files:
        print("❌ No processed .npy files found!")
        return

    print(f"Found {len(npy_files)} sessions to analyze.")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 3. Run Visualization for each file
    for npy_file in sorted(npy_files):
        print(f"\n--- Analyzing: {os.path.basename(npy_file)} ---")
        try:
            subprocess.run([
                sys.executable, 
                VISUALIZE_SCRIPT, 
                npy_file, 
                "--output_dir", RESULTS_DIR
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error processing {npy_file}: {e}")
            
    print(f"\n✅ BATCH ANALYSIS COMPLETE. Results saved in {RESULTS_DIR}")

if __name__ == "__main__":
    main()

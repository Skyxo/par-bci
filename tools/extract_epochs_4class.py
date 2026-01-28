"""
DATA PREPROCESSOR (4 CLASSES) 🧹
================================
Extracts 3-second epochs for:
- Left Hand (1)
- Right Hand (2)
- Feet (3)
- Rest (10)

Strictly aligns with Pre-training input shape (N, 8, 750).
"""

import pandas as pd
import numpy as np
import glob
import os
import mne

# CONFIGURATION
RAW_DIR = "." # Current dir
OUTPUT_DIR = "processed_data"
CH_NAMES = ['FC3', 'FC4', 'CP3', 'Cz', 'C3', 'C4', 'Pz', 'CP4']
SFREQ = 250

# PROCESSING SETTINGS
# Strict Action Window [0s - 3s] -> 750 samples
TMIN = 0.0 
TMAX = 3.0 
# Filters
L_FREQ = 2.0  # Match PhysioNet (2Hz)
H_FREQ = 40.0 # Match PhysioNet (40Hz)

def preprocess_all():
    print("=== 🧹 STARTING DATA CLEANING (4 CLASSES | 3.0s) ===")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    files = glob.glob(os.path.join(RAW_DIR, "EEG_Session_*.csv"))
    if not files:
        print("❌ No CSV files found.")
        return

    print(f"📂 Found {len(files)} files.")
    
    # Calculate samples for the window
    n_samples = int(TMAX * SFREQ) # 750
    
    for fname in files:
        base_name = os.path.splitext(os.path.basename(fname))[0]
        print(f"   -> Processing {base_name}...")
        
        file_X_list = []
        file_y_list = []
        
        try:
            # Load CSV
            try:
                 df = pd.read_csv(fname, sep='\t', header=None)
            except:
                 df = pd.read_csv(fname, sep=',', header=None)
            
            if df.shape[1] < 24:
                print("      ⚠️ Invalid columns.")
                continue

            # Check & Scale Units (uV to Volts)
            raw_data = df.iloc[:, 1:9].values.T
            if np.mean(np.abs(raw_data)) > 100: # Heuristic for uV
                raw_data = raw_data * 1e-6
            
            # Apply Filtering
            info = mne.create_info(CH_NAMES, SFREQ, 'eeg')
            raw = mne.io.RawArray(raw_data, info, verbose=False)
            raw.notch_filter(50, verbose=False)
            raw.filter(L_FREQ, H_FREQ, verbose=False)
            
            clean_data = raw.get_data() # (8, n_total)
            
            # Extract Markers
            markers = df.iloc[:, 23].values
            diff_markers = np.diff(markers, prepend=0)
            
            # Find ONSETS of markers 1, 2, 3, 10
            starts = np.where(np.isin(markers, [1, 2, 3, 10]) & (diff_markers != 0))[0]
            
            nb_extracted = 0
            for start_idx in starts:
                label = int(markers[start_idx])
                
                # UNIFORM WINDOWING (3.0s for EVERYONE)
                
                # Slice indices
                i_start = start_idx
                i_end = start_idx + n_samples
                
                # Check bounds
                if i_end <= clean_data.shape[1]:
                    epoch = clean_data[:, i_start:i_end]
                    
                    file_X_list.append(epoch)
                    file_y_list.append(label)
                    nb_extracted += 1
            
            print(f"      ✅ Extracted {nb_extracted} epochs.")
            
            if file_X_list:
                X_arr = np.array(file_X_list)
                y_arr = np.array(file_y_list)
                
                out_name_X = os.path.join(OUTPUT_DIR, f"X_{base_name}.npy")
                out_name_y = os.path.join(OUTPUT_DIR, f"y_{base_name}.npy")
                
                np.save(out_name_X, X_arr)
                np.save(out_name_y, y_arr)
                print(f"      💾 Saved: {out_name_X} (Shape: {X_arr.shape})")

        except Exception as e:
            print(f"      ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n✅ PREPROCESSING COMPLETE.")

if __name__ == "__main__":
    preprocess_all()

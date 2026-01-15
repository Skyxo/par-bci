"""
DATA PREPROCESSOR 🧹
====================
Converts raw OpenBCI CSV files into clean, ready-to-use Numpy arrays.
Solves the "Pulse Marker" issue by manually slicing data.

OUTPUT:
- processed_data/X.npy : The EEG epochs (N_epochs, N_channels, N_timepoints)
- processed_data/y.npy : The labels (0=Left, 1=Right, 2=Feet)
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
# Strict Action Window [0s - 4s]
TMIN = 0.0 
TMAX = 4.0 
# Filters
L_FREQ = 1.0  # High-pass to remove drift
H_FREQ = 40.0 # Low-pass to remove mains/muscle HF noise

def preprocess_all():
    print("=== 🧹 STARTING DATA CLEANING (STRICT 0-4s) ===")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    files = glob.glob(os.path.join(RAW_DIR, "EEG_Session_*.csv"))
    if not files:
        print("❌ No CSV files found.")
        return

    print(f"📂 Found {len(files)} files.")
    
    X_list = []
    y_list = []
    
    # Calculate samples for the window
    # TMIN=0 means start index is the marker index
    n_post = int(TMAX * SFREQ)
    total_samples = n_post # Since n_pre is 0
    
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
            
            # MNE find_events equivalent logic for Pulse
            # We also look for Marker 10 (Rest) from the first session
            starts = np.where(np.isin(markers, [1, 2, 3, 10]) & (diff_markers != 0))[0]
            
            nb_extracted = 0
            for start_idx in starts:
                label = int(markers[start_idx])
                
                # DYNAMIC WINDOWING
                # Action (1,2,3) -> 4.0s
                # Rest (10) -> 1.9s (Valid gap confirmed by inspection: Min 1.92s)
                if label == 10:
                    n_samples = int(1.9 * SFREQ)
                else:
                    n_samples = int(4.0 * SFREQ)
                
                # Slice indices
                i_start = start_idx
                i_end = start_idx + n_samples
                
                # Check bounds
                if i_end <= clean_data.shape[1]:
                    epoch = clean_data[:, i_start:i_end]
                    
                    # PADDING FOR REST
                    if epoch.shape[1] < n_post:
                        full_epoch = np.zeros((clean_data.shape[0], n_post))
                        full_epoch[:, :epoch.shape[1]] = epoch
                        epoch = full_epoch
                    
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
                print(f"      💾 Saved: {out_name_X}")

        except Exception as e:
            print(f"      ❌ Error: {e}")
            
    print("\n✅ PREPROCESSING COMPLETE on all files individually.")

if __name__ == "__main__":
    preprocess_all()

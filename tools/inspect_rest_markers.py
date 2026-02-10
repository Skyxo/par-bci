import pandas as pd
import numpy as np
import os

# Target File
FNAME = "EEG_Session_2026-01-14_13-35.csv"

def inspect_markers():
    print(f"=== 🕵️ MARKER INSPECTION: {FNAME} ===")
    
    try:
        # Load Data
        print("Loading CSV...")
        try:
            df = pd.read_csv(FNAME, sep='\t', header=None)
        except:
            df = pd.read_csv(FNAME, sep=',', header=None)
            
        markers = df.iloc[:, 23].values
        
        # indices where marker == 10
        idx_10 = np.where(markers == 10)[0]
        
        if len(idx_10) == 0:
            print("❌ No Marker 10 found in this file.")
            return

        print(f"✅ Found {len(idx_10)} samples with Marker 10.")
        
        # Check if they are contiguous (Block) or sparse (Pulse)
        diff_idx = np.diff(idx_10)
        is_block = np.all(diff_idx == 1)
        
        if is_block and len(idx_10) > 1:
            print("Layout: BLOCK (Continuous) 🧱")
            duration_samples = len(idx_10)
            duration_sec = duration_samples / 250.0
            print(f"Total Block Duration: {duration_sec:.2f}s")
        else:
            print("Layout: PULSE (Sparse/Discrete) ⚡")
            
            # Analyze interval to NEXT markers
            # Get all markers
            diff_all = np.diff(markers, prepend=0)
            starts_all = np.where(diff_all != 0)[0] # Indices where marker changes
            
            print("\n--- Intervals between Marker 10 and NEXT Marker ---")
            deltas = []
            
            for i in range(len(starts_all)-1):
                idx = starts_all[i]
                val = markers[idx]
                
                if val == 10:
                    # Find next NON-ZERO marker
                    # We search forward from i+1
                    found_next = False
                    for k in range(i+1, len(starts_all)):
                        idx_next = starts_all[k]
                        val_next = markers[idx_next]
                        if val_next != 0:
                            delta_samples = idx_next - idx
                            delta_sec = delta_samples / 250.0
                            deltas.append(delta_sec)
                            found_next = True
                            break
                    if not found_next:
                         pass # End of file
            
            if deltas:
                d_arr = np.array(deltas)
                print(f"Count: {len(d_arr)}")
                print(f"MIN Interval: {d_arr.min():.3f}s")
                print(f"MAX Interval: {d_arr.max():.3f}s")
                print(f"MEAN Interval: {d_arr.mean():.3f}s")
                print(f"MEDIAN Interval: {np.median(d_arr):.3f}s")
                
                print("\nHistogram (ASCII):")
                hist, bins = np.histogram(d_arr, bins=5)
                for count, edge in zip(hist, bins):
                    print(f"{edge:.2f}s : {'#'*count}")
            else:
                print("No following markers found for Marker 10.")
                    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_markers()

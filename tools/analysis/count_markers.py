import pandas as pd
import numpy as np
import os

files = [
    "data_markiv/EEG_Session_2026-02-10_16-09.csv",
    "data_markiv/EEG_Session_2026-02-03_18-06.csv",
    "data_markiv/EEG_Session_2026-02-10_17-10.csv"
]

def count_markers(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    try:
        df = pd.read_csv(filepath, sep='\t', header=None, index_col=False)
        # Marker column is 23 (last one)
        markers = df.iloc[:, 23].to_numpy()
        
        # Find non-zero markers
        indices = np.where(markers != 0)[0]
        values = markers[indices]
        
        counts = {
            1: 0, # Left
            2: 0, # Right
            3: 0, # Feet
            10: 0, # Rest
            99: 0 # End Run
        }
        
        for v in values:
            v_int = int(v)
            if v_int in counts:
                counts[v_int] += 1
                
        total_trials = counts[1] + counts[2] + counts[3] + counts[10]
        
        print(f"--- {os.path.basename(filepath)} ---")
        print(f"Total Lines: {len(df)}")
        print(f"Left (1): {counts[1]}")
        print(f"Right (2): {counts[2]}")
        print(f"Feet (3): {counts[3]}")
        print(f"Rest (10): {counts[10]}")
        print(f"EndRuns (99): {counts[99]}")
        print(f"Total Valid Trials: {total_trials}")
        print("-" * 30)
        
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")

if __name__ == "__main__":
    print("Analyzing Session Markers...")
    for f in files:
        count_markers(f)

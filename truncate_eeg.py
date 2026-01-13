import pandas as pd
import numpy as np
import os

# Configuration
FILENAME = "EEG_Session_2026-01-13_15-30.csv"
SFREQ = 250
START_SECONDS = 140

def truncate_file():
    if not os.path.exists(FILENAME):
        print(f"File {FILENAME} not found.")
        return

    print(f"Reading {FILENAME}...")
    try:
        # Try tab first as seen in other scripts
        df = pd.read_csv(FILENAME, sep='\t', header=None)
    except:
        # Fallback to comma
        df = pd.read_csv(FILENAME, sep=',', header=None)
    
    print(f"Original shape: {df.shape}")
    
    start_index = int(START_SECONDS * SFREQ)
    
    if start_index >= len(df):
        print("Error: Start time is beyond the end of the file.")
        return

    df_truncated = df.iloc[start_index:]
    print(f"Truncated shape: {df_truncated.shape} (Removed first {START_SECONDS}s / {start_index} samples)")
    
    # Save to a new file first to be safe
    new_filename = f"truncated_{FILENAME}"
    print(f"Saving to {new_filename}...")
    df_truncated.to_csv(new_filename, sep='\t', header=False, index=False)
    
    # Optional: Replace original
    # os.replace(new_filename, FILENAME)
    print("Done. Created truncated file.")
    
    # We will replace the original as requested implicitly, but keep a backup
    backup_name = f"backup_{FILENAME}"
    if not os.path.exists(backup_name):
        os.rename(FILENAME, backup_name)
        os.rename(new_filename, FILENAME)
        print(f"Original file backed up to {backup_name} and replaced with truncated version.")
    else:
        print(f"Backup {backup_name} already exists. Please verify manually.")
        os.rename(new_filename, FILENAME) # Overwrite if backup exists

if __name__ == "__main__":
    truncate_file()

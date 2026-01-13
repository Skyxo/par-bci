import pandas as pd
import numpy as np
import mne
import os

# CONFIG
# Get the directory of this script to make paths robust
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'EEG_Session_2026-01-13_15-30.csv')
SAVE_PATH_X = os.path.join(PROJECT_ROOT, 'data/processed/X.npy')
SAVE_PATH_Y = os.path.join(PROJECT_ROOT, 'data/processed/y.npy')
SFREQ = 250
CH_NAMES = ['Cz', 'FCz', 'P3', 'Pz', 'C3', 'C4', 'O1', 'P4']

def preprocess_data(file_path):
    print(f"Loading {file_path}...")
    
    # 1. READ CSV (BrainFlow format)
    # BrainFlow CSVs usually have no header and use tab or comma
    # Columns 1-8 (index 1-8) are EEG channels for Cyton.
    # Column 23 (index 23) is the Marker channel.
    try:
        data = pd.read_csv(file_path, sep='\t', header=None)
    except:
        data = pd.read_csv(file_path, sep=',', header=None)
        
    print(f"Data shape: {data.shape}")
    
    # Extract EEG and Scale (BrainFlow returns uV, MNE expects Volts)
    eeg_data = data.iloc[:, 1:9].values.T / 1e6 # (n_channels, n_samples)
    
    # Extract Markers
    # Marker channel is usually the last one (index 23 for Cyton with default settings)
    # or the one before timestamp. Checking index 23 based on previous analysis.
    marker_channel_idx = 23
    if marker_channel_idx >= data.shape[1]:
        marker_channel_idx = -2 # Fallback if columns differ
        
    raw_markers = data.iloc[:, marker_channel_idx].values
    
    # Create MNE Info
    info = mne.create_info(ch_names=CH_NAMES, sfreq=SFREQ, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info)
    
    # 2. FILTERING
    # First, Notch filter to kill 50Hz mains noise (and harmonic 100Hz)
    # Nyquist is 125Hz, so we only filter 50 and 100.
    raw.notch_filter([50, 100], fir_design='firwin')
    
    # Then, 8-30 Hz (Golden Band) for Motor Imagery
    raw.filter(8., 30., fir_design='firwin', skip_by_annotation='edge')
    
    # 3. EPOCHING
    # Find events where marker changes from 0 to something else
    # Markers in prog.py: 1 (Left), 2 (Right), 3 (Feet), 10 (Baseline)
    
    # Convert separate marker channel to MNE events array: [sample, 0, id]
    # We only care about the ONSET of the marker.
    diff_markers = np.diff(raw_markers, prepend=0)
    # Events start where diff > 0 (assuming markers go 0 -> 1 -> 0)
    # Note: In prog.py, markers might be held? Let's check. 
    # Usually BoardShim.insert_marker adds a single sample marker.
    # But if standard marker logic is used, it's safer to find non-zero indices.
    
    # Robust event detection: find indices where marker != 0
    # BrainFlow insert_marker puts the value for ONE sample.
    event_indices = np.where(raw_markers != 0)[0]
    event_ids = raw_markers[event_indices].astype(int)
    
    if len(event_indices) == 0:
        print("ERROR: No events found in marker channel!")
        return None, None
        
    events = np.column_stack((event_indices, np.zeros_like(event_indices), event_ids))
    
    print(f"Found {len(events)} events: {np.unique(event_ids)}")
    
    # Define Event IDs for MNE
    # We want to classify Left vs Right vs Feet vs Rest.
    # Rest is the baseline period (marker 10)
    event_id_map = {'Left': 1, 'Right': 2, 'Feet': 3, 'Rest': 10}
    
    try:
        epochs = mne.Epochs(raw, events, event_id=event_id_map, 
                            tmin=0, tmax=3.0, baseline=None, preload=True)
        
        X = epochs.get_data()
        y = epochs.events[:, -1]
        return X, y
        
    except ValueError as e:
        print(f"Epoching Error (maybe missing classes?): {e}")
        return None, None

if __name__ == "__main__":
    if not os.path.exists(RAW_DATA_PATH):
        print(f"File not found: {RAW_DATA_PATH}")
        print("Please place your CSV file there.")
    else:
        X, y = preprocess_data(RAW_DATA_PATH)
        print(f"Saving X shape: {X.shape}, y shape: {y.shape}")
        np.save(SAVE_PATH_X, X)
        np.save(SAVE_PATH_Y, y)
        print("Preprocessing Done.")

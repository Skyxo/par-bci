import pandas as pd
import numpy as np
import mne
import os

# --- CONFIGURATION (Shared) ---
CH_NAMES = ['Cz', 'FCz', 'P3', 'Pz', 'C3', 'C4', 'O1', 'P4']
# Note: Previous scripts used different order? 
# process_all_sessions used: ['Cz', 'FCz', 'P3', 'Pz', 'C3', 'C4', 'O1', 'P4']
# visualize_envelope used: ['FC3', 'FC4', 'CP3', 'Cz', 'C3', 'C4', 'Pz', 'CP4']
# CRITICAL: We must match the channel names to the actual data columns.
# data_structure.md says: 1-8 are EEG channels (FC3, FC4, CP3, Cz, C3, C4, Pz, CP4) ???
# WAIT. Let's re-read data_structure.md to be sure about the order.
# The user's `data_structure.md` says:
# "1-8 EEG Channels Données EEG brutes (FC3, FC4, CP3, Cz, C3, C4, Pz, CP4)" (Indices 1 to 8)
#
# My `process_all_sessions.py` used: ['Cz', 'FCz', 'P3', 'Pz', 'C3', 'C4', 'O1', 'P4']
# This seems to be a mismatch if data_structure.md is correct. 
# However, the user might have set up OpenBCI differently.
# But `visualize_envelope` used: ['FC3', 'FC4', 'CP3', 'Cz', 'C3', 'C4', 'Pz', 'CP4']
# I will use the one from `data_structure.md` to be safe/consistent with documentation.

CH_NAMES_DOC = ['FC3', 'FC4', 'CP3', 'Cz', 'C3', 'C4', 'Pz', 'CP4']
SFREQ = 250
EVENT_ID_MAP = {'Left': 1, 'Right': 2, 'Feet': 3, 'Rest': 10}

def load_and_epoch_csv(file_path):
    """
    Loads a CSV file, filters it (1-100Hz, 50Hz Notch), and epochs it (0-3s).
    Returns X (epochs, channels, times) and y (events).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    print(f"Loading {os.path.basename(file_path)}...")
    
    # 1. READ CSV
    try:
        data = pd.read_csv(file_path, sep='\t', header=None)
    except:
        data = pd.read_csv(file_path, sep=',', header=None)
        
    if data.empty or data.shape[1] < 9:
        raise ValueError("File is empty or not enough columns")
        
    # Extract EEG (uV)
    # Assuming columns 1-8 are the EEG channels
    eeg_data = data.iloc[:, 1:9].values.T
    
    # Extract Markers
    marker_col = 23 if data.shape[1] > 23 else -1
    raw_markers = data.iloc[:, marker_col].values
    
    # Create MNE Raw (Units: uV but treating as 'eeg')
    info = mne.create_info(ch_names=CH_NAMES_DOC, sfreq=SFREQ, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info, verbose=False)
    
    # 2. FILTERING
    # Notch 50 Hz
    raw.notch_filter([50], fir_design='firwin', verbose=False)
    # Bandpass 1-100 Hz
    raw.filter(1., 100., fir_design='firwin', skip_by_annotation='edge', verbose=False)
    
    # 3. EPOCHING
    event_indices = np.where(raw_markers != 0)[0]
    if len(event_indices) == 0:
        print("No events found.")
        return np.array([]), np.array([])
        
    event_ids = raw_markers[event_indices].astype(int)
    events = np.column_stack((event_indices, np.zeros_like(event_indices), event_ids))
    
    # Filter for known classes
    unique_events = np.unique(event_ids)
    current_ids = {k: v for k, v in EVENT_ID_MAP.items() if v in unique_events}
    
    if not current_ids:
        print("No relevant classes found.")
        return np.array([]), np.array([])
        
    # Epoching
    # 3 seconds -> 750 samples
    tmax = 3.0 - (1/SFREQ)
    epochs = mne.Epochs(raw, events, event_id=current_ids,
                        tmin=0, tmax=tmax, baseline=None, preload=True, verbose=False)
                        
    if len(epochs) == 0:
        return np.array([]), np.array([])
        
    X = epochs.get_data() # (N, 8, 750)
    y = epochs.events[:, -1]
    
    print(f"  -> X: {X.shape}, y: {y.shape} (Range: {X.min():.1f} to {X.max():.1f} uV)")
    
    return X, y, info

import numpy as np
import pandas as pd
import mne
import torch
from torch.utils.data import Dataset, DataLoader
from moabb.datasets import BNCI2014001

# OpenBCI Channels
OPENBCI_CHANNELS = ['FC3', 'FC4', 'CP3', 'Cz', 'C3', 'C4', 'Pz', 'CP4']
SFREQ = 250

def load_bnci_data(subject_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9], tmin=0, tmax=4.0):
    """
    Load BNCI data manually to bypass moabb.paradigms filtering issues.
    """
    dataset = BNCI2014001()
    dataset.subject_list = list(subject_ids)
    
    all_events = []
    all_data = [] # List of (n_channels, n_times) arrays matching our subset
    
    # 1. Load Data per subject
    for subj in dataset.subject_list:
        subj_data = dataset.get_data(subjects=[subj])
        for session_name, runs in subj_data[subj].items():
            for run_name, raw in runs.items():
                # raw: MNE Raw object
                # Resample if needed (BNCI is 250Hz, so usually no-op)
                if raw.info['sfreq'] != SFREQ:
                    raw.resample(SFREQ, verbose=False)
                
                # 2. Pick Channels
                bnci_chs = raw.ch_names
                common_chs = [ch for ch in OPENBCI_CHANNELS if ch in bnci_chs]
                if len(common_chs) != len(OPENBCI_CHANNELS):
                    # Missing channels, skip or handle?
                    # BNCI should have them.
                    pass
                
                raw.pick_channels(common_chs, ordered=False)
                raw.reorder_channels(OPENBCI_CHANNELS)
                
                # 3. Extract Events
                # BNCI annotations: 'left_hand', 'right_hand', 'feet', 'tongue'
                events_map = {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'tongue': 3}
                # We only want 0, 1, 2
                
                try:
                    events, event_id_map = mne.events_from_annotations(raw, verbose=False)
                except:
                    continue
                    
                # Filter events
                # event_id_map maps 'left_hand' -> Int
                # We need to map that Int -> 0/1/2
                
                # Reverse the map logic
                wanted_ids = {}
                for label, code in events_map.items():
                    if label in event_id_map:
                        wanted_ids[event_id_map[label]] = code
                        
                if not wanted_ids:
                    continue
                    
                # Create epochs manually or use MNE Epochs
                # Use MNE Epochs for convenience
                # Map event codes to ours (0,1,2,3)
                # Actually MNE Epochs needs a dict of int->int or we construct proper event array
                
                # Remap event second column (value)
                new_events = []
                for ev in events:
                    if ev[2] in wanted_ids:
                        code = wanted_ids[ev[2]]
                        if code < 3: # Keep only Left, Right, Feet
                            new_events.append([ev[0], 0, code])
                            
                if not new_events:
                    continue
                    
                new_events = np.array(new_events)
                
                # 4. Epoch
                # tmin, tmax
                epochs = mne.Epochs(raw, new_events, event_id=None, 
                                    tmin=tmin, tmax=tmax, baseline=None, 
                                    preload=True, verbose=False)
                
                # Get data (N, Ch, T)
                data = epochs.get_data() * 1e6 # V -> uV
                all_data.append(data)
                
                # Get labels (N,)
                labels = new_events[:, 2] # Already 0, 1, 2
                all_events.append(labels)

    if not all_data:
        raise ValueError("No BNCI data loaded. Check files/paths.")
        
    X_combined = np.concatenate(all_data, axis=0)
    y_combined = np.concatenate(all_events, axis=0)
    
    return X_combined, y_combined

def load_openbci_data(file_paths):
    """
    Load user OpenBCI CSVs and epoch them.
    """
    all_X = []
    all_y = []
    
    for fp in file_paths:
        try:
            df = pd.read_csv(fp, sep='\t', header=None, index_col=False)
            
            # Data: Cols 1-8
            data = df.iloc[:, 1:9].to_numpy().T / 1e6 # uV -> V
            
            # Markers: Col 23
            markers = df.iloc[:, 23].to_numpy()
            
            # Info
            info = mne.create_info(OPENBCI_CHANNELS, SFREQ, 'eeg')
            raw = mne.io.RawArray(data, info, verbose=False)
            
            # Find events
            stim_indices = np.where(markers != 0)[0]
            stim_values = markers[stim_indices].astype(int)
            
            # We only care about 1 (Left), 2 (Right), 3 (Feet) for transfer
            # (Marker 10 is Rest)
            mask = np.isin(stim_values, [1, 2, 3])
            stim_indices = stim_indices[mask]
            stim_values = stim_values[mask]
            
            if len(stim_values) == 0:
                continue
                
            events = np.column_stack((stim_indices, np.zeros_like(stim_indices, dtype=int), stim_values))
            
            # Epoching
            # MNE expects event_id map
            event_id = {'Left': 1, 'Right': 2, 'Feet': 3}
            
            epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=4.0, baseline=None, preload=True, verbose=False)
            
            all_X.append(epochs.get_data())
            # Map 1->0, 2->1, 3->2
            y_mapped = (epochs.events[:, 2] - 1)
            all_y.append(y_mapped)
            
        except Exception as e:
            print(f"Error loading {fp}: {e}")
            
    if not all_X:
        return None, None
        
    X_combined = np.concatenate(all_X, axis=0) # (N, Ch, T)
    y_combined = np.concatenate(all_y, axis=0)
    
    return X_combined, y_combined

class EEGDataset(Dataset):
    def __init__(self, X, y, transform=True):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.longlong)
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            # Apply standardization per epoch
            # (Note: preprocessing.standardize_epochs works on batch, here we do single)
            # Just simple z-score
            mean = np.mean(x, axis=1, keepdims=True)
            std = np.std(x, axis=1, keepdims=True)
            x = (x - mean) / (std + 1e-4)
            
        return torch.from_numpy(x), self.y[idx]

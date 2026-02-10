import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
import glob
import os

print("=== SPECTROGRAM: MANUAL SLICING MODE ===")

files = glob.glob("EEG_Session_*.csv")
if not files:
    print("❌ No files found.")
    exit()

SFREQ = 250
# Window: -1s to +3s
TMIN = -1.0
TMAX = 3.0
N_PRE = int(abs(TMIN) * SFREQ)
N_POST = int(TMAX * SFREQ)
TOTAL_SAMPLES = N_PRE + N_POST

all_epochs_data = [] # (n_epochs, n_channels, n_times)
all_events = []

for fname in files:
    try:
        try:
            df = pd.read_csv(fname, sep='\t', header=None)
        except:
            df = pd.read_csv(fname, sep=',', header=None)
            
        if df.shape[1] < 24: continue

        # Data Setup
        raw_values = df.iloc[:, 1:9].values.T # (8, n_samples)
        # Scale if needed
        if np.mean(np.abs(raw_values)) > 100: # uV likely
            raw_values = raw_values * 1e-6
            
        markers = df.iloc[:, 23].values
        
        # Find events
        diff_markers = np.diff(markers, prepend=0)
        starts = np.where((diff_markers != 0) & (markers != 0))[0]
        
        n_file_samples = raw_values.shape[1]
        
        for idx in starts:
            start_samp = idx - N_PRE
            end_samp = idx + N_POST
            
            # Check bounds
            if start_samp >= 0 and end_samp <= n_file_samples:
                epoch_data = raw_values[:, start_samp:end_samp]
                
                # Check shapes match exactly
                if epoch_data.shape[1] == TOTAL_SAMPLES:
                    all_epochs_data.append(epoch_data)
                    # Event: [index, 0, id]
                    evt_id = int(markers[idx])
                    # Only keep 1, 2, 3
                    if evt_id in [1, 2, 3]:
                        all_events.append([len(all_epochs_data)-1, 0, evt_id])
                    else:
                         all_epochs_data.pop() # Remove if not relevant
                else:
                    pass # Size mismatch
            else:
                 pass # Out of bounds
                 
    except Exception as e:
        print(f"Error {fname}: {e}")

if not all_epochs_data:
    print("❌ No epochs extracted manually.")
    exit()

# Convert to Numpy
X = np.array(all_epochs_data) # (N, 8, T)
y = np.array(all_events) # (N, 3)

print(f"✅ Extracted {len(X)} epochs manually.")

# Create EpochsArray
info = mne.create_info(['FC3', 'FC4', 'CP3', 'Cz', 'C3', 'C4', 'Pz', 'CP4'], SFREQ, 'eeg')
epochs = mne.EpochsArray(X, info, events=y, event_id={'G':1, 'D':2, 'P':3}, tmin=TMIN, verbose=False)

# Filter Now (on epochs)
epochs.filter(1, 40, verbose=False)

# TFR
freqs = np.arange(4, 35, 1)
n_cycles = freqs / 2.

print("Computing TFR...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Indices for C3 (4) and C4 (5)
idx_c3 = 4
idx_c4 = 5

# --- RIGHT HAND (Class 2 'D') ---
if 'D' in epochs.event_id:
    print("Plotting Right Hand...")
    tfr = mne.time_frequency.tfr_morlet(epochs['D'], freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, average=True)
    tfr.plot([idx_c3], baseline=(-1.0, -0.1), mode='percent', title="Right Hand -> C3 (Left ctx)", axes=axes[0,0], show=False, cmap='RdBu_r')
    tfr.plot([idx_c4], baseline=(-1.0, -0.1), mode='percent', title="Right Hand -> C4 (Right ctx)", axes=axes[0,1], show=False, cmap='RdBu_r')

# --- LEFT HAND (Class 1 'G') ---
if 'G' in epochs.event_id:
    print("Plotting Left Hand...")
    tfr = mne.time_frequency.tfr_morlet(epochs['G'], freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, average=True)
    tfr.plot([idx_c3], baseline=(-1.0, -0.1), mode='percent', title="Left Hand -> C3 (Left ctx)", axes=axes[1,0], show=False, cmap='RdBu_r')
    tfr.plot([idx_c4], baseline=(-1.0, -0.1), mode='percent', title="Left Hand -> C4 (Right ctx)", axes=axes[1,1], show=False, cmap='RdBu_r')

plt.tight_layout()
plt.savefig("spectrogram_analysis.png")
print("✅ Saved spectrogram_analysis.png")

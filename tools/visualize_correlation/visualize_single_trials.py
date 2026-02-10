import numpy as np
import mne
import matplotlib.pyplot as plt
import os
import glob
from scipy.signal import hilbert

# --- CONFIGURATION ---
CH_NAMES = ['FC3', 'FC4', 'CP3', 'Cz', 'C3', 'C4', 'Pz', 'CP4']
SFREQ = 250
CLASS_NAMES = {1: 'Left', 2: 'Right', 3: 'Feet', 10: 'Rest'}
BANDS = {
    'Alpha (7-13Hz)': (7, 13),
    'Beta (13-31Hz)': (13, 31),
    'Gamma (71-91Hz)': (71, 91)
}

def compute_envelope_power(data, sfreq, l_freq, h_freq):
    # data: (n_channels, n_times)
    # Check if data length allows filtering
    if data.shape[1] < sfreq: # Less than 1 sec
        # Simple FFT power or just variance as fallback?
        # But here we have 3s (750 samples), should be fine.
        pad = 'reflect'
    else:
        pad = 'reflect'
        
    data_filt = mne.filter.filter_data(data, sfreq, l_freq, h_freq, verbose=False, pad=pad)
    envelope = np.abs(hilbert(data_filt, axis=-1)) ** 2
    return np.mean(envelope, axis=-1) # Average Power over time -> (n_channels,)

def main():
    # Identify Last Session File
    # X_EEG_Session_2026-02-05_18-02.npy
    # I will hardcode finding the last one or allow argument.
    processed_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'processed_data')
    files = sorted(glob.glob(os.path.join(processed_dir, 'X_EEG_Session_*.npy')))
    
    if not files:
        print("No files found.")
        return

    # Last file
    target_file = files[-1]
    base_name = os.path.basename(target_file).replace("X_EEG_Session_", "").replace(".npy", "")
    print(f"Targeting last session: {base_name}")
    
    X = np.load(target_file)
    y_path = target_file.replace("X_", "y_")
    y = np.load(y_path)
    
    print(f"Loaded {X.shape} trials.")
    
    # Output Directory
    output_dir = os.path.join(os.path.dirname(__file__), "results", f"single_trials_{base_name}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving images to: {output_dir}")
    
    # MNE Info
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(CH_NAMES, SFREQ, ch_types='eeg')
    info.set_montage(montage)
    
    # Iterate Trials
    n_trials = X.shape[0]
    
    for i in range(n_trials):
        trial_data = X[i] # (n_channels, n_times)
        label_id = int(y[i])
        label_name = CLASS_NAMES.get(label_id, f"Unknown ({label_id})")
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"Trial {i+1}/{n_trials} - Class: {label_name}", fontsize=16)
        
        for ax, (band_name, (l_freq, h_freq)) in zip(axes, BANDS.items()):
            # Compute Power
            # Note: We are filtering short signal (3s). Edge artifacts possible.
            # But sufficient for visual inspection.
            power = compute_envelope_power(trial_data, SFREQ, l_freq, h_freq)
            
            # Plot
            # Determine vlim for this specific trial/band to see contrast
            vmax = np.max(power)
            
            im, _ = mne.viz.plot_topomap(
                power, info, axes=ax, show=False, cmap='Reds', contours=0, sphere=0.12
            )
            ax.set_title(f"{band_name}\nMax: {vmax:.1f} uV²")
            
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"trial_{i+1:03d}_{label_name}.png")
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{n_trials}")

    print("Done.")

if __name__ == "__main__":
    main()

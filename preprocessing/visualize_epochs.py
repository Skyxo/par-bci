import numpy as np
import matplotlib.pyplot as plt

# CONFIG
X_PATH = '../data/processed/X.npy'
Y_PATH = '../data/processed/y.npy'
CH_NAMES = ['Cz', 'FCz', 'P3', 'Pz', 'C3', 'C4', 'O1', 'P4']
SFREQ = 250
CLASS_MAP = {1: 'Left', 2: 'Right', 3: 'Feet'}

def visualize_epochs():
    # 1. Load Data
    try:
        X = np.load(X_PATH)
        y = np.load(Y_PATH)
        print(f"Loaded X: {X.shape}, y: {y.shape}")
    except FileNotFoundError:
        print("Error: Files not found. Run preprocess_eeg.py first.")
        return

    n_trials, n_ch, n_times = X.shape
    time = np.arange(n_times) / SFREQ
    
    # 2. Compute Averages per Class
    unique_classes = np.unique(y)
    
    # --- PLOT 1: AVERAGE SIGNAL PER CHANNEL (COMPARING CLASSES) ---
    # We want to see how C3 differs for Left vs Right vs Feet
    # Best channels for this: C3 (Right Hand), C4 (Left Hand), Cz (Feet)
    
    # --- PLOT 1: PSD PER CHANNEL (Power Spectrum) ---
    # Motor Imagery is seen in Frequency Domain (Mu/Beta power drop), not Time Domain averages.
    from scipy.signal import welch
    
    target_channels = ['C3', 'C4', 'Cz']
    plt.figure(figsize=(15, 6*len(target_channels)))
    
    for i, ch_name in enumerate(target_channels):
        if ch_name not in CH_NAMES:
            continue
            
        ch_idx = CH_NAMES.index(ch_name)
        plt.subplot(len(target_channels), 1, i+1)
        
        for cls in unique_classes:
            # Filter trials for this class
            trials_cls = X[y == cls] # (n_trials, n_ch, n_times)
            
            # Compute PSD for each trial, then average
            # trials_cls[:, ch_idx, :] is (n_trials, n_times)
            freqs, psds = welch(trials_cls[:, ch_idx, :], fs=SFREQ, nperseg=SFREQ)
            avg_psd = np.mean(psds, axis=0)
            
            plt.plot(freqs, avg_psd, label=f'{CLASS_MAP.get(cls, cls)} (n={len(trials_cls)})')
            
        plt.title(f"Power Spectral Density (PSD) - Channel {ch_name}")
        plt.xlim(8, 30) # Focus on Mu/Beta
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (uV^2/Hz)")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # --- PLOT 2: TIME DOMAIN (ENVELOPE) ---
    # Show envelope of C3 to see ERD
    # (Optional, keeping PSD is usually enough)

if __name__ == "__main__":
    visualize_epochs()

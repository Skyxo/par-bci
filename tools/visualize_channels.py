import numpy as np
import mne
import matplotlib.pyplot as plt
import os
import glob

# --- CONFIGURATION ---
CH_NAMES = ['FC3', 'FC4', 'CP3', 'Cz', 'C3', 'C4', 'Pz', 'CP4']
SFREQ = 250
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "audit_results")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed_data")

# Channels to inspect
TARGET_chs = ['C3', 'Cz', 'C4'] 
# Classes
LABELS = {1: 'Gauche (L)', 2: 'Droite (R)', 3: 'Pieds (F)'}
COLORS = {1: 'blue', 2: 'red', 3: 'green'}

def load_data():
    X_files = glob.glob(os.path.join(PROCESSED_DIR, "X_*.npy"))
    if not X_files: return None, None
    
    X_list, y_list = [], []
    for f in X_files:
        base = os.path.basename(f).replace("X_", "y_")
        y_f = os.path.join(PROCESSED_DIR, base)
        if os.path.exists(y_f):
            X_list.append(np.load(f))
            y_list.append(np.load(y_f))
            
    if not X_list: return None, None
    return np.concatenate(X_list), np.concatenate(y_list)

def main():
    print("👀 Visualisation Spécifique Canaux...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    X, y = load_data()
    if X is None: return

    # Filter 1-40Hz
    X_filt = mne.filter.filter_data(X, SFREQ, 1, 40, verbose=False)
    
    # Crop active window [0.0 - 3.5s] (User Request)
    t_start, t_end = 0.0, 3.5
    idx_s, idx_e = int(t_start*SFREQ), int(t_end*SFREQ)
    X_crop = X_filt[:, :, idx_s:idx_e]

    # Create MNE Epochs
    info = mne.create_info(CH_NAMES, SFREQ, 'eeg')
    events = np.column_stack((np.arange(len(y)), np.zeros(len(y), dtype=int), y.astype(int)))
    epochs = mne.EpochsArray(X_crop, info, events=events, tmin=t_start, verbose=False)

    # Plot PSD for each Target Channel
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, ch_name in enumerate(TARGET_chs):
        ax = axes[i]
        
        # Check if channel exists (might have been dropped in data but here we assume full set)
        if ch_name not in epochs.ch_names:
            ax.set_title(f"{ch_name} Missing")
            continue
            
        ax.set_title(f"Canal : {ch_name}")
        
        for cls_id, cls_name in LABELS.items():
            # Get subset of epochs
            if str(cls_id) in epochs.event_id: # MNE quirk if passed event_id dict, but here events are raw ints
                # We select manually
                mask = (y == cls_id)
                if not np.any(mask): continue
                
                # Compute PSD
                # psd: (n_epochs, n_freqs)
                psd_obj = epochs[str(cls_id)].compute_psd(fmin=4, fmax=35, picks=[ch_name])
                psd_data = psd_obj.get_data(return_freqs=True) 
                
                # psd_data is (n_epochs, n_channels, n_freqs) or tuple depending on version
                # In modern MNE get_data returns (data, freqs)
                psds = psd_data[0] # (N, 1, F)
                freqs = psd_data[1]
                
                # Mean over epochs
                mean_psd = np.mean(psds, axis=0).flatten()
                
                # Convert to dB implies 10*log10, but usually we just plot power
                # Let's plot raw power or Log power. Log is better for spectrum
                # ax.plot(freqs, 10 * np.log10(mean_psd), label=cls_name, color=COLORS[cls_id])
                ax.plot(freqs, mean_psd, label=cls_name, color=COLORS[cls_id], linewidth=2)
        
        ax.set_xlabel("Fréquence (Hz)")
        ax.set_ylabel("Puissance (µV²/Hz)")
        ax.axvspan(8, 13, color='gray', alpha=0.2, label='Mu (8-13Hz)')
        ax.axvspan(18, 25, color='orange', alpha=0.1, label='Beta (18-25Hz)')
        if i == 0: ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Comparaison Spectrale par Canal et par Classe (0.0s - 3.5s)", fontsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, "channel_comparison_0s_3.5s.png")
    plt.savefig(save_path)
    print(f"✅ Image générée : {save_path}")

if __name__ == "__main__":
    main()

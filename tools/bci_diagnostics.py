import numpy as np
import mne
import matplotlib.pyplot as plt
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from sklearn.model_selection import cross_val_score, StratifiedKFold
import os
import glob
import sys

# --- CONFIGURATION ---
CH_NAMES = ['FC3', 'FC4', 'CP3', 'Cz', 'C3', 'C4', 'Pz', 'CP4']
SFREQ = 250
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "audit_results")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed_data")

def load_processed_data():
    print(f"📥 Loading data from: {PROCESSED_DIR}")
    X_files = glob.glob(os.path.join(PROCESSED_DIR, "X_*.npy"))
    if not X_files:
        print("❌ No .npy files found.")
        return None, None

    X_list = []
    y_list = []

    for f in X_files:
        base = os.path.basename(f).replace("X_", "y_")
        y_f = os.path.join(PROCESSED_DIR, base)
        
        if os.path.exists(y_f):
            X_part = np.load(f)
            y_part = np.load(y_f)
            X_list.append(X_part)
            y_list.append(y_part)

    if not X_list: return None, None
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    return X, y

def check_signal_quality(X, ch_names, sfreq):
    """Checks for dead channels, rails, and line noise."""
    print("\n🔍 --- 1. SIGNAL QUALITY AUDIT ---")
    
    # 1. Global RMS Amplitude
    # X shape: (Epochs, Channels, Time)
    # Concatenate all epochs to get continuous-like signal for stats
    X_cont = np.hstack(X) # (Channels, TotalTime)
    
    rms_per_ch = np.sqrt(np.mean(X_cont**2, axis=1))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(ch_names, rms_per_ch * 1e6) # Convert to uV
    ax.set_ylabel("RMS Amplitude (uV)")
    ax.set_title("Average Signal Amplitude per Channel\n(Normal range: 5-100 uV)")
    ax.axhline(y=1.0, color='r', linestyle='--', label='Dead Channel (<1uV)')
    ax.axhline(y=100.0, color='orange', linestyle='--', label='Noisy (>100uV)')
    ax.legend()
    
    # Color coding
    for bar, val in zip(bars, rms_per_ch * 1e6):
        if val < 1.0: bar.set_color('red') # Dead
        elif val > 100.0: bar.set_color('orange') # Noisy
        else: bar.set_color('green') # OK

    plt.savefig(os.path.join(OUTPUT_DIR, "diagnostic_amplitude.png"))
    plt.close()
    
    print("   -> Generated 'diagnostic_amplitude.png'")
    
    # 2. PSD Check (50Hz line noise)
    # Using MNE Raw for PSD calculation
    info = mne.create_info(ch_names, sfreq, 'eeg')
    raw = mne.io.RawArray(X_cont, info, verbose=False)
    
    fig_psd = raw.compute_psd(fmax=60).plot(show=False)
    plt.savefig(os.path.join(OUTPUT_DIR, "diagnostic_psd.png"))
    plt.close()
    print("   -> Generated 'diagnostic_psd.png' (Check for 50Hz peak)")

    # Report text
    bad_chans = []
    for i, val in enumerate(rms_per_ch * 1e6):
        status = "OK"
        if val < 1.0: status = "DEAD/FLAT"
        if val > 100.0: status = "NOISY"
        print(f"   - {ch_names[i]:<5}: {val:.2f} uV [{status}]")
        if status != "OK": bad_chans.append(ch_names[i])
        
    return bad_chans

def compute_erds(epochs, tmin=1.0, tmax=3.0):
    """Computes basic ERDS maps (Event-Related Desynchronization)."""
    print("\n🧠 --- 2. BRAIN RESPONSE (ERDS) ---")
    
    freqs = np.arange(4, 30, 1)  # 4-30Hz (Theta, Alpha, Beta)
    n_cycles = freqs / 2.0
    
    # Calculate TFR (Time Frequency Representation)
    power = mne.time_frequency.tfr_morlet(
        epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
        return_itc=False, decim=2, n_jobs=1, verbose=False
    )
    
    # Scan for C3 and C4 indices
    ch_map = {name: i for i, name in enumerate(epochs.info['ch_names'])}
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Helper to plot TFR
    def plot_channel_tfr(ax, ch_name):
        if ch_name not in ch_map:
            ax.text(0.5, 0.5, f"{ch_name} not found", ha='center')
            return
            
        # Plot power for the specific channel
        # We baseline correct using the beginning of the epoch itself if necessary
        # ideally we use a separate baseline period, but here we look for relative change
        power.plot([ch_map[ch_name]], baseline=(0.0, 0.5), mode='percent', axes=ax, show=False, colorbar=True)
        ax.set_title(f"Spectrogram: {ch_name}")
    
    plot_channel_tfr(axes[0], 'C3')
    plot_channel_tfr(axes[1], 'C4')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "diagnostic_erds.png"))
    plt.close()
    print("   -> Generated 'diagnostic_erds.png'")
    print("      Look for BLUE spots (ERD) in 8-13Hz range during 1s-3s.")

def check_separability(X, y):
    """Riemannian Geometric Classifier Test."""
    print("\n⚖️ --- 3. CLASS SEPARABILITY TEST ---")
    
    # Filter 8-30Hz (Broad Motor Band)
    X_filt = mne.filter.filter_data(X, SFREQ, 8, 30, verbose=False)
    
    # Crop to active window [1s - 3.5s]
    t_start, t_end = 1.0, 3.5
    idx_s = int(t_start * SFREQ)
    idx_e = int(t_end * SFREQ)
    X_crop = X_filt[:, :, idx_s:idx_e]
    
    print(f"   Data shape used for ML: {X_crop.shape}")
    
    # Covariance
    cov = Covariances(estimator='lwf').fit_transform(X_crop)
    
    # MDM Classifier (simplest, robust)
    mdm = MDM()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scores = cross_val_score(mdm, cov, y, cv=cv, n_jobs=1)
    
    mean_acc = np.mean(scores)
    std_acc = np.std(scores)
    n_classes = len(np.unique(y))
    chance = 1.0 / n_classes
    
    print(f"   Classes found: {np.unique(y)}")
    print(f"   Chance Level: {chance:.2%}")
    print(f"   👉 5-Fold Cross-Val Accuracy: {mean_acc:.2%} (+/- {std_acc:.2%})")
    
    if mean_acc < chance + 0.10:
        print("   ⚠️ WARNING: Accuracy is close to chance. The model cannot distinguish classes.")
    else:
        print("   ✅ PASS: Significant separability detected.")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load
    X, y = load_processed_data()
    if X is None: return
    
    # 2. Quality
    bad_chans = check_signal_quality(X, CH_NAMES, SFREQ)
    
    # 3. ERDS (Create Epochs object for MNE tools)
    info = mne.create_info(CH_NAMES, SFREQ, 'eeg')
    info.set_montage('standard_1020')
    events = np.column_stack((np.arange(len(y)), np.zeros(len(y), dtype=int), y.astype(int)))
    epochs = mne.EpochsArray(X, info, events=events, tmin=0.0, verbose=False)
    
    compute_erds(epochs)
    
    # 4. Separability
    check_separability(X, y)

    if bad_chans:
        print(f"\n🚨 DIAGNOSTIC RESULT: FAIL. Found bad channels: {bad_chans}")
    else:
        print("\n✅ DIAGNOSTIC RESULT: Signal physical quality looks OK. Check ERDS maps for biological validity.")

if __name__ == "__main__":
    main()

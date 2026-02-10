import numpy as np
import mne
import matplotlib.pyplot as plt
import os
import sys
from scipy.signal import hilbert
from scipy.stats import pearsonr

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.utils import eeg_loader

# --- CONFIGURATION ---
CH_NAMES = eeg_loader.CH_NAMES_DOC
SFREQ = eeg_loader.SFREQ
CLASS_NAMES = {1: 'Hand (L)', 2: 'Hand (R)', 3: 'Feet', 10: 'Rest'}
BANDS = {
    '7-13 Hz (Alpha)': (7, 13),
    '13-31 Hz (Beta)': (13, 31),
    '71-91 Hz (Gamma)': (71, 91)
}
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")

def compute_envelope(data, sfreq, l_freq, h_freq):
    data_filt = mne.filter.filter_data(data, sfreq, l_freq, h_freq, verbose=False)
    envelope = np.abs(hilbert(data_filt, axis=-1)) ** 2
    return envelope

def compute_correlations(envelope_mean, y, target_class):
    target_vector = (y == target_class).astype(float)
    if np.all(target_vector == 0) or np.all(target_vector == 1):
        return np.zeros(envelope_mean.shape[1])

    n_channels = envelope_mean.shape[1]
    corrs = np.zeros(n_channels)
    for ch in range(n_channels):
        if np.std(envelope_mean[:, ch]) == 0:
            corrs[ch] = 0
        else:
            r, _ = pearsonr(envelope_mean[:, ch], target_vector)
            corrs[ch] = r
    return corrs

def plot_correlation_maps(correlations, info, bands, classes):
    n_bands = len(bands)
    n_classes = len(classes)
    fig, axes = plt.subplots(n_bands, n_classes, figsize=(3 * n_classes, 2.5 * n_bands))
    band_names = list(bands.keys())
    # Sort bands logically
    sorted_band_names = [b for b in ['7-13 Hz (Alpha)', '13-31 Hz (Beta)', '71-91 Hz (Gamma)'] if b in band_names]
    sorted_classes = sorted(list(classes.keys()))
    
    if n_bands == 1: axes = axes.reshape(1, -1)
    if n_classes == 1: axes = axes.reshape(-1, 1)

    for row, band in enumerate(sorted_band_names):
        vals = []
        for cls in sorted_classes:
            vals.append(correlations[band][cls])
        vals = np.concatenate(vals)
        vmax = np.percentile(np.abs(vals), 100)
        vmin = -vmax
        print(f"Band {band}: Vmax (for plot) = {vmax:.3f}")

        for col, cls in enumerate(sorted_classes):
            ax = axes[row, col]
            corr = correlations[band][cls]
            im, _ = mne.viz.plot_topomap(
                corr, info, axes=ax, show=False, vlim=(vmin, vmax),
                cmap='coolwarm', contours=0, sphere=0.12
            )
            if row == 0:
                ax.set_title(classes[cls], fontsize=12, fontweight='bold')
            if col == 0:
                ax.set_ylabel(band, fontsize=12, fontweight='bold')
                ax.text(-0.3, 0.5, band, transform=ax.transAxes, 
                        rotation=90, va='center', ha='right', fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(left=0.1)
    return fig

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize envelope correlation.')
    parser.add_argument('input_file', help='Path to CSV file')
    args = parser.parse_args()

    print(f"Starting Envelope-Class Correlation Analysis on {args.input_file}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # LOAD DATA via Loader
    X, y, info_raw = eeg_loader.load_and_epoch_csv(args.input_file)
    if len(X) == 0:
        return

    # Apply Common Average Reference (CAR)
    print("Applying Common Average Reference (CAR)...")
    X = X - np.mean(X, axis=1, keepdims=True)

    unique_y = np.unique(y)
    available_classes = {k: v for k, v in CLASS_NAMES.items() if k in unique_y}
    print(f"Analyzing classes: {available_classes}")
    
    # Create MNE Info with Montage
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(CH_NAMES, SFREQ, ch_types='eeg')
    info.set_montage(montage)
    
    all_correlations = {}
    for band_name, (l_freq, h_freq) in BANDS.items():
        print(f"\n--- Processing {band_name} ---")
        env = compute_envelope(X, SFREQ, l_freq, h_freq)
        env_mean = np.mean(env, axis=2)
        all_correlations[band_name] = {}
        for cls_id, cls_name in available_classes.items():
            r_vals = compute_correlations(env_mean, y, cls_id)
            all_correlations[band_name][cls_id] = r_vals
            
    print("\nGenerating Plot...")
    fig = plot_correlation_maps(all_correlations, info, BANDS, available_classes)
    
    base_name = os.path.basename(args.input_file).replace(".csv", "")
    save_path = os.path.join(OUTPUT_DIR, f"envelope_correlation_{base_name}.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {save_path}")

if __name__ == "__main__":
    main()

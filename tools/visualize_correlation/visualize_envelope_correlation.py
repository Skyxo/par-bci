
import numpy as np
import mne
import matplotlib.pyplot as plt
import os
import glob
from scipy.signal import hilbert
from scipy.stats import pearsonr

# --- CONFIGURATION ---
# Channel mapping from visualize_epochs.py
CH_NAMES = ['FC3', 'FC4', 'CP3', 'Cz', 'C3', 'C4', 'Pz', 'CP4']
SFREQ = 250
# Expected classes: 1: Left, 2: Right, 3: Feet, 10: Rest (if available)
CLASS_NAMES = {1: 'Hand (L)', 2: 'Hand (R)', 3: 'Feet', 10: 'Rest'}
# Defined Frequency Bands
BANDS = {
    '7-13 Hz (Alpha)': (7, 13),
    '13-31 Hz (Beta)': (13, 31),
    '71-91 Hz (Gamma)': (71, 91)
}

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")

def load_data(x_path):
    """Loads X and y from the specified X path."""
    print(f"Loading data from: {x_path}")
    
    if not os.path.exists(x_path):
        print(f"Error: File {x_path} not found.")
        return None, None
        
    y_path = x_path.replace("X_", "y_")
    if not os.path.exists(y_path):
        print(f"Error: Corresponding labels file {y_path} not found.")
        return None, None
        
    X = np.load(x_path)
    y = np.load(y_path)
    
    # CROP DATA: From 0.5s to 3.5s (Action phase is 4s, cut start/end transients)
    # X shape: (n_trials, n_channels, n_times)
    # SFREQ = 250
    t_start = int(0.5 * SFREQ)
    t_end = int(3.5 * SFREQ)
    
    if X.shape[2] > t_end:
        print(f"Cropping data from {t_start} to {t_end} samples ({0.5}-{3.5}s)")
        X = X[:, :, t_start:t_end]
    else:
        print("Warning: Data epoch too short to crop as requested.")
        
    return X, y

def compute_envelope(data, sfreq, l_freq, h_freq):
    """
    Filters data and computes SQUARED envelope (Power) using Hilbert transform.
    data: (n_trials, n_channels, n_times)
    Returns: envelope (n_trials, n_channels, n_times)
    """
    print(f"Filtering {l_freq}-{h_freq} Hz...")
    # Filter
    data_filt = mne.filter.filter_data(data, sfreq, l_freq, h_freq, verbose=False)
    # Hilbert -> Envelope -> Square (Power)
    envelope = np.abs(hilbert(data_filt, axis=-1)) ** 2
    return envelope

def compute_correlations(envelope_mean, y, target_class):
    """
    Computes correlation between envelope mean and binary class vector.
    envelope_mean: (n_trials, n_channels) - Average envelope power per trial
    y: (n_trials,) - Labels
    target_class: int - Class to correlate against
    
    Returns: correlations (n_channels,)
    """
    target_vector = (y == target_class).astype(float)
    
    # Check if we have variance
    if np.all(target_vector == 0) or np.all(target_vector == 1):
        print(f"Warning: Class {target_class} has 0 or all samples. Returning zeros.")
        return np.zeros(envelope_mean.shape[1])

    n_channels = envelope_mean.shape[1]
    corrs = np.zeros(n_channels)
    
    for ch in range(n_channels):
        # Pearson correlation
        if np.std(envelope_mean[:, ch]) == 0:
            corrs[ch] = 0
        else:
            # use point biserial which is equivalent to pearson on binary
            r, _ = pearsonr(envelope_mean[:, ch], target_vector)
            corrs[ch] = r
            
    return corrs

def plot_correlation_maps(correlations, info, bands, classes):
    """
    correlations: Dict[band_name][class_id] -> (n_channels,)
    """
    n_bands = len(bands)
    n_classes = len(classes)
    
    fig, axes = plt.subplots(n_bands, n_classes, figsize=(3 * n_classes, 2.5 * n_bands))
    
    band_names = list(bands.keys())
    sorted_band_names = [
        '7-13 Hz (Alpha)', 
        '13-31 Hz (Beta)', 
        '71-91 Hz (Gamma)'
    ]
    
    sorted_classes = sorted(list(classes.keys())) # Usually 1, 2, 3, 10
    
    if n_bands == 1:
        axes = axes.reshape(1, -1)
    if n_classes == 1:
        axes = axes.reshape(-1, 1)

    for row, band in enumerate(sorted_band_names):
        # Determine vmin/vmax for this band across all classes
        vals = []
        for cls in sorted_classes:
            vals.append(correlations[band][cls])
        vals = np.concatenate(vals)
        
        # Robust scaling: Use 95th percentile to avoid outliers from Rest if any
        vmax = np.percentile(np.abs(vals), 100) # Use max for now
        vmin = -vmax
        
        print(f"Band {band}: Vmax (for plot) = {vmax:.3f}")

        for col, cls in enumerate(sorted_classes):
            ax = axes[row, col]
            corr = correlations[band][cls]
            
            # Plot topomap
            im, _ = mne.viz.plot_topomap(
                corr, 
                info, 
                axes=ax, 
                show=False, 
                vlim=(vmin, vmax),
                cmap='coolwarm',
                contours=0,
                sphere=0.12 # Fit 8 channels tightly
            )
            
            # Titles
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
    parser.add_argument('input_file', help='Path to X.npy file')
    parser.add_argument('--output_dir', default=OUTPUT_DIR, help='Directory to save results')
    args = parser.parse_args()

    print(f"Starting Envelope-Class Correlation Analysis on {args.input_file}...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data
    X, y = load_data(args.input_file)
    if X is None:
        print("Failed to load data.")
        return

    # Apply Common Average Reference (CAR)
    print("Applying Common Average Reference (CAR)...")
    X = X - np.mean(X, axis=1, keepdims=True)

    print(f"Data Loaded: {X.shape}, Labels: {y.shape}")
    unique_y = np.unique(y)
    print(f"Found classes: {unique_y}")
    
    available_classes = {k: v for k, v in CLASS_NAMES.items() if k in unique_y}
    print(f"Analyzing classes: {available_classes}")
    
    # Create MNE Info
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(CH_NAMES, SFREQ, ch_types='eeg')
    info.set_montage(montage)
    
    all_correlations = {}
    
    # 2. Process per Band
    for band_name, (l_freq, h_freq) in BANDS.items():
        print(f"\n--- Processing {band_name} ---")
        
        env = compute_envelope(X, SFREQ, l_freq, h_freq)
        env_mean = np.mean(env, axis=2) # Average Power over time
        
        all_correlations[band_name] = {}
        
        print(f"Correlations (Min / Max):")
        for cls_id, cls_name in available_classes.items():
            r_vals = compute_correlations(env_mean, y, cls_id)
            all_correlations[band_name][cls_id] = r_vals
            print(f"  {cls_name:10}: Min={np.min(r_vals):.3f}, Max={np.max(r_vals):.3f}")

    # 3. Plot
    print("\nGenerating Plot...")
    fig = plot_correlation_maps(all_correlations, info, BANDS, available_classes)
    
    # Generate filename from input file
    base_name = os.path.basename(args.input_file).replace("X_EEG_Session_", "").replace(".npy", "")
    save_path = os.path.join(args.output_dir, f"envelope_correlation_{base_name}.png")
    
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {save_path}")

if __name__ == "__main__":
    main()

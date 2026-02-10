import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys
from scipy.spatial.distance import euclidean

# Add project root to path to import tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.utils import eeg_loader

# --- CONFIGURATION ---
CH_NAMES = eeg_loader.CH_NAMES_DOC
SFREQ = eeg_loader.SFREQ
CLASS_NAMES = {1: 'Left', 2: 'Right', 3: 'Feet', 10: 'Rest'}
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

def find_last_csv():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data_markiv')
    files = sorted(glob.glob(os.path.join(data_dir, 'EEG_Session_*.csv')))
    if not files:
        raise FileNotFoundError("No CSV files found in data_markiv")
    return files[-1]

def analyze_variance(file_path):
    base_name = os.path.basename(file_path).replace(".csv", "")
    print(f"Analyzing {base_name}...")
    
    # LOAD DATA via Shared Loader
    X, y, _ = eeg_loader.load_and_epoch_csv(file_path)
    
    if len(X) == 0:
        print("No valid epochs found.")
        return

    # Create Output Dir
    session_results_dir = os.path.join(RESULTS_DIR, f"variance_{base_name}")
    os.makedirs(session_results_dir, exist_ok=True)
    
    unique_classes = np.unique(y)
    outliers = []
    
    for cls in unique_classes:
        cls_name = CLASS_NAMES.get(cls, f"Class {cls}")
        print(f"\n--- Processing {cls_name} ---")
        
        # Get trials for this class
        indices = np.where(y == cls)[0]
        trials = X[indices] # (n_trials, n_channels, n_times)
        
        if len(trials) < 2:
            print("Not enough trials for variance.")
            continue
            
        # 1. Compute Mean Trial
        mean_trial = np.mean(trials, axis=0) # (n_channels, n_times)
        std_trial = np.std(trials, axis=0)   # (n_channels, n_times)
        
        # 2. Outlier Detection (Euclidean distance from mean trial)
        distances = []
        for i, trial in enumerate(trials):
            # Flatten to vector for distance
            dist = euclidean(trial.flatten(), mean_trial.flatten())
            distances.append(dist)
            
        distances = np.array(distances)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + 2 * std_dist
        
        class_outliers = indices[distances > threshold]
        for idx, dist in zip(class_outliers, distances[distances > threshold]):
            score = (dist - mean_dist) / std_dist
            outliers.append({
                'trial_idx': idx, 
                'class': cls_name, 
                'score': score
            })
            print(f"  -> Outlier detected: Trial {idx+1} (Score: {score:.2f} sigma)")

        # 3. Visualization (Mean +/- SD)
        target_chs = ['C3', 'C4', 'Cz']
        ch_indices = [CH_NAMES.index(ch) for ch in target_chs if ch in CH_NAMES]
        
        fig, axes = plt.subplots(len(ch_indices), 1, figsize=(10, 8), sharex=True)
        if len(ch_indices) == 1: axes = [axes]
        
        times = np.arange(trials.shape[2]) / SFREQ
        
        for ax, ch_idx, ch_name in zip(axes, ch_indices, target_chs):
            # Plot ALL Trials (Gray spaghetti)
            # Use higher alpha to see distinct lines
            for i in range(trials.shape[0]):
                ax.plot(times, trials[i, ch_idx], color='k', linewidth=0.8, alpha=0.4)

            # Plot Mean (Blue for contrast)
            ax.plot(times, mean_trial[ch_idx], color='blue', linewidth=2, label='Mean')
            
            # Plot +/- 1 Std Dev (Shaded Blue)
            ax.fill_between(times, 
                            mean_trial[ch_idx] - std_trial[ch_idx], 
                            mean_trial[ch_idx] + std_trial[ch_idx], 
                            color='blue', alpha=0.1, label='±1 SD')
            
            # Plot Outliers (Red, thin)
            for out_idx in class_outliers:
                # Find local index
                loc_idx = np.where(indices == out_idx)[0][0]
                ax.plot(times, trials[loc_idx, ch_idx], color='red', linewidth=1.5, alpha=0.8, linestyle='--')
            
            ax.set_title(f"Channel {ch_name} - {cls_name}")
            ax.set_ylabel("Amplitude (uV)")
            if ch_name == target_chs[0]:
                ax.legend(loc='upper right')
                
        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig(os.path.join(session_results_dir, f"variance_plot_{cls_name}.png"))
        plt.close(fig)

    # Save Outlier Report
    report_path = os.path.join(session_results_dir, "outliers_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Outlier Report for {base_name}\n")
        f.write("================================\n")
        if outliers:
            for o in outliers:
                f.write(f"Trial {o['trial_idx']+1:03d} [{o['class']}]: +{o['score']:.2f} sigma from mean\n")
        else:
            f.write("No significant outliers detected (> 2 sigma).\n")
            
    print(f"\nAnalysis complete. Results in {session_results_dir}")
    if outliers:
        print(f"Found {len(outliers)} outliers. See report.")

if __name__ == "__main__":
    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('input_file', nargs='?', help='Path to CSV file')
        args = parser.parse_args()
        
        if args.input_file:
            path = args.input_file
        else:
            path = find_last_csv()
            
        analyze_variance(path)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

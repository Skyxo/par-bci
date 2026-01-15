"""
GRID SEARCH BCI OPTIMIZER 🕵️‍♂️
=============================
Systematically tests multiple Frequency Bands and Time Windows 
to find the "Sweet Spot" for Motor Imagery classification.
"""

import pandas as pd
import numpy as np
import mne
import glob
import os
import joblib
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CH_NAMES = ['FC3', 'FC4', 'CP3', 'Cz', 'C3', 'C4', 'Pz', 'CP4']
SFREQ = 250

# --- SEARCH GRID ---
FREQ_BANDS = {
    "Mu (Anchor)":  (8, 13),
    "Low Beta":     (13, 20),
    "High Beta":    (20, 30),
    "Broad Motor":  (8, 30),
    "Extended":     (7, 35)
}

TIME_WINDOWS = {
    "Early":    (0.0, 2.0),
    "Middle":   (0.5, 2.5),
    "Late":     (1.0, 3.0),
    "Short":    (0.5, 1.5),
}

def load_data(fmin, fmax, tmin, tmax):
    files = glob.glob(os.path.join(BASE_DIR, "EEG_Session_*.csv"))
    X_list = []
    y_list = []
    
    for fname in files:
        try:
            try:
                df = pd.read_csv(fname, sep='\t', header=None)
            except:
                df = pd.read_csv(fname, sep=',', header=None)
            if df.shape[1] < 24: continue

            # Scaling
            raw_values = df.iloc[:, 1:9].values.T
            if np.mean(np.abs(raw_values)) > 1.0: eeg = raw_values * 1e-6
            else: eeg = raw_values
            
            markers = df.iloc[:, 23].values
            
            info = mne.create_info(CH_NAMES, SFREQ, 'eeg')
            raw = mne.io.RawArray(eeg, info, verbose=False)
            
            # FILTERING (Critical Step)
            raw.notch_filter(50, verbose=False)
            raw.filter(fmin, fmax, fir_design='firwin', verbose=False)
            
            # EVENTS (Pulse Handling)
            diff_markers = np.diff(markers, prepend=0)
            idx = np.where(np.isin(markers, [1, 2]) & (diff_markers != 0))[0] # Left(1) vs Right(2) ONLY
            vals = markers[idx].astype(int)
            
            if len(idx) > 0:
                events = np.column_stack((idx, np.zeros_like(idx), vals))
                # EPOCHING
                epochs = mne.Epochs(raw, events, {'Left':1, 'Right':2}, 
                                    tmin=tmin, tmax=tmax, 
                                    baseline=None, reject=None, 
                                    verbose=False, on_missing='ignore')
                
                data = epochs.get_data()
                if len(data) > 0:
                    X_list.append(data)
                    y_list.append(epochs.events[:, -1])
                    
        except Exception:
            pass
            
    if not X_list: return None, None
    return np.concatenate(X_list), np.concatenate(y_list)

def main():
    print(f"=== 🕵️‍♂️ BCI GRID SEARCH: {len(FREQ_BANDS)*len(TIME_WINDOWS)} COMBINATIONS ===")
    print("Optimization Target: Binary Classification (Left vs Right)\n")
    
    results = []
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = make_pipeline(Covariances(estimator='lwf'), TangentSpace(), LogisticRegression())
    
    print(f"{'BAND':<15} | {'WINDOW':<10} | {'ACCURACY':<10}")
    print("-" * 45)
    
    best_acc = 0
    best_cfg = ""
    
    for f_name, (fmin, fmax) in FREQ_BANDS.items():
        for t_name, (tmin, tmax) in TIME_WINDOWS.items():
            
            X, y = load_data(fmin, fmax, tmin, tmax)
            
            if X is None or len(X) < 10:
                print(f"{f_name:<15} | {t_name:<10} | N/A (No Data)")
                continue
                
            scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
            mean_acc = np.mean(scores) * 100
            
            print(f"{f_name:<15} | {t_name:<10} | {mean_acc:.2f}%")
            results.append((mean_acc, f_name, t_name))
            
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_cfg = f"{f_name} ({fmin}-{fmax}Hz) + {t_name} ({tmin}-{tmax}s)"
                
    print("\n" + "="*45)
    print(f"🏆 BEST CONFIGURATION: {best_cfg}")
    print(f"🎯 ACCURACY: {best_acc:.2f}%")
    print("="*45)

if __name__ == "__main__":
    main()

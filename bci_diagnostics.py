
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.distance import distance_riemann
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.manifold import TSNE
import seaborn as sns
from scipy.signal import welch

# Configuration
TRAIN_FILE = "EEG_Session_2026-01-13_16-59.csv"
TEST_FILE = "EEG_Session_2026-01-13_15-30.csv"
SFREQ = 250
CH_NAMES = ['C3', 'C4', 'Cz', 'P3', 'P4', 'Pz', 'F3', 'F4']

def load_data(filename):
    print(f"Loading {filename}...")
    try:
        df = pd.read_csv(filename, sep='\t', header=None)
        eeg = df.iloc[:, 1:9].values.T * 1e-6  # uV to V
        markers = df.iloc[:, 23].values
        
        info = mne.create_info(ch_names=CH_NAMES, sfreq=SFREQ, ch_types='eeg')
        raw = mne.io.RawArray(eeg, info, verbose=False)
        return raw, markers
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None, None

def check_signal_quality(raw, name="Dataset"):
    print(f"\n--- Signal Quality Analysis: {name} ---")
    data = raw.get_data() * 1e6 # Convert to uV for analysis
    
    # 1. Saturation (Railing)
    # Check if values are close to typical hardware limits (e.g. +/- 187500 uV for some OpenBCI) or just very large
    saturation_threshold = 1000 # uV, conservative threshold for "bad" artifact
    pct_saturated = np.mean(np.abs(data) > saturation_threshold) * 100
    print(f"Saturation (> {saturation_threshold} uV): {pct_saturated:.4f}%")
    
    # 2. 50Hz Noise (Mains hum)
    n_per_seg = int(SFREQ * 2)
    freqs, psd = welch(data, fs=SFREQ, nperseg=n_per_seg)
    
    # Find index for 50Hz (+/- 2Hz) and neighbor frequencies
    idx_50 = np.where((freqs >= 48) & (freqs <= 52))[0]
    idx_base = np.where(((freqs >= 40) & (freqs <= 45)) | ((freqs >= 55) & (freqs <= 60)))[0]
    
    power_50 = np.mean(psd[:, idx_50], axis=1)
    power_base = np.mean(psd[:, idx_base], axis=1)
    ratio_50 = power_50 / (power_base + 1e-9)
    
    print("50Hz Mains Noise Ratio (target < 10, warning > 100):")
    for i, ch in enumerate(CH_NAMES):
        status = "✅" if ratio_50[i] < 10 else "⚠️" if ratio_50[i] < 100 else "❌"
        print(f"  {ch}: {ratio_50[i]:.1f} {status}")
        
    # 3. Overall Noise Level (RMS)
    rms = np.sqrt(np.mean(data**2, axis=1))
    print("Channel RMS Amplitude (uV) - High > 50uV, Low < 1uV:")
    for i, ch in enumerate(CH_NAMES):
        status = "✅" if 1 < rms[i] < 50 else "❌"
        print(f"  {ch}: {rms[i]:.1f} {status}")

    return data

def analyze_epochs(raw, markers, name="Dataset"):
    print(f"\n--- Epochs Analysis: {name} ---")
    
    # Create events
    diff_markers = np.diff(markers, prepend=0)
    events_idx = np.where(np.isin(markers, [1, 2, 3, 10]) & (diff_markers != 0))[0]
    events_vals = markers[events_idx].astype(int)
    events = np.column_stack((events_idx, np.zeros_like(events_idx), events_vals))
    
    event_id = {'GAUCHE': 1, 'DROITE': 2, 'PIEDS': 3, 'REPOS': 10}
    
    # Filter for epoching
    raw_filt = raw.copy().filter(8., 30., fir_design='firwin', verbose=False)
    
    epochs = mne.Epochs(raw_filt, events, event_id, tmin=0.5, tmax=3.5, 
                        proj=False, baseline=None, verbose=False, preload=True)
    
    print(f"Total Epochs: {len(epochs)}")
    counts = epochs.events[:, -1]
    for label, code in event_id.items():
        count = np.sum(counts == code)
        print(f"  {label}: {count}")
        
    return epochs

def analyze_separability(epochs_train, epochs_test):
    print("\n--- Class Separability Analysis (Riemannian Geometry) ---")
    
    cov_est = Covariances(estimator='lwf')
    cov_train = cov_est.fit_transform(epochs_train.get_data())
    
    ts = TangentSpace()
    X_train_ts = ts.fit_transform(cov_train)
    y_train = epochs_train.events[:, -1]
    
    # t-SNE Visualization
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X_train_ts)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y_train, palette='viridis', legend='full')
    plt.title('t-SNE of Training Data (Tangent Space)')
    plt.savefig('bci_tsne_train.png')
    print("Saved t-SNE plot to 'bci_tsne_train.png'")
    
    # Model Performance
    print("\n--- Model Evaluation ---")
    clf = make_pipeline(Covariances(estimator='lwf'), TangentSpace(), LogisticRegression(solver='lbfgs', max_iter=1000))
    
    # CV on Train
    scores = cross_val_score(clf, epochs_train.get_data(), y_train, cv=5)
    print(f"Cross-Validation Accuracy (Train): {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    # Train and Test
    clf.fit(epochs_train.get_data(), y_train)
    
    if epochs_test:
        y_test = epochs_test.events[:, -1]
        y_pred = clf.predict(epochs_test.get_data())
        
        test_acc = accuracy_score(y_test, y_pred)
        print(f"Test Set Accuracy: {test_acc:.4f}")
        
        print("\nClassification Report (Test):")
        print(classification_report(y_test, y_pred, target_names=['GAUCHE (1)', 'DROITE (2)', 'PIEDS (3)', 'REPOS (10)']))
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['G', 'D', 'P', 'R'], 
                    yticklabels=['G', 'D', 'P', 'R'])
        plt.title('Confusion Matrix (Test Set)')
        plt.xlabel('Predicted')
        plt.ylabel('Ground Truth')
        plt.savefig('bci_confusion_matrix.png')
        print("Saved confusion matrix to 'bci_confusion_matrix.png'")
        
        # Check Distribution Shift (Covariate Shift) simply by comparing mean covariance distances?
        # A simple check: compare accuracy trends or visual inspection of t-SNE if we projected test data too.
        
        # Project Test data for t-SNE overlay
        mask_good_events = np.isin(epochs_test.events[:, -1], [1, 2, 3, 10])
        if np.sum(mask_good_events) > 0:
             cov_test = cov_est.fit_transform(epochs_test.get_data())
             X_test_ts = ts.transform(cov_test)
             
             # Combine for drift visualization
             X_combined = np.vstack((X_train_ts, X_test_ts))
             y_combined = np.concatenate((y_train, y_test))
             domain_labels = np.concatenate((['Train']*len(y_train), ['Test']*len(y_test)))
             
             tsne_drift = TSNE(n_components=2, random_state=42)
             X_drift_emb = tsne_drift.fit_transform(X_combined)
             
             plt.figure(figsize=(12, 5))
             
             plt.subplot(1, 2, 1)
             sns.scatterplot(x=X_drift_emb[:, 0], y=X_drift_emb[:, 1], hue=y_combined, palette='viridis', style=domain_labels)
             plt.title('t-SNE: Class Distribution')
             
             plt.subplot(1, 2, 2)
             sns.scatterplot(x=X_drift_emb[:, 0], y=X_drift_emb[:, 1], hue=domain_labels, palette='rocket')
             plt.title('t-SNE: Domain Drift (Train vs Test)')
             
             plt.savefig('bci_drift_analysis.png')
             print("Saved drift analysis to 'bci_drift_analysis.png'")

def main():
    print("=== BCI SYSTEM DIAGNOSTICS ===\n")
    
    # 1. Load Data
    raw_train, markers_train = load_data(TRAIN_FILE)
    if raw_train is None: return
    
    raw_test, markers_test = load_data(TEST_FILE)
    if raw_test is None: return

    # 2. Check Signal Quality
    check_signal_quality(raw_train, "TRAIN SET")
    check_signal_quality(raw_test, "TEST SET")
    
    # 3. Analyze Epochs
    epochs_train = analyze_epochs(raw_train, markers_train, "TRAIN SET")
    epochs_test = analyze_epochs(raw_test, markers_test, "TEST SET")
    
    # 4. Separability & Model
    analyze_separability(epochs_train, epochs_test)
    
    print("\n=== DIAGNOSTICS COMPLETE ===")

if __name__ == "__main__":
    main()

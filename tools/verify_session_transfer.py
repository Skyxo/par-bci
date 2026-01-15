import pandas as pd
import numpy as np
import mne
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as plt
import matplotlib.pyplot as plt

# CONFIG
import os

# CONFIG
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_FILE = os.path.join(BASE_DIR, "EEG_Session_2026-01-13_16-59.csv")
TEST_FILE = os.path.join(BASE_DIR, "EEG_Session_2026-01-13_15-30.csv")
SFREQ = 250
# Important: Use the corrected channel order
CH_NAMES = ['Cz', 'FCz', 'P3', 'Pz', 'C3', 'C4', 'O1', 'P4']

def load_and_epoch(filename, name):
    print(f"\n[{name}] Loading {filename}...")
    try:
        df = pd.read_csv(filename, sep='\t', header=None)
    except:
        df = pd.read_csv(filename, sep=',', header=None)
        
    eeg = df.iloc[:, 1:9].values.T * 1e-6
    markers = df.iloc[:, 23].values
    
    info = mne.create_info(ch_names=CH_NAMES, sfreq=SFREQ, ch_types='eeg')
    raw = mne.io.RawArray(eeg, info, verbose=False)
    
    # Apply filters (Same pipeline as corrected viewer)
    raw.notch_filter([50, 100], fir_design='firwin', verbose=False)
    raw.filter(8., 30., fir_design='firwin', verbose=False)
    
    # Events
    diff_markers = np.diff(markers, prepend=0)
    events_idx = np.where(np.isin(markers, [1, 2, 3, 10]) & (diff_markers != 0))[0]
    events_vals = markers[events_idx].astype(int)
    events = np.column_stack((events_idx, np.zeros_like(events_idx), events_vals))
    
    # Epoching
    event_id = {'GAUCHE': 1, 'DROITE': 2, 'PIEDS': 3, 'REPOS': 10}
    epochs = mne.Epochs(raw, events, event_id, tmin=0.5, tmax=3.5, 
                        proj=False, baseline=None, verbose=False)
    
    return epochs

def main():
    print("=== TRANSFER LEARNING VERIFICATION ===")
    
    # 1. Load Data
    epochs_train = load_and_epoch(TRAIN_FILE, "TRAIN SESSION")
    epochs_test_cross = load_and_epoch(TEST_FILE, "TEST SESSION (CROSS)")
    
    # 2. Train Model
    print("\nTraining Model on TRAIN SESSION...")
    clf = make_pipeline(Covariances(estimator='lwf'), TangentSpace(), LogisticRegression(solver='lbfgs'))
    X_train = epochs_train.get_data()
    y_train = epochs_train.events[:, -1]
    clf.fit(X_train, y_train)
    
    # 3. Test on SAME Session (Train vs Train subset ideally, but here full for sanity check)
    print("\n--- RESULTS: SAME SESSION (Train on 16-59 -> Test on 16-59) ---")
    y_pred_same = clf.predict(X_train)
    acc_same = accuracy_score(y_train, y_pred_same)
    print(f"Accuracy: {acc_same:.2%}")
    print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred_same))
    
    # 4. Test on CROSS Session
    print("\n--- RESULTS: CROSS SESSION (Train on 16-59 -> Test on 15-30) ---")
    X_test = epochs_test_cross.get_data()
    y_test = epochs_test_cross.events[:, -1]
    y_pred_cross = clf.predict(X_test)
    acc_cross = accuracy_score(y_test, y_pred_cross)
    
    print(f"Accuracy: {acc_cross:.2%}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_cross))
    print("\nClassification Report (Cross Session):")
    print(classification_report(y_test, y_pred_cross, target_names=['Left', 'Right', 'Feet', 'Rest']))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    cm_same = confusion_matrix(y_train, y_pred_same, normalize='true')
    axes[0].imshow(cm_same, cmap='Blues')
    axes[0].set_title(f'Same Session\nAcc: {acc_same:.0%}')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    cm_cross = confusion_matrix(y_test, y_pred_cross, normalize='true')
    axes[1].imshow(cm_cross, cmap='Reds')
    axes[1].set_title(f'Cross Session\nAcc: {acc_cross:.0%}')
    axes[1].set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig('session_transfer_check.png')
    print("\nSaved comparison plot to 'session_transfer_check.png'")

if __name__ == "__main__":
    main()

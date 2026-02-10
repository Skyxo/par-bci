"""
ULTRA RIEMANNIAN TRAINING SCRIPT 🧠🚀
=====================================
State-of-the-art BCI classification for "Small Data" scenarios.

Methodology:
1. XdawnCovariances: 
   - Supervised spatial filtering to enhance the "Signal-to-Noise" ratio of ERPs.
   - Far superior to standard Covariances for event-locked signals.
   
2. TangentSpace:
   - Projects the curved Riemannian manifold onto a flat Euclidean space.
   - Allows using standard, robust classifiers on complex geometric data.

3. LogisticRegression (with L2 penalty):
   - The most robust linear classifier for Tangent Space features.
   - Provides probability outputs (confidence scores).

Pipeline:
[Raw EEG] -> [Bandpass 8-30Hz] -> [XdawnCovariances] -> [TangentSpace] -> [LogisticRegression]
"""

import pandas as pd
import numpy as np
import mne
import glob
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pyriemann.estimation import XdawnCovariances, Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
# IMPORTANT: Correct Channel Order for your setup
CH_NAMES = ['FC3', 'FC4', 'CP3', 'Cz', 'C3', 'C4', 'Pz', 'CP4']
SFREQ = 250

def load_all_sessions(adjust_labels=True):
    print("📥 Loading Preprocessed Data from 'processed_data'...")
    try:
        # Glob ALL processed files
        X_files = glob.glob(os.path.join("processed_data", "X_*.npy"))
        if not X_files:
            print("❌ No .npy files found.")
            return None, None
            
        X_list = []
        y_list = []
        
        for f in X_files:
            # Find matching Y
            base = os.path.basename(f).replace("X_", "y_")
            y_f = os.path.join("processed_data", base)
            
            if os.path.exists(y_f):
                print(f"   -> Loading {os.path.basename(f)}...")
                X_part = np.load(f)
                y_part = np.load(y_f)
                X_list.append(X_part)
                y_list.append(y_part)
        
        if not X_list: return None, None
        
        X = np.concatenate(X_list)
        y = np.concatenate(y_list)

        # MAPPINGS: 1(L), 2(R), 3(F), 10(Rest)
        # We map to 0, 1, 2, 3
        y_new = np.zeros_like(y)
        y_new[y == 1] = 0 # Left
        y_new[y == 2] = 1 # Right
        y_new[y == 3] = 2 # Feet
        y_new[y == 10] = 3 # Rest
        y = y_new
        
        return X, y
    except Exception as e:
        print(f"❌ Error loading .npy files: {e}")
        return None, None

def evaluate_pipeline(name, pipeline, X, y, cv):
    print(f"\n🧪 Testing Pipeline: {name}")
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    mean_acc = np.mean(scores)
    std_acc = np.std(scores)
    print(f"   Accuracy: \033[1m{mean_acc*100:.2f}%\033[0m (+/- {std_acc*100:.2f}%)")
    return mean_acc

def main():
    print("\n=== ✨ ULTRA RIEMANNIAN TRAINING (Motor Imagery Edition) ✨ ===")
    
    # 1. Load Data (Raw Window 0-4s)
    X, y = load_all_sessions()
    if X is None: return

    # 1.5 Temporal Filtering (Common)
    X = mne.filter.filter_data(X, SFREQ, 8, 13, verbose=False) # Mu Band
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total Epochs: {X.shape[0]}")
    print(f"   Class Distribution: {np.unique(y, return_counts=True)}") # 0,1,2,3

    pipelines = {
        "Standard Riemannian": make_pipeline(Covariances('lwf'), TangentSpace(), LogisticRegression())
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ---------------------------------------------------------
    # PART A: MOTOR IMAGERY BATTLE (Left vs Right vs Feet)
    # Window: 1.0s to 3.0s (Best for ERD)
    # We EXCLUDE Rest (Class 3) because it's only valid 0-1s.
    # ---------------------------------------------------------
    print("\n--- ⚔️  3-CLASS BATTLE (Left vs Right vs Feet) [1.0s - 3.0s] ⚔️ ---")
    
    idx_star = int(1.0 * SFREQ)
    idx_stop = int(3.0 * SFREQ)
    if idx_stop > X.shape[2]: idx_stop = X.shape[2]
    
    mask_active = y != 3 # Exclude Rest
    X_motor = X[mask_active][:, :, idx_star:idx_stop]
    y_motor = y[mask_active]
    
    best_pipe = pipelines["Standard Riemannian"]
    evaluate_pipeline("Standard Riemannian (3-Class)", best_pipe, X_motor, y_motor, cv)

    # ---------------------------------------------------------
    # PART B: LEFT VS RIGHT (Focus)
    # Window: 1.0s to 3.0s
    # ---------------------------------------------------------
    print("\n--- 🔍 BINARY DIAGNOSTIC (Left vs Right) [1.0s - 3.0s] ---")
    mask_lr = (y_motor == 0) | (y_motor == 1)
    evaluate_pipeline("Standard Riemannian (L vs R)", best_pipe, X_motor[mask_lr], y_motor[mask_lr], cv)

    # ---------------------------------------------------------
    # PART C: REST VS ACTIVE
    # Window: 0.0s to 1.0s (The only valid common window)
    # ---------------------------------------------------------
    print("\n--- 💤 REST VS ACTIVE [0.0s - 1.0s] ---")
    idx_rest_stop = int(1.0 * SFREQ)
    X_rest = X[:, :, :idx_rest_stop]
    
    # Create Binary Labels: 0=Active(L/R/F), 1=Rest
    y_rest_binary = np.zeros_like(y)
    y_rest_binary[y == 3] = 1 
    
    evaluate_pipeline("Rest vs Active", best_pipe, X_rest, y_rest_binary, cv)
    
    # SAVE REST MODEL
    print(f"\n💾 Saving Rest Model...")
    best_pipe.fit(X_rest, y_rest_binary)
    joblib.dump(best_pipe, os.path.join(MODELS_DIR, "riemann_rest.pkl"))
    print(f"✅ Saved to: models/riemann_rest.pkl (Classes: 0=Active, 1=Rest)")

    # ---------------------------------------------------------
    # PART D: EXPERIMENTAL 4-CLASS (Left, Right, Feet, Rest)
    # ... (Skipped for saving, kept for logs if needed but we focus on 3-class)
    # ---------------------------------------------------------
    # print("\n--- 🧪 4-CLASS EXPERIMENT ...") 
    
    # SAVE BEST L/R/F MODEL
    print(f"\n💾 Saving Best Motor Model (3-Class)...")
    best_pipe.fit(X_motor, y_motor)
    joblib.dump(best_pipe, os.path.join(MODELS_DIR, "riemann_motor.pkl"))
    print(f"✅ Saved to: models/riemann_motor.pkl (Classes: 0=Left, 1=Right, 2=Feet)")

if __name__ == "__main__":
    main()

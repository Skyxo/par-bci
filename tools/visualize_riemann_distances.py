import numpy as np
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.distance import distance_riemann
import os
import glob

# --- CONFIGURATION ---
CLASSES = {1: 'Gauche', 2: 'Droite'}
COLORS = {1: 'blue', 2: 'red'}
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed_data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "presentation_figures")
SFREQ = 250

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
    print("📏 Calcul des Distances Riemanniennes par Essai...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    X, y = load_data()
    # Filter only Left(1) / Right(2)
    mask = np.isin(y, [1, 2])
    X = X[mask]
    y = y[mask]
    
    # 2. Filter Mu Band (8-13Hz) & Crop active window
    X_filt = mne.filter.filter_data(X, SFREQ, 8, 13, verbose=False)
    X_crop = X_filt[:, :, int(1.0*SFREQ):int(3.5*SFREQ)]
    
    # 3. Compute Covariances
    cov_est = Covariances(estimator='lwf')
    covs = cov_est.fit_transform(X_crop)
    
    # 4. Compute Means (The "Centroids")
    print("   Calcul des Moyennes de Classe (Centroids)...")
    idx_L = (y == 1)
    idx_R = (y == 2)
    
    mean_L = mean_covariance(covs[idx_L], metric='riemann')
    mean_R = mean_covariance(covs[idx_R], metric='riemann')
    
    # 5. Calculate Distances for EACH trial
    print("   Calcul des distances individuelles...")
    dists_L = [] # Distance to Left Mean
    dists_R = [] # Distance to Right Mean
    true_labels = []
    
    n_correct = 0
    total = len(y)
    
    for i in range(len(covs)):
        C_i = covs[i]
        
        # Riemannian Distance to centroids
        d_L = distance_riemann(C_i, mean_L)
        d_R = distance_riemann(C_i, mean_R)
        
        dists_L.append(d_L)
        dists_R.append(d_R)
        true_labels.append(y[i])
        
        # Classification Rule (MDM): Assign to closest mean
        # If true is Left(1), we want d_L < d_R
        # If true is Right(2), we want d_R < d_L
        pred = 1 if d_L < d_R else 2
        if pred == y[i]:
            n_correct += 1

    acc = n_correct / total
    print(f"   Précision MDM (sur jeu complet): {acc:.1%}")

    # 6. Visualization
    plt.figure(figsize=(10, 8))
    
    dists_L = np.array(dists_L)
    dists_R = np.array(dists_R)
    true_labels = np.array(true_labels)
    
    # Plot Left Trials (Blue)
    plt.scatter(dists_L[true_labels==1], dists_R[true_labels==1], 
                c='blue', label='Vrai: GAUCHE', alpha=0.7, edgecolors='w', s=80)
    
    # Plot Right Trials (Red)
    plt.scatter(dists_L[true_labels==2], dists_R[true_labels==2], 
                c='red', label='Vrai: DROITE', alpha=0.7, edgecolors='w', s=80)
    
    # Decision Boundary (y = x)
    # Since scaling might be similar, we just plot diagonal
    lims = [
        np.min([plt.xlim(), plt.ylim()]),  # min of both axes
        np.max([plt.xlim(), plt.ylim()]),  # max of both axes
    ]
    plt.plot(lims, lims, 'k--', alpha=0.75, label="Frontière de Décision (Incertaite)")
    
    # Uncertainty / Margin zones
    # Fill area where d_L < d_R (Predicted Left)
    plt.fill_between(lims, lims, np.max(lims), color='red', alpha=0.05)
    plt.fill_between(lims, np.min(lims), lims, color='blue', alpha=0.05)
    
    # Labels
    plt.xlabel("Distance au Centre GAUCHE (Riemann)")
    plt.ylabel("Distance au Centre DROITE (Riemann)")
    plt.title(f"Classification Riemannienne - Essai par Essai\nPrécision: {acc:.1%} ({n_correct}/{total})", fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Annotations
    plt.text(lims[0]*1.1, lims[1]*0.9, "Zone Prédite GAUCHE", color='blue', fontweight='bold', ha='left')
    plt.text(lims[1]*0.9, lims[0]*1.1, "Zone Prédite DROITE", color='red', fontweight='bold', ha='right')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "slide_3_riemann_distances.png")
    plt.savefig(save_path)
    print(f"✅ Image générée : {save_path}")

if __name__ == "__main__":
    main()

import numpy as np
import mne
import matplotlib.pyplot as plt
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance
import os
import glob
import sys

# --- CONFIGURATION ---
CH_NAMES = ['FC3', 'FC4', 'CP3', 'Cz', 'C3', 'C4', 'Pz', 'CP4']
SFREQ = 250
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "audit_results")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed_data")

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

def plot_cov_matrix(cov, title, ax, vmin=None, vmax=None):
    im = ax.imshow(cov, cmap='RdBu_r', interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(CH_NAMES)))
    ax.set_yticks(np.arange(len(CH_NAMES)))
    ax.set_xticklabels(CH_NAMES, rotation=45)
    ax.set_yticklabels(CH_NAMES)
    return im

def main():
    print("🧠 Visualisation des Covariances Riemanniennes...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    X, y = load_data()
    if X is None: return

    print(f"   Données brutes: {X.shape}")

    # 1. Filtrage Bande Mu (8-13 Hz) - C'est là que la magie opère
    print("   Filtrage 8-13 Hz (Mu)...")
    X_filt = mne.filter.filter_data(X, SFREQ, 8, 13, verbose=False)
    
    # 2. Crop 1.0s - 3.5s
    t_start, t_end = 1.0, 3.5
    idx_s, idx_e = int(t_start*SFREQ), int(t_end*SFREQ)
    X_crop = X_filt[:, :, idx_s:idx_e]
    
    # 3. Calcul des Covariances (Epoch par Epoch)
    cov_est = Covariances(estimator='lwf') # Lederhit-Wolf est robuste
    covs = cov_est.fit_transform(X_crop)
    print(f"   Matrices de Covariance calculées: {covs.shape}")

    # 4. Moyenne Riemannienne par Classe
    classes = {1: 'Gauche', 2: 'Droite', 3: 'Pieds'}
    mean_covs = {}
    
    for cls, name in classes.items():
        mask = (y == cls)
        if np.sum(mask) == 0: continue
        
        # Moyenne Riemannienne (Geodesic Mean) est mieux, mais LogEuclidean est plus rapide/robuse pour la visu
        # On utilise mean_covariance de pyriemann qui gère ça
        mean_covs[cls] = mean_covariance(covs[mask], metric='riemann')
        
    # 5. Visualisation
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Ligne 1 : Les Moyennes Absolues
    vmin, vmax = None, None # Auto scale
    
    im1 = plot_cov_matrix(mean_covs[1], "Moyenne GAUCHE", axes[0, 0])
    im2 = plot_cov_matrix(mean_covs[2], "Moyenne DROITE", axes[0, 1])
    im3 = plot_cov_matrix(mean_covs[3], "Moyenne PIEDS", axes[0, 2])
    
    # Barre de couleur commune pour la ligne 1 (un peu hacky)
    # fig.colorbar(im3, ax=axes[0, :], shrink=0.6)

    # Ligne 2 : Les Différences (C'est ça qu'on veut voir !)
    # Gauche vs Droite
    diff_GD = mean_covs[1] - mean_covs[2]
    # Droite vs Pieds
    diff_DP = mean_covs[2] - mean_covs[3]
    # Gauche vs Pieds
    diff_GP = mean_covs[1] - mean_covs[3]

    # On centre la colormap sur 0 pour bien voir les oppositions
    limit = max(np.max(np.abs(diff_GD)), np.max(np.abs(diff_DP)), np.max(np.abs(diff_GP)))
    
    plot_cov_matrix(diff_GD, "Diff: GAUCHE - DROITE\n(Bleu=D plus fort, Rouge=G plus fort)", axes[1, 0], vmin=-limit, vmax=limit)
    plot_cov_matrix(diff_DP, "Diff: DROITE - PIEDS", axes[1, 1], vmin=-limit, vmax=limit)
    plot_cov_matrix(diff_GP, "Diff: GAUCHE - PIEDS", axes[1, 2], vmin=-limit, vmax=limit)
    
    plt.suptitle("Empreintes Riemanniennes (Covariance 8-13Hz)\nLigne 2 : Les Différences de Synchronisation", fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, "covariance_matrices.png")
    plt.savefig(save_path)
    print(f"✅ Image générée : {save_path}")

if __name__ == "__main__":
    main()

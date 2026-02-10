import numpy as np
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.tangentspace import tangent_space
from scipy.linalg import fractional_matrix_power, logm
import os
import glob

# --- CONFIGURATION ---
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed_data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "presentation_figures")
CH_NAMES = ['FC3', 'FC4', 'CP3', 'Cz', 'C3', 'C4', 'Pz', 'CP4']
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

def compute_riemann_log_map(cov_target, cov_ref):
    """
    Map cov_target into the Tangent Space of cov_ref.
    Formula: Log( Ref^-1/2 * Target * Ref^-1/2 )
    This represents the 'vector' to go from Ref to Target.
    """
    # 1. Compute Ref^-1/2 (Whitening Matrix)
    # Eigendecomposition is safer numerically
    eigvals, eigvecs = np.linalg.eigh(cov_ref)
    # Inv Sqrt
    eigvals_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    whiten = eigvecs @ eigvals_inv_sqrt @ eigvecs.T
    
    # 2. Whiten the Target
    target_white = whiten @ cov_target @ whiten.T
    
    # 3. Logarithm
    # Since target_white is SPD, we can use matrix log
    eigvals_w, eigvecs_w = np.linalg.eigh(target_white)
    log_eigvals = np.diag(np.log(eigvals_w))
    tangent_vector_mat = eigvecs_w @ log_eigvals @ eigvecs_w.T
    
    return tangent_vector_mat

def main():
    print("🌌 Visualisation de la Différence Riemannienne (Espace Tangent)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    X, y = load_data()
    # Filter Left(1) / Right(2)
    mask = np.isin(y, [1, 2])
    X = X[mask]
    y = y[mask]
    
    # 2. Filter Mu Band (8-13Hz) & Crop
    X_filt = mne.filter.filter_data(X, SFREQ, 8, 13, verbose=False)
    X_crop = X_filt[:, :, int(1.0*SFREQ):int(3.5*SFREQ)]
    
    # 3. Compute Covariances & Means
    cov_est = Covariances(estimator='lwf')
    covs = cov_est.fit_transform(X_crop)
    
    mean_G = mean_covariance(covs[y==1], metric='riemann')
    mean_D = mean_covariance(covs[y==2], metric='riemann')
    
    # 4. Compute Differences
    # A. Euclidean Difference (Soustraction simple)
    diff_euclid = mean_G - mean_D
    
    # B. Riemannian Difference (Tangent Map)
    # "Comment GAUCHE voit DROITE" (Quel mouvement faire pour aller de G à D ?)
    diff_riemann = compute_riemann_log_map(mean_G, mean_D)
    
    # 5. Plot Comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Scale colors symmetrically
    clim_e = np.max(np.abs(diff_euclid))
    clim_r = np.max(np.abs(diff_riemann))
    
    # Plot Euclidean
    sns.heatmap(diff_euclid, ax=axes[0], cmap='RdBu_r', square=True, 
                vmin=-clim_e, vmax=clim_e, xticklabels=CH_NAMES, yticklabels=CH_NAMES)
    axes[0].set_title("Différence Euclidienne (Soustraction)\n(Approximation)", fontsize=14)
    
    # Plot Riemann
    sns.heatmap(diff_riemann, ax=axes[1], cmap='RdBu_r', square=True, 
                vmin=-clim_r, vmax=clim_r, xticklabels=CH_NAMES, yticklabels=CH_NAMES)
    axes[1].set_title("Différence Riemannienne (Log Map)\n(La vraie 'Distance' vue par l'IA)", fontsize=14, fontweight='bold')
    
    plt.suptitle("Comparaison : Qu'est-ce qui change entre GAUCHE et DROITE ?", fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, "slide_4_riemann_diff_theory.png")
    plt.savefig(save_path)
    print(f"✅ Image générée : {save_path}")
    
    print("\n💡 NOTE D'INTERPRÉTATION :")
    print("La matrice de droite montre les modifications relatives des connectivités.")
    print("Les éléments diagonaux (C3, C4) montrent les changements de PUISSANCE relative.")
    print("Les éléments hors-diagonaux montrent les changements de SYNCHRONISATION pure.")

if __name__ == "__main__":
    main()

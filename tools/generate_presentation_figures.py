import numpy as np
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance
import os
import glob
import pandas as pd

# --- CONFIGURATION ---
# On se concentre sur Gauche vs Droite
CLASSES = {1: 'Gauche', 2: 'Droite'}
COLORS = {1: 'blue', 2: 'red'}
CH_NAMES = ['FC3', 'FC4', 'CP3', 'Cz', 'C3', 'C4', 'Pz', 'CP4']
SFREQ = 250
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "presentation_figures")
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

def main():
    print("🎨 Génération des Figures de Présentation (Gauche vs Droite)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    X, y = load_data()
    # Filter only Left/Right
    mask = np.isin(y, [1, 2])
    X = X[mask]
    y = y[mask]
    
    # Filter 8-13 Hz (Mu)
    X_filt = mne.filter.filter_data(X, SFREQ, 8, 13, verbose=False)
    # Crop 1.0 - 3.5s
    X_crop = X_filt[:, :, int(1.0*SFREQ):int(3.5*SFREQ)]
    
    # Create Epochs
    info = mne.create_info(CH_NAMES, SFREQ, 'eeg')
    
    # ---------------------------------------------------------
    # FIGURE 1: L'ECHEC DE LA PUISSANCE (AMPLITUDE)
    # ---------------------------------------------------------
    print("   1. Génération figure Amplitude...")
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot A: Average Power Spectrum (PSD) on C3
    # Manual PSD Calculation to have full control
    from scipy.signal import welch
    
    C3_idx = CH_NAMES.index('C3')
    C4_idx = CH_NAMES.index('C4')
    
    # Calculate Band Power for Scatter Plot
    # Mean square of the filtered signal (since it's already bandpassed 8-13Hz)
    power_c3 = np.var(X_crop[:, C3_idx, :], axis=1)
    power_c4 = np.var(X_crop[:, C4_idx, :], axis=1)
    
    # Convert to log scale for better viz
    log_c3 = 10 * np.log10(power_c3)
    log_c4 = 10 * np.log10(power_c4)
    
    # Scatter Plot
    ax_scatter = axes1[0]
    for c in CLASSES:
        mask_c = (y == c)
        ax_scatter.scatter(log_c3[mask_c], log_c4[mask_c], 
                           c=COLORS[c], label=CLASSES[c], alpha=0.6, edgecolors='w')
    
    ax_scatter.set_xlabel("Puissance C3 (Log)")
    ax_scatter.set_ylabel("Puissance C4 (Log)")
    ax_scatter.set_title("Distribution des Essais (Basé sur la Puissance)\n❌ Impossible de tracer une ligne de séparation")
    ax_scatter.legend()
    ax_scatter.grid(True, alpha=0.3)
    
    # Box Plot for 'Lateralization Score' (C3 - C4)
    # Theoretically: Left hand -> C4 drops (Right active) -> C3 > C4
    #                Right hand -> C3 drops (Left active) -> C4 > C3
    lat_score = log_c4 - log_c3 
    
    data_box = []
    labels_box = []
    colors_box = []
    for c in CLASSES:
        data_box.append(lat_score[y==c])
        labels_box.append(CLASSES[c])
        colors_box.append(COLORS[c])
        
    ax_box = axes1[1]
    bplot = ax_box.boxplot(data_box, patch_artist=True, labels=labels_box)
    
    for patch, color in zip(bplot['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        
    ax_box.set_title("Score de Latéralisation (Différence C4 - C3)\n⚠️ Les distributions se chevauchent énormément")
    ax_box.set_ylabel("Différence de Puissance (dB)")
    ax_box.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle("POURQUOI L'AMPLITUDE NE SUFFIT PAS\n(Les signaux d'imagerie motrice sont trop faibles/bruités)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "slide_1_amplitude_fail.png"))
    plt.close()

    # ---------------------------------------------------------
    # FIGURE 2: LE SUCCES DE LA COVARIANCE (RIEMANN)
    # ---------------------------------------------------------
    print("   2. Génération figure Covariance...")
    
    cov_est = Covariances(estimator='lwf')
    covs = cov_est.fit_transform(X_crop)
    
    mean_G = mean_covariance(covs[y==1], metric='riemann')
    mean_D = mean_covariance(covs[y==2], metric='riemann')
    diff = mean_G - mean_D
    
    fig2 = plt.figure(figsize=(15, 6))
    gs = fig2.add_gridspec(1, 3)
    
    ax_g = fig2.add_subplot(gs[0, 0])
    ax_d = fig2.add_subplot(gs[0, 1])
    ax_diff = fig2.add_subplot(gs[0, 2])
    
    # Helper
    def plot_mat(ax, mat, title, cmap='Blues'):
        sns.heatmap(mat, ax=ax, cmap=cmap, square=True, cbar=False,
                    xticklabels=CH_NAMES, yticklabels=CH_NAMES)
        ax.set_title(title)
        
    plot_mat(ax_g, mean_G, "Covariance Moyenne GAUCHE")
    plot_mat(ax_d, mean_D, "Covariance Moyenne DROITE")
    
    # Difference with diverging colormap
    limit = np.max(np.abs(diff))
    sns.heatmap(diff, ax=ax_diff, cmap='RdBu_r', square=True, cbar=True,
                vmin=-limit, vmax=limit, xticklabels=CH_NAMES, yticklabels=CH_NAMES)
    ax_diff.set_title("DIFFÉRENCE (Le signal utile !)\n(Zones Rouges/Bleues = Changement de Connexion)")
    
    # Annotate significant changes?
    # Simple threshold
    # for i in range(8):
    #     for j in range(8):
    #         if abs(diff[i, j]) > limit * 0.7 and i != j:
    #             ax_diff.add_patch(plt.Circle((j+0.5, i+0.5), 0.4, color='yellow', fill=False, lw=2))

    plt.suptitle("POURQUOI RIEMANN FONCTIONNE\n(On capte la modification de structure du réseau)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "slide_2_covariance_success.png"))
    plt.close()
    
    print("✅ Terminé.")

if __name__ == "__main__":
    main()

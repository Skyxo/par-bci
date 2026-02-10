import numpy as np
import mne
import matplotlib.pyplot as plt
# import seaborn as sns
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.base import logm
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.base import logm
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from sklearn.manifold import MDS
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import os
import glob
import pandas as pd

# --- CONFIGURATION ---
# 4 Classes: Gauche, Droite, Pieds, Repos
CLASSES = {1: 'Gauche', 2: 'Droite', 3: 'Pieds', 10: 'Repos'}
COLORS = {1: 'blue', 2: 'red', 3: 'green', 10: 'orange'}
CH_NAMES = ['FC3', 'FC4', 'CP3', 'Cz', 'C3', 'C4', 'Pz', 'CP4']
SFREQ = 250
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "presentation_figures_2")
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
    # Filter for 4 classes
    mask = np.isin(y, [1, 2, 3, 10])
    X = X[mask]
    y = y[mask]
    
    # Filter 8-30 Hz (Mu + Beta) - Try to capture more info for Rest vs Active
    X_filt = mne.filter.filter_data(X, SFREQ, 8, 30, verbose=False)
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
    
    fig2 = plt.figure(figsize=(12, 10))
    # 2x2 Grid for 4 classes
    gs = fig2.add_gridspec(2, 2)
    
    axes_cov = [
        fig2.add_subplot(gs[0, 0]),
        fig2.add_subplot(gs[0, 1]),
        fig2.add_subplot(gs[1, 0]),
        fig2.add_subplot(gs[1, 1])
    ]
    
    # Enable global colorbar scaling
    # Calculate global min/max for comparable heatmaps
    means = []
    for c in [1, 2, 3, 10]:
        means.append(mean_covariance(covs[y==c], metric='riemann'))
    
    vmin = np.min(means)
    vmax = np.max(means)

    # Helper
    def plot_mat(ax, mat, title):
        im = ax.imshow(mat, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(CH_NAMES)))
        ax.set_yticks(np.arange(len(CH_NAMES)))
        ax.set_xticklabels(CH_NAMES)
        ax.set_yticklabels(CH_NAMES)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    # Plot each class
    plot_mat(axes_cov[0], means[0], f"Moyenne {CLASSES[1]}")
    plot_mat(axes_cov[1], means[1], f"Moyenne {CLASSES[2]}")
    plot_mat(axes_cov[2], means[2], f"Moyenne {CLASSES[3]}")
    plot_mat(axes_cov[3], means[3], f"Moyenne {CLASSES[10]}")

    plt.suptitle("SIGNATURES DE COVARIANCE (RIEMANN)\n(Chaque classe a une structure unique)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "slide_2_covariance_4class.png"))
    plt.close()

    # ---------------------------------------------------------
    # FIGURE 3: LEFT vs RIGHT COMPARISON (Riemannian Diff)
    # ---------------------------------------------------------
    print("   3. Génération figure Comparaison Gauche/Droite (Riemann)...")
    
    mean_G = means[0] # Left (Index 0 in loop list [1, 2, 3, 10])
    mean_D = means[1] # Right (Index 1)
    
    # Log-Euclidean Difference: Log(G) - Log(D)
    # Maps matrices to Tangent Space at Identity, where subtraction is valid
    diff_riemann = np.real(logm(mean_G)) - np.real(logm(mean_D))
    
    fig3 = plt.figure(figsize=(18, 5))
    gs3 = fig3.add_gridspec(1, 3)
    ax_g = fig3.add_subplot(gs3[0, 0])
    ax_d = fig3.add_subplot(gs3[0, 1])
    ax_diff = fig3.add_subplot(gs3[0, 2])
    
    # Plot Means (using global scale vmin/vmax)
    plot_mat(ax_g, mean_G, "Covariance Moyenne GAUCHE")
    plot_mat(ax_d, mean_D, "Covariance Moyenne DROITE")
    
    # Plot Difference
    limit = np.max(np.abs(diff_riemann))
    im_diff = ax_diff.imshow(diff_riemann, cmap='RdBu_r', vmin=-limit, vmax=limit)
    ax_diff.set_xticks(np.arange(len(CH_NAMES)))
    ax_diff.set_yticks(np.arange(len(CH_NAMES)))
    ax_diff.set_xticklabels(CH_NAMES)
    ax_diff.set_yticklabels(CH_NAMES)
    ax_diff.set_title("DIFFÉRENCE LOG-EUCLIDIENNE (Riemann)\nLog(G) - Log(D)")
    plt.colorbar(im_diff, ax=ax_diff)
    
    plt.suptitle("COMPARAISON GAUCHE vs DROITE (Géométrie Riemannienne)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "slide_2_covariance_success.png"))
    plt.close()
    

    
    # ---------------------------------------------------------
    # FIGURE 5: MULTICLASS GEOMETRY (MDS in Tangent Space)
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # FIGURE 5: MULTICLASS GEOMETRY (Centroids MDS)
    # ---------------------------------------------------------
    print("   4. Génération figure Géométrie Multiclasse (Centroids)...")
    
    # 1. Compute Class Means and Global Mean
    means_list = []
    labels_list = []
    colors_list = []
    
    # Global Mean (Center of the Manifold for this subject)
    mean_global = mean_covariance(covs, metric='riemann')
    means_list.append(mean_global)
    labels_list.append("Moyenne Globale (Centre)")
    colors_list.append('black')
    
    # Class Means
    for c in CLASSES:
        # Check if we have enough data for this class
        covs_c = covs[y==c]
        if len(covs_c) > 0:
            m = mean_covariance(covs_c, metric='riemann')
            means_list.append(m)
            labels_list.append(CLASSES[c])
            colors_list.append(COLORS[c])
            
    means_array = np.array(means_list)
    
    # 2. Compute Pairwise Riemannian Distances Matrix (5x5)
    from pyriemann.utils.distance import distance_riemann
    n_points = len(means_array)
    dist_matrix = np.zeros((n_points, n_points))
    
    for i in range(n_points):
        for j in range(n_points):
            dist_matrix[i, j] = distance_riemann(means_array[i], means_array[j])
            
    # 3. MDS to project this Distance Matrix to 2D
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    X_2d_means = mds.fit_transform(dist_matrix)
    
    # 4. PLOTTING
    fig5, ax5 = plt.subplots(figsize=(8, 8))
    
    # Draw connections from Global Mean (Index 0) to others
    center = X_2d_means[0]
    for i in range(1, n_points):
        point = X_2d_means[i]
        ax5.plot([center[0], point[0]], [center[1], point[1]], 
                 linestyle='--', color='gray', alpha=0.5)
        
    # Plot Points
    for i in range(n_points):
        size = 300 if i == 0 else 500
        marker = 'o' if i == 0 else 'D'
        edge = 'black'
        
        ax5.scatter(X_2d_means[i, 0], X_2d_means[i, 1], 
                    c=colors_list[i], s=size, marker=marker, 
                    edgecolors=edge, linewidth=2, label=labels_list[i], zorder=10)
        
        # Add Text Label offset
        ax5.text(X_2d_means[i, 0]+0.02, X_2d_means[i, 1]+0.02, labels_list[i], 
                 fontsize=12, fontweight='bold')

    ax5.set_title("STRUCTURE GÉOMÉTRIQUE IDÉALE (Simplex)\n(Distances Riemanniennes entre les Moyennes)", fontsize=14, fontweight='bold')
    
    # Equal aspect ratio to preserve geometry
    ax5.axis('equal')
    ax5.grid(True, alpha=0.3, linestyle='--')
    
    # Hide axis ticks (they don't mean much in MDS relative space)
    ax5.set_xticks([])
    ax5.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "slide_5_multiclass_simplex.png"))
    plt.close()
    
    print("✅ Terminé.")



    # ---------------------------------------------------------
    # FIGURE 6: 3D MULTICLASS CLOUDS (Tangent Space MDS)
    # ---------------------------------------------------------
    print("   5. Génération figure Géométrie Multiclasse 3D (Nuages)...")

    # 0. Calculate MDM and Tangent Space Accuracy (Multiclass)
    print("      -> Calcul de la précision (Validation Croisée)...")
    
    # MDM
    mdm = MDM(metric='riemann')
    scores_mdm = cross_val_score(mdm, covs, y, cv=5)
    acc_mdm = scores_mdm.mean() * 100
    
    # Tangent Space + Logistic Regression (State of the Art)
    clf_ts = make_pipeline(TangentSpace(metric='riemann'), LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000))
    scores_ts = cross_val_score(clf_ts, covs, y, cv=5)
    acc_ts = scores_ts.mean() * 100
    
    print(f"      ✅ Précision MDM : {acc_mdm:.1f}% (+/- {scores_mdm.std()*100:.1f}%)")
    print(f"      🚀 Précision Tangent Space (LR) : {acc_ts:.1f}% (+/- {scores_ts.std()*100:.1f}%)")

    # 1. Map ALL Covariances to Tangent Space (Reference = Global Mean)
    mean_global = mean_covariance(covs, metric='riemann')
    ts = TangentSpace(metric='riemann', tsupdate=False)
    # Fit on all, but we transform all
    X_ts = ts.fit_transform(covs)
    
    # 2. MDS 3D on ALL points (might be slow if N is huge, but usually fine for <5000)
    print(f"      -> MDS on {X_ts.shape[0]} points...")
    mds_3d = MDS(n_components=3, dissimilarity='euclidean', random_state=42, n_init=2, max_iter=100)
    X_3d = mds_3d.fit_transform(X_ts)
    
    # Define Views (Elev, Azim)
    views = [
        (30, -60, "Vue Standard"),
        (90, -90, "Vue de Haut (2D)"),
        (10, 45, "Vue de Côté")
    ]
    
    for v_idx, (elev, azim, v_name) in enumerate(views):
        fig6 = plt.figure(figsize=(10, 10))
        ax6 = fig6.add_subplot(111, projection='3d')
        
        # Plot Clouds
        for c in CLASSES:
            mask_c = (y == c)
            # Scatter Points (Opaque alpha=1)
            ax6.scatter(X_3d[mask_c, 0], X_3d[mask_c, 1], X_3d[mask_c, 2],
                        c=COLORS[c], s=20, alpha=1.0, label=CLASSES[c] if v_idx==0 else "")
            
            # Plot Centroid (Mean of 3D points)
            centroid = np.mean(X_3d[mask_c], axis=0)
            ax6.scatter(centroid[0], centroid[1], centroid[2],
                        c=COLORS[c], s=400, marker='X', 
                        edgecolors='black', linewidth=2, label=f"Moyenne {CLASSES[c]}" if v_idx==0 else "")
            
            # Add text
            ax6.text(centroid[0], centroid[1], centroid[2], CLASSES[c], fontsize=12, fontweight='bold', color='black')

        # Draw Global Mean Centroid (approx 0,0,0 usually but let's compute)
        global_center = np.mean(X_3d, axis=0)
        ax6.scatter(global_center[0], global_center[1], global_center[2], c='black', s=200, marker='o')

        ax6.set_title(f"GÉOMÉTRIE DES NUAGES DE POINTS (3D)\n{v_name} | MDM: {acc_mdm:.1f}% | TS+LR: {acc_ts:.1f}%", fontsize=14, fontweight='bold')
        
        ax6.view_init(elev=elev, azim=azim)
        
        ax6.set_xlabel('Dim 1')
        ax6.set_ylabel('Dim 2')
        ax6.set_zlabel('Dim 3')
        if v_idx == 0:
            ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        filename = f"slide_6_multiclass_clouds_3d_view{v_idx+1}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        plt.close()
        print(f"      -> Sauvegardé {filename}")
    
    print("✅ Figures 3D Terminées.")

if __name__ == "__main__":
    main()

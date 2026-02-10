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
CH_NAMES = ['FC3', 'FC4', 'CP3', 'Cz', 'C3', 'C4', 'Pz', 'CP4']
SFREQ = 250
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "presentation_figures")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed_data")

# Pair definitions
PAIRS = [
    # (Label A, Name A, Label B, Name B, Chan A, Chan B, Suffix)
    # Feet(3) vs Left(1). Feet->Cz active. Left->C4 active (Right motor area).
    (3, 'Pieds', 1, 'Gauche', 'Cz', 'C4', 'pieds_vs_gauche'),
    # Feet(3) vs Right(2). Feet->Cz active. Right->C3 active (Left motor area).
    (3, 'Pieds', 2, 'Droite', 'Cz', 'C3', 'pieds_vs_droite')
]

COLORS = {1: 'blue', 2: 'red', 3: 'green'}

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
    eigvals, eigvecs = np.linalg.eigh(cov_ref)
    eigvals_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    whiten = eigvecs @ eigvals_inv_sqrt @ eigvecs.T
    target_white = whiten @ cov_target @ whiten.T
    eigvals_w, eigvecs_w = np.linalg.eigh(target_white)
    log_eigvals = np.diag(np.log(eigvals_w))
    return eigvecs_w @ log_eigvals @ eigvecs_w.T

def generate_amplitude_slide(X, y, pair, output_path):
    la, name_a, lb, name_b, ch_a, ch_b, _ = pair
    print(f"   📊 Slide 1 (Amplitude): {name_a} vs {name_b}...")
    
    idx_a = CH_NAMES.index(ch_a)
    idx_b = CH_NAMES.index(ch_b)
    
    # Power
    pow_a = np.var(X[:, idx_a, :], axis=1)
    pow_b = np.var(X[:, idx_b, :], axis=1)
    log_a = 10 * np.log10(pow_a)
    log_b = 10 * np.log10(pow_b)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter
    mask_a = (y == la)
    mask_b = (y == lb)
    
    axes[0].scatter(log_a[mask_a], log_b[mask_a], c=COLORS[la], label=name_a, alpha=0.6, edgecolors='w')
    axes[0].scatter(log_a[mask_b], log_b[mask_b], c=COLORS[lb], label=name_b, alpha=0.6, edgecolors='w')
    axes[0].set_xlabel(f"Puissance {ch_a} (dB) - Zone {name_a}")
    axes[0].set_ylabel(f"Puissance {ch_b} (dB) - Zone {name_b}")
    axes[0].set_title("Chevauchement des Amplitudes")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Boxplot
    # Diff score: Power(Ch_A) - Power(Ch_B)
    # Hypothesis: If class A, Ch_A should drop (ERD).
    score = log_a - log_b
    data = [score[mask_a], score[mask_b]]
    bplot = axes[1].boxplot(data, patch_artist=True, labels=[name_a, name_b])
    for patch, color in zip(bplot['boxes'], [COLORS[la], COLORS[lb]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    axes[1].set_title(f"Score Différentiel ({ch_a} - {ch_b})")
    axes[1].grid(True, axis='y')
    
    plt.suptitle(f"AMPLITUDE: {name_a} vs {name_b}\n(Séparation difficile)", fontsize=16)
    plt.savefig(output_path)
    plt.close()

def generate_covariance_slide(covs, y, pair, output_path):
    la, name_a, lb, name_b, _, _, _ = pair
    print(f"   🧠 Slide 2 (Covariance): {name_a} vs {name_b}...")
    
    mean_a = mean_covariance(covs[y==la], metric='riemann')
    mean_b = mean_covariance(covs[y==lb], metric='riemann')
    diff = mean_a - mean_b
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sns.heatmap(mean_a, ax=axes[0], cmap='Greens' if la==3 else 'Blues', square=True, cbar=False, xticklabels=CH_NAMES, yticklabels=CH_NAMES)
    axes[0].set_title(f"Moyenne {name_a}")
    
    sns.heatmap(mean_b, ax=axes[1], cmap='Reds' if lb==2 else 'Blues', square=True, cbar=False, xticklabels=CH_NAMES, yticklabels=CH_NAMES)
    axes[1].set_title(f"Moyenne {name_b}")
    
    limit = np.max(np.abs(diff))
    sns.heatmap(diff, ax=axes[2], cmap='RdBu_r', square=True, vmin=-limit, vmax=limit, xticklabels=CH_NAMES, yticklabels=CH_NAMES)
    axes[2].set_title("Différence (Structurelle)")
    
    plt.suptitle(f"EMPREINTES RIEMANNIENNES: {name_a} vs {name_b}", fontsize=16)
    plt.savefig(output_path)
    plt.close()

def generate_distance_slide(covs, y, pair, output_path):
    la, name_a, lb, name_b, _, _, _ = pair
    print(f"   📏 Slide 3 (Distance): {name_a} vs {name_b}...")
    
    mean_a = mean_covariance(covs[y==la], metric='riemann')
    mean_b = mean_covariance(covs[y==lb], metric='riemann')
    
    dists_a = []
    dists_b = []
    labels = []
    
    # Select only trials from these two classes
    mask = np.isin(y, [la, lb])
    sub_covs = covs[mask]
    sub_y = y[mask]
    
    correct = 0
    for i in range(len(sub_covs)):
        da = distance_riemann(sub_covs[i], mean_a)
        db = distance_riemann(sub_covs[i], mean_b)
        dists_a.append(da)
        dists_b.append(db)
        labels.append(sub_y[i])
        
        pred = la if da < db else lb
        if pred == sub_y[i]: correct += 1
        
    acc = correct / len(sub_y)
    
    plt.figure(figsize=(8, 8))
    dists_a = np.array(dists_a)
    dists_b = np.array(dists_b)
    labels = np.array(labels)
    
    plt.scatter(dists_a[labels==la], dists_b[labels==la], c=COLORS[la], label=name_a, edgecolors='w', s=60)
    plt.scatter(dists_a[labels==lb], dists_b[labels==lb], c=COLORS[lb], label=name_b, edgecolors='w', s=60)
    
    lims = [np.min([plt.xlim(), plt.ylim()]), np.max([plt.xlim(), plt.ylim()])]
    plt.plot(lims, lims, 'k--', alpha=0.5, label='Frontière')
    
    plt.xlabel(f"Distance à {name_a}")
    plt.ylabel(f"Distance à {name_b}")
    plt.title(f"Classification Riemannienne: {name_a} vs {name_b}\nAcc: {acc:.1%}")
    plt.legend()
    plt.grid(True, linestyle=':')
    
    plt.savefig(output_path)
    plt.close()

def generate_diff_theory_slide(covs, y, pair, output_path):
    la, name_a, lb, name_b, _, _, _ = pair
    print(f"   🌌 Slide 4 (Tangent): {name_a} vs {name_b}...")
    
    mean_a = mean_covariance(covs[y==la], metric='riemann')
    mean_b = mean_covariance(covs[y==lb], metric='riemann')
    
    diff_euclid = mean_a - mean_b
    diff_riemann = compute_riemann_log_map(mean_a, mean_b) # Map A into B's space roughly (vector B->A)
    # Actually order matters for sign but visualization is symmetric in magnitude structure
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ce = np.max(np.abs(diff_euclid))
    cr = np.max(np.abs(diff_riemann))
    
    sns.heatmap(diff_euclid, ax=axes[0], cmap='RdBu_r', square=True, vmin=-ce, vmax=ce, xticklabels=CH_NAMES, yticklabels=CH_NAMES)
    axes[0].set_title("Différence Euclidienne")
    
    sns.heatmap(diff_riemann, ax=axes[1], cmap='RdBu_r', square=True, vmin=-cr, vmax=cr, xticklabels=CH_NAMES, yticklabels=CH_NAMES)
    axes[1].set_title("Différence Riemannienne (Tangent)")
    
    plt.suptitle(f"COMPARAISON THÉORIQUE: {name_a} vs {name_b}", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    print("🚀 Génération des slides 'Pieds' (Comparaisons croisées)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    X, y = load_data()
    # Filter only relevant classes (1, 2, 3)
    mask = np.isin(y, [1, 2, 3])
    X = X[mask]
    y = y[mask]
    
    # 2. Filter Mu & Crop
    X_filt = mne.filter.filter_data(X, SFREQ, 8, 13, verbose=False)
    X_crop = X_filt[:, :, int(1.0*SFREQ):int(3.5*SFREQ)]
    
    # 3. Pre-compute Covariances
    cov_est = Covariances(estimator='lwf')
    covs = cov_est.fit_transform(X_crop)
    
    # Iterate over pairs
    for pair in PAIRS:
        la, name_a, lb, name_b, ch_a, ch_b, suffix = pair
        print(f"\n👉 Traitement paire : {name_a} vs {name_b}")
        
        # Subset data for this pair (for scatter plots etc)
        # Note: Slide generation functions do their own masking/subsetting or accept full data and mask inside.
        # Passing full X_crop/covs and y is easier.
        
        generate_amplitude_slide(X_crop, y, pair, os.path.join(OUTPUT_DIR, f"slide_1_bis_{suffix}_amplitude.png"))
        generate_covariance_slide(covs, y, pair, os.path.join(OUTPUT_DIR, f"slide_2_bis_{suffix}_covariance.png"))
        generate_distance_slide(covs, y, pair, os.path.join(OUTPUT_DIR, f"slide_3_bis_{suffix}_distance.png"))
        generate_diff_theory_slide(covs, y, pair, os.path.join(OUTPUT_DIR, f"slide_4_bis_{suffix}_tangent.png"))
        
    print("\n✅ Terminé toutes les variations !")

if __name__ == "__main__":
    main()

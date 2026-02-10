import numpy as np
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from pyriemann.estimation import Covariances
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.mean import mean_covariance
from pyriemann.classification import MDM
from sklearn.model_selection import cross_val_score, StratifiedKFold
import os
import glob

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

def main():
    print("📏 Évaluation de la Séparabilité Riemannienne...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load & Filter
    X, y = load_data()
    if X is None: return
    
    # Filter 8-13Hz (Mu Band)
    X_filt = mne.filter.filter_data(X, SFREQ, 8, 13, verbose=False)
    
    # Crop (Active part)
    X_crop = X_filt[:, :, int(1.0*SFREQ):int(3.5*SFREQ)]
    
    # 2. Covariances
    cov_est = Covariances(estimator='lwf')
    covs = cov_est.fit_transform(X_crop)
    
    # 3. Compute Means
    classes = [1, 2, 3] # Left, Right, Feet
    class_names = ['Gauche', 'Droite', 'Pieds']
    means = []
    
    for c in classes:
        mean_c = mean_covariance(covs[y==c], metric='riemann')
        means.append(mean_c)
        
    # 4. Compute Pairwise Distances (The "Signal Strength")
    n_classes = len(classes)
    dist_matrix = np.zeros((n_classes, n_classes))
    
    print("\n--- Distances Riemanniennes entre Centres de Classes ---")
    for i in range(n_classes):
        for j in range(n_classes):
            d = distance_riemann(means[i], means[j])
            dist_matrix[i, j] = d
            print(f"   {class_names[i]} <-> {class_names[j]} : {d:.4f}")

    # 5. Cross-Validation Score (The "Robustness")
    mdm = MDM()
    # 5-Fold, repeated 5 times for stability
    scores = []
    for i in range(5):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42+i)
        sc = cross_val_score(mdm, covs, y, cv=cv, n_jobs=1)
        scores.extend(sc)
        
    mean_acc = np.mean(scores)
    std_acc = np.std(scores)
    chance = 1/3.0
    
    print(f"\n--- Performance Prédictive (MDM) ---")
    print(f"   Précision Moyenne : {mean_acc:.2%} (+/- {std_acc:.2%})")
    print(f"   Niveau Chance : {chance:.2%}")

    # 6. Visualization
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Distance Matrix
    sns.heatmap(dist_matrix, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names, ax=ax[0], cbar=False)
    ax[0].set_title("Distance Riemannienne Moyenne\n(Plus c'est bleu foncé, mieux c'est)")
    
    # Plot 2: Accuracy Gauge (Simulated)
    # Just a simple bar chart comparing to chance
    ax[1].bar(["Hasard", "Votre Data (MDM)"], [chance, mean_acc], color=['gray', 'green'])
    ax[1].axhline(y=chance, color='gray', linestyle='--')
    ax[1].set_ylim(0, 1.0)
    ax[1].set_ylabel("Précision")
    ax[1].set_title(f"Score de Séparabilité : {mean_acc:.1%}")
    for i, v in enumerate([chance, mean_acc]):
        ax[1].text(i, v + 0.02, f"{v:.1%}", ha='center', fontweight='bold')

    plt.suptitle("Analyse de Séparabilité Riemannienne", fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, "class_separability.png")
    plt.savefig(save_path)
    print(f"✅ Image générée : {save_path}")
    
    # Interpretation Text
    print("\n📝 ANALYSE :")
    if mean_acc > 0.55:
        print("   EXCELLENT. Vos classes sont bien distinctes.")
    elif mean_acc > 0.40:
        print("   MOYEN. Il y a de l'information, mais c'est bruité.")
    else:
        print("   FAIBLE. Les classes se chevauchent trop.")

if __name__ == "__main__":
    main()

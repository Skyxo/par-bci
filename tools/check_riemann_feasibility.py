
import numpy as np
import mne
import matplotlib.pyplot as plt
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.manifold import TSNE
import os
import glob
import sys

# --- CONFIGURATION ---
# Correspondance des canaux (Doit matcher preprocess_data_to_npy.py)
CH_NAMES = ['FC3', 'FC4', 'CP3', 'Cz', 'C3', 'C4', 'Pz', 'CP4']
DROP_CHANS = ['C3'] # User request: Test separate without C3
SFREQ = 250

# Dossier de sortie
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "audit_results")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed_data")

def load_processed_data():
    print(f"📥 Chargement des données depuis : {PROCESSED_DIR}")
    
    X_files = glob.glob(os.path.join(PROCESSED_DIR, "X_*.npy"))
    if not X_files:
        print("❌ Aucune donnée prétraitée trouvée (.npy).")
        print("💡 Astuce : Lancez d'abord 'python tools/preprocess_data_to_npy.py'")
        return None, None

    X_list = []
    y_list = []

    for f in X_files:
        base = os.path.basename(f).replace("X_", "y_")
        y_f = os.path.join(PROCESSED_DIR, base)
        
        if os.path.exists(y_f):
            print(f"   -> {os.path.basename(f)}")
            X_part = np.load(f)
            y_part = np.load(y_f)
            X_list.append(X_part)
            y_list.append(y_part)

    if not X_list: return None, None

    # Concaténation
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    
    return X, y

def main():
    # 1. CHARGEMENT (DATASET EXACT)
    X, y = load_processed_data()
    if X is None: return

    print(f"📊 Données brutes chargées : {X.shape} (N_epochs, N_chans, N_times)")
    print(f"ℹ️ Classes disponibles : {np.unique(y, return_counts=True)}")

    # 1.5 DROPPING BAD CHANNELS (Ablation Study)
    if DROP_CHANS:
        print(f"✂️ ABLATION: Suppression des canaux {DROP_CHANS}")
        # Find indices
        keep_indices = [i for i, name in enumerate(CH_NAMES) if name not in DROP_CHANS]
        new_ch_names = [CH_NAMES[i] for i in keep_indices]
        
        X = X[:, keep_indices, :]
        print(f"   -> Nouvelle forme : {X.shape}")
        current_ch_names = new_ch_names
    else:
        current_ch_names = CH_NAMES

    # 2. FILTRAGE SPECIFIQUE (Miroir de ultra_riemannian_training.py)
    # Les données .npy sont déjà filtrées 1-40Hz. 
    # Pour Riemann/Motor Imagery, on isole la bande Mu (8-13Hz).
    print("🔄 Filtrage Bande Mu (8-13 Hz)...")
    X_mu = mne.filter.filter_data(X, SFREQ, 8, 13, verbose=False)

    # 3. SELECTION TEMPORELLE (CROP)
    # Le script d'entrainement utilise [1.0s - 3.0s] pour éviter les potentiels évoqués visuels du début
    t_start = 1.0
    t_end = 3.0
    idx_start = int(t_start * SFREQ)
    idx_end = int(t_end * SFREQ)
    
    # Vérification bounds
    if idx_end > X_mu.shape[2]: idx_end = X_mu.shape[2]
    
    X_crop = X_mu[:, :, idx_start:idx_end]
    print(f"✂️ Découpage Temporel [{t_start}s - {t_end}s] -> Forme : {X_crop.shape}")

    # 4. PREPARATION MNE (Pour Topomaps)
    # On crée une structure EpochsArray pour utiliser les outils de plot MNE
    info = mne.create_info(current_ch_names, SFREQ, 'eeg')
    info.set_montage('standard_1020')
    
    # On filtre Gauche (1), Droite (2) et Pieds (3)
    mask_viz = np.isin(y, [1, 2, 3])
    X_viz = X_crop[mask_viz]
    y_viz = y[mask_viz]
    
    # Mapping labels -> Noms
    event_id = {'Gauche': 1, 'Droite': 2, 'Pieds': 3}
    # Events array pour MNE : colonne 2 contient le label
    events = np.column_stack((np.arange(len(y_viz)), np.zeros(len(y_viz), dtype=int), y_viz.astype(int)))
    
    epochs = mne.EpochsArray(X_viz, info, events=events, event_id=event_id, tmin=t_start, verbose=False)

    # =========================================================
    # PREUVE 1 : TOPOGRAPHIES (Latéralisation)
    # =========================================================
    print("\n🎨 Génération des cartes mentales (Topomaps)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        fig_topo, ax = plt.subplots(1, 3, figsize=(15, 4))
        
        # On définit explicitement la bande à afficher pour éviter que MNE ne cherche
        # à afficher les 5 bandes par défaut (Delta, Theta, ...) sur un seul axe.
        bands = {'Mu': (8, 13)}

        # Note: on a déjà filtré le signal (X_viz) en 8-13Hz, donc le PSD sera concentré là.
        epochs['Gauche'].compute_psd().plot_topomap(bands=bands, axes=ax[0], show=False, cmap='viridis')
        ax[0].set_title("Pensée GAUCHE\n(Attendu: Bleu/Faible à C4)")
        
        epochs['Droite'].compute_psd().plot_topomap(bands=bands, axes=ax[1], show=False, cmap='viridis')
        ax[1].set_title("Pensée DROITE\n(Attendu: Bleu/Faible à C3)")

        epochs['Pieds'].compute_psd().plot_topomap(bands=bands, axes=ax[2], show=False, cmap='viridis')
        ax[2].set_title("Pensée PIEDS\n(Attendu: Bleu/Faible à Cz)")
        
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, 'riemann_topomaps.png')
        plt.savefig(save_path)
        print(f"✅ Topomaps sauvegardées : {save_path}")
        plt.close(fig_topo)
    except Exception as e:
        print(f"❌ Erreur Topomaps: {e}")

    # =========================================================
    # PREUVE 2 : GÉOMÉTRIE (Cluster t-SNE)
    # =========================================================
    print("\n🧮 Calcul de la géométrie Riemannienne (t-SNE)...")
    try:
        # Covariances sur le signal croppé et filtré
        cov_est = Covariances(estimator='lwf')
        covs = cov_est.fit_transform(X_viz)

        # Projection Espace Tangent
        ts = TangentSpace()
        features = ts.fit_transform(covs)

        # t-SNE
        tsne = TSNE(n_components=2, perplexity=min(30, len(features)-1), random_state=42, init='pca', learning_rate='auto')
        X_embedded = tsne.fit_transform(features)

        # Plot
        fig_tsne = plt.figure(figsize=(8, 6))
        
        # Masques pour couleurs
        mask_g = (y_viz == 1)
        mask_d = (y_viz == 2)
        mask_p = (y_viz == 3)
        
        plt.scatter(X_embedded[mask_g, 0], X_embedded[mask_g, 1], c='blue', label='Gauche', alpha=0.7)
        plt.scatter(X_embedded[mask_d, 0], X_embedded[mask_d, 1], c='red', label='Droite', alpha=0.7)
        plt.scatter(X_embedded[mask_p, 0], X_embedded[mask_p, 1], c='green', label='Pieds', alpha=0.7)
        
        plt.title(f"Séparabilité Riemannienne (t-SNE)\n{len(features)} essais analysés")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path_tsne = os.path.join(OUTPUT_DIR, 'riemann_tsne.png')
        plt.savefig(save_path_tsne)
        print(f"✅ t-SNE sauvegardé : {save_path_tsne}")
        plt.close(fig_tsne)
        
    except Exception as e:
        print(f"❌ Erreur t-SNE: {e}")

if __name__ == "__main__":
    main()

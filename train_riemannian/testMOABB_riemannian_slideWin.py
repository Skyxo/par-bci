import os
import sys

# --- CONFIGURATION CRITIQUE (A PLACER EN PREMIÈRE LIGNE ABSOLUE) ---
# 1. Définir un dossier local dans votre projet pour éviter les problèmes de droits
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOABB_DATA_PATH = os.path.join(BASE_DIR, "moabb_data")

# 2. Créer le dossier physiquement MAINTENANT (C'est ça qui manquait)
if not os.path.exists(MOABB_DATA_PATH):
    try:
        os.makedirs(MOABB_DATA_PATH)
        print(f"✅ Dossier créé : {MOABB_DATA_PATH}")   
    except PermissionError:
        print("❌ Erreur de permission. Essayez de lancer le script avec 'sudo' ou changez de dossier.")
        sys.exit(1)

# 3. Forcer les variables d'environnement AVANT d'importer les librairies
os.environ['MNE_DATA'] = MOABB_DATA_PATH
os.environ['MOABB_RESULTS'] = os.path.join(MOABB_DATA_PATH, "results")

# 4. Configuration explicite de MNE
import mne
# On désactive les warnings de configuration pour y voir plus clair
mne.set_log_level('WARNING') 
mne.set_config('MNE_DATA', MOABB_DATA_PATH, set_env=True)


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Important sur Mac
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# --- 1. TES IMPORTS (Les pièces du moteur) ---
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin

# --- IMPORTS MOABB (Le banc d'essai) ---
from moabb.datasets import BNCI2014001
from moabb.paradigms import MotorImagery
from moabb.evaluations import WithinSessionEvaluation

# --- 1. FONCTIONS OUTILS (Indispensables) ---

class DelayEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self, delays=[4,8]):
        """
        delays: liste des décalages en 'samples'. 
        Exemple: [4] signifie qu'on ajoute t-4 (environ 16ms à 250Hz)
        """
        self.delays = delays

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X shape: (n_epochs, n_channels, n_times)
        n_epochs, n_channels, n_times = X.shape
        
        # On va stocker les signaux originaux + les signaux décalés
        X_aug = [X]
        
        for d in self.delays:
            if d > 0:
                # Décalage vers le passé (t-d)
                # On pad avec des zéros au début pour garder la même taille
                shifted = np.roll(X, shift=d, axis=2)
                shifted[:, :, :d] = 0 
                X_aug.append(shifted)
            elif d < 0:
                # Décalage vers le futur (t+d) (si on veut t+1)
                # Attention, en temps réel pur, le futur n'existe pas !
                # Mais en analyse de fenêtre glissante (buffer), on peut le faire.
                shifted = np.roll(X, shift=d, axis=2)
                shifted[:, :, d:] = 0
                X_aug.append(shifted)
                
        # On concatène sur l'axe des canaux (axis 1)
        # Résultat: (n_epochs, n_channels * (1 + n_delays), n_times)
        return np.concatenate(X_aug, axis=1)


def apply_bandpass(X, lowcut, highcut, sfreq=250):
    nyq = 0.5 * sfreq
    b, a = butter(4, [lowcut/nyq, highcut/nyq], btype='band')
    return lfilter(b, a, X, axis=2)

# --- 2. TA FONCTION DE PIPELINE (Déduite de ton code) ---
def make_my_pipeline():
    # On définit la fréquence par défaut (250Hz pour BCI Comp IV 2a)
    sfreq = 250 
    delays = [4, 8]  # Décalages en samples (16ms, 32ms à 250Hz)

    # Branche 1 : Mu (8-12 Hz) - Note: tu as changé 13 en 12 dans ton code
    cov_mu = make_pipeline(
        FunctionTransformer(apply_bandpass, kw_args={'lowcut':8, 'highcut':12, 'sfreq':sfreq}, validate=False),
        DelayEmbedding(delays=delays),
        Covariances(estimator='scm'), 
        TangentSpace()
    )
    
    # Branche 2 : Beta Low (12-20 Hz)
    cov_beta_low = make_pipeline(
        FunctionTransformer(apply_bandpass, kw_args={'lowcut':12, 'highcut':20, 'sfreq':sfreq}, validate=False),
        DelayEmbedding(delays=delays),
        Covariances(estimator='scm'), 
        TangentSpace()
    )
    
    # Branche 3 : Beta High (20-30 Hz)
    cov_beta_high = make_pipeline(
        FunctionTransformer(apply_bandpass, kw_args={'lowcut':20, 'highcut':30, 'sfreq':sfreq}, validate=False),
        DelayEmbedding(delays=delays),
        Covariances(estimator='scm'), 
        TangentSpace()
    )
    
    # Assemblage des 3 branches
    feats = FeatureUnion([
        ('mu', cov_mu), 
        ('beta_low', cov_beta_low), 
        ('beta_high', cov_beta_high)
    ])
    
    # Pipeline Final avec tes paramètres de régression logistique
    clf = make_pipeline(
        feats, 
        StandardScaler(), 
        LogisticRegression(solver='lbfgs', max_iter=2000, class_weight='balanced')
    )
    
    return clf

# --- 4. CONFIGURATION MOABB ---

if __name__ == "__main__":
    # A. On prépare le dictionnaire des pipelines à tester
    # C'est ici que le "branchement" final se fait.
    # Tu donnes un nom (clé) et tu appelles ta fonction (valeur)
    pipelines = {}
    pipelines['Mon_Algo_FilterBank'] = make_my_pipeline() 

    # B. On choisit le dataset (BCI Competition IV 2a)
    dataset = BNCI2014001()
    dataset.subject_list = [1, 2, 3] # Testons sur 3 sujets pour commencer

    # C. On définit les règles (3 classes, comme ton projet)
    # Important : MOABB va filtrer les données brutes entre 8 et 32Hz avant de te les donner
    paradigm = MotorImagery(events=['left_hand', 'right_hand', 'feet'], 
                            n_classes=3, 
                            fmin=8, fmax=32, tmin=0.5,
        tmax=3.25)

    # D. On lance l'évaluation
    print("Lancement du benchmark MOABB...")
    evaluation = WithinSessionEvaluation(paradigm=paradigm, datasets=[dataset], overwrite=True)
    
    # MOABB va prendre 'pipelines', et pour chaque sujet :
    # 1. Faire pipeline.fit(X_train, y_train)
    # 2. Faire pipeline.predict(X_test)
    # 3. Noter le score
    results = evaluation.process(pipelines)

    # --- 5. RÉSULTATS ---
    print("\n--- RÉSULTATS PAR SUJET ---")
    print(results.groupby('subject')['score'].mean())
    
    print("\n--- SCORE MOYEN GLOBAL ---")
    print(results['score'].mean())
    
    # Sauvegarde
    results.to_csv("benchmark_resultats.csv")
    print("\nRésultats sauvegardés dans 'benchmark_resultats.csv'")
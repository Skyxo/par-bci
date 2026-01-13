import pandas as pd
import numpy as np
import mne
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 1. CHARGEMENT DU FICHIER
# -----------------------
filename = "EEG_Session_2026-01-13_16-59.csv"
# BrainFlow exporte en CSV avec tabulations (\t) sans en-tête
df = pd.read_csv(filename, sep='\t', header=None)

# Conversion : OpenBCI (µV) -> MNE (Volts)
eeg_data = df.iloc[:, 1:9].values.T * 1e-6
markers = df.iloc[:, 23].values

# Création de l'objet MNE
ch_names = ['Cz', 'FCz', 'P3', 'Pz', 'C3', 'C4', 'O1', 'P4']
info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types='eeg')
raw = mne.io.RawArray(eeg_data, info)

# 2. TRAITEMENT DU SIGNAL
# -----------------------
# Filtrage Spatial & Fréquentiel : On garde 8-30Hz (Ondes Motrices)
raw.filter(8., 30., fir_design='firwin', verbose=False)

# Extraction des Événements
# On cherche où le marqueur change pour trouver le début des essais
diff_markers = np.diff(markers, prepend=0)
events_idx = np.where(np.isin(markers, [1, 2, 3]) & (diff_markers != 0))[0]
events_vals = markers[events_idx].astype(int)

# Création de la matrice d'événements pour MNE
events = np.column_stack((events_idx, np.zeros_like(events_idx), events_vals))
event_id = {'Main_Gauche': 1, 'Main_Droite': 2, 'Pieds': 3}

# Découpage (Epoching) : On prend 3 secondes après l'apparition de la flèche
epochs = mne.Epochs(raw, events, event_id, tmin=0.5, tmax=3.5, 
                    proj=False, baseline=None, verbose=False)

# 3. CLASSIFICATION RIEMANNIENNE
# ------------------------------
# C'est le "State-of-the-Art" pour l'OpenBCI
clf = make_pipeline(
    Covariances(estimator='lwf'),    # 1. Calcul des covariances robuste
    TangentSpace(),                  # 2. Projection dans un espace géométrique plat
    LogisticRegression(solver='lbfgs') # 3. Décision
)

# 4. ÉVALUATION DE LA PRÉCISION
# -----------------------------
# On simule une utilisation réelle avec une validation croisée (5-fold)
cv = StratifiedKFold(n_splits=5, shuffle=True)
scores = cross_val_score(clf, epochs.get_data(), epochs.events[:, -1], cv=cv, scoring='accuracy')

print(f"\n--- RÉSULTATS DE L'ANALYSE ---")
print(f"Nombre d'essais détectés : {len(epochs)}")
print(f"Précision Moyenne : {np.mean(scores)*100:.2f}%")
print(f"Chance théorique (au hasard) : 33.33%")

if np.mean(scores) > 0.6:
    print("✅ SUCCÈS : Le modèle arrive à différencier vos mouvements !")
else:
    print("⚠️ ATTENTION : La différenciation est difficile. Vérifiez le contact des électrodes C3/C4/Cz.")

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# 1. Ré-entraîner le modèle sur toutes les données
# (Cela permet de voir comment le modèle a appris sur ce jeu de données spécifique)
clf.fit(epochs.get_data(), epochs.events[:, -1])
y_pred = clf.predict(epochs.get_data())
y_true = epochs.events[:, -1]

# 2. Calculer la matrice
cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3])

# --- AFFICHAGE CONSOLE ---
print("\n" + "="*40)
print("      RÉSULTATS DÉTAILLÉS")
print("="*40)

print("\n1. MATRICE DE CONFUSION BRUTE :")
# Affichage manuel pour être lisible
print(f"{'':<10} | {'Prédit G':<10} | {'Prédit D':<10} | {'Prédit P':<10}")
print("-" * 46)
classes = ['Vrai G', 'Vrai D', 'Vrai P']
for i, row in enumerate(cm):
    print(f"{classes[i]:<10} | {row[0]:<10} | {row[1]:<10} | {row[2]:<10}")

print("\n2. RAPPORT DE PERFORMANCE :")
# Affiche la précision, le rappel et le score F1 pour chaque classe
print(classification_report(y_true, y_pred, target_names=['Gauche', 'Droite', 'Pieds']))
print("="*40 + "\n")

# 3. Afficher le graphique (Matplotlib)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Gauche', 'Droite', 'Pieds'])
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues')
plt.title(f"Matrice de Confusion (Précision Globale: {np.mean(y_pred == y_true)*100:.1f}%)")
plt.show()
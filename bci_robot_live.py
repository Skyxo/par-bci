import time
import numpy as np
import pandas as pd
import mne
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Mettez ici le nom EXACT de votre bon fichier CSV enregistré tout à l'heure
TRAIN_FILE = "EEG_Session_2026-01-13_16-59.csv" 

# Paramètres OpenBCI
COM_PORT = "COM10"  # Vérifiez votre port !
BOARD_ID = BoardIds.CYTON_BOARD.value

# Paramètres de Traitement
SFREQ = 250
WINDOW_SIZE = 3.0  # On analyse des fenêtres de 3 secondes (comme l'entraînement)
UPDATE_SPEED = 0.5 # On fait une prédiction toutes les 0.5 secondes (fenêtre glissante)

# ==========================================
# 2. ENTRAINEMENT DU MODÈLE (Sur vos données enregistrées)
# ==========================================
print(f"--- Chargement des données d'entraînement : {TRAIN_FILE} ---")
df = pd.read_csv(TRAIN_FILE, sep='\t', header=None)
eeg_train = df.iloc[:, 1:9].values.T * 1e-6
markers_train = df.iloc[:, 23].values

# Création structure MNE
info = mne.create_info(ch_names=['Cz', 'FCz', 'P3', 'Pz', 'C3', 'C4', 'O1', 'P4'], sfreq=SFREQ, ch_types='eeg')
raw_train = mne.io.RawArray(eeg_train, info)
raw_train.filter(8., 30., fir_design='firwin', verbose=False)

# Extraction événements
diff_markers = np.diff(markers_train, prepend=0)
events_idx = np.where(np.isin(markers_train, [1, 2, 3]) & (diff_markers != 0))[0]
events_vals = markers_train[events_idx].astype(int)
events = np.column_stack((events_idx, np.zeros_like(events_idx), events_vals))

# Epoching
epochs_train = mne.Epochs(raw_train, events, {'G':1, 'D':2, 'P':3}, tmin=0.5, tmax=3.5, 
                          proj=False, baseline=None, verbose=False)

# Pipeline Riemannien
clf = make_pipeline(
    Covariances(estimator='lwf'),
    TangentSpace(),
    LogisticRegression(solver='lbfgs')
)
clf.fit(epochs_train.get_data(), epochs_train.events[:, -1])
print("✅ MODÈLE ENTRAÎNÉ ET PRÊT (Basé sur votre session précédente)")

# ==========================================
# 3. BOUCLE DE CONTRÔLE TEMPS RÉEL
# ==========================================
params = BrainFlowInputParams()
params.serial_port = COM_PORT
board = BoardShim(BOARD_ID, params)

try:
    print("\n--- CONNEXION AU CASQUE... ---")
    board.prepare_session()
    board.start_stream()
    print("--- DÉMARRAGE DU CONTRÔLE ROBOT (Ctrl+C pour arrêter) ---")
    
    # On attend un peu pour remplir le buffer
    time.sleep(WINDOW_SIZE)
    
    while True:
        # 1. Récupérer les dernières secondes de données
        # On a besoin de WINDOW_SIZE secondes * 250 Hz échantillons
        n_samples = int(WINDOW_SIZE * SFREQ)
        data = board.get_current_board_data(n_samples)
        
        # Vérification si on a assez de données
        if data.shape[1] < n_samples:
            continue
            
        # 2. Préparer les données (Même format que l'entraînement)
        # Canaux 1-8 (indices 1 à 9 non inclus), conversion µV -> V
        eeg_live = data[1:9, :] * 1e-6
        
        # 3. Filtrage "Online" (Astuce: on filtre le buffer brut)
        # Note: Pour une vraie implémentation robuste, on utiliserait un filtre temps réel (sos),
        # mais ici on filtre la fenêtre statique pour simplifier.
        live_raw = mne.io.RawArray(eeg_live, info, verbose=False)
        live_raw.filter(8., 30., fir_design='firwin', verbose=False)
        
        # 4. Prédiction
        # On doit ajouter une dimension pour faire (1 essai, 8 canaux, n_temps)
        X_live = live_raw.get_data()[np.newaxis, :, :]
        
        # Le seuil de confiance (Probabilité)
        # clf.predict_proba renvoie [Prob_G, Prob_D, Prob_P]
        probs = clf.predict_proba(X_live)[0]
        prediction = np.argmax(probs) + 1 # +1 car nos classes sont 1, 2, 3
        confidence = np.max(probs)
        
        # 5. Logique de commande (seulement si confiant)
        COMMAND_THRESHOLD = 0.55 # Il faut être sûr à 55% minimum
        
        if confidence > COMMAND_THRESHOLD:
            if prediction == 1:
                cmd = "<< GAUCHE"
                # ICI : Code pour envoyer au robot (ex: socket.send('L'))
            elif prediction == 2:
                cmd = "DROITE >>"
                # ICI : Code pour envoyer au robot
            elif prediction == 3:
                cmd = "vv PIEDS (DESCENDRE) vv"
                # ICI : Code pour envoyer au robot
        else:
            cmd = "... (Incertain)"
            
        print(f"Intention: {cmd:25} | Confiance: {confidence:.2f} | G:{probs[0]:.2f} D:{probs[1]:.2f} P:{probs[2]:.2f}")
        
        time.sleep(UPDATE_SPEED)

except KeyboardInterrupt:
    print("Arrêt utilisateur.")
finally:
    if board.is_prepared():
        board.stop_stream()
        board.release_session()
    print("Déconnecté.")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams
from scipy.signal import butter, lfilter, welch, iirnotch

import os

# --- CONFIGURATION ---
BOARD_ID = BoardIds.CYTON_BOARD.value
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE_PATH = os.path.join(BASE_DIR, "EEG_Session_2026-01-13_15-30.csv")  # Remplacez par le nom de votre fichier
SAMPLING_RATE = 250  # Fréquence d'échantillonnage par défaut du Cyton

def load_data(file_path):
    """Charge les données BrainFlow depuis le CSV."""
    print(f"Chargement du fichier : {file_path}")
    # BrainFlow écrit les données avec les canaux en LIGNES et les échantillons en COLONNES.
    # Cependant, votre fichier semble avoir les échantillons en LIGNES (transposé par défaut de DataFilter.write_file en mode 'w' ? Non, DataFilter écrit std).
    # Vérifions : Le fichier a 168962 lignes. C'est beaucoup pour des canaux. Donc Samples = Lignes.
    # BrainFlow DataFilter.write_file écrit normalement (n_channels, n_samples).
    # Mais si vous avez utilisé pandas ou autre chose, ça peut varier.
    # Regardons le header ou le format. Votre fichier n'a pas de header.
    
    # Chargement standard avec pandas (supposant délimiteur tabulation si .csv Brainflow standard)
    # Note: BrainFlow utilise '\t' comme séparateur par défaut.
    try:
        data = pd.read_csv(file_path, sep='\t', header=None)
    except:
        data = pd.read_csv(file_path, sep=',', header=None)
        
    print(f"Dimensions des données brutes : {data.shape}")
    return data

def get_eeg_channels():
    """Récupère les indices des canaux EEG pour le masque Cyton."""
    # BoardShim.get_eeg_channels(BOARD_ID) renvoie les indices eeg.
    # Attention: Si le fichier a été transposé, il faudra adapter.
    eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
    marker_channel = BoardShim.get_marker_channel(BOARD_ID)
    return eeg_channels, marker_channel

def plot_raw_data(data, eeg_channels, marker_channel):
    """Affiche les données brutes et les marqueurs."""
    plt.figure(figsize=(15, 10))
    
    # Création d'un vecteur temps
    n_samples = data.shape[0]
    time = np.arange(n_samples) / SAMPLING_RATE
    
    # Plot EEG
    for i, ch in enumerate(eeg_channels):
        # On décale chaque canal pour mieux voir (offset)
        offset = i * 1000  # Ajustez l'offset selon l'amplitude
        plt.plot(time, data.iloc[:, ch] + offset, label=f'Ch {i+1}')
        
    # Plot Markers (multipliés pour être visibles)
    # Le canal marker est souvent à 0, et passe à une valeur (ex: 1, 2, 10) lors d'un event.
    markers = data.iloc[:, marker_channel]
    # On ne plot que les points où marker != 0
    event_indices = markers[markers != 0].index
    event_times = event_indices / SAMPLING_RATE
    event_values = markers[markers != 0]
    
    if len(event_values) > 0:
        plt.vlines(event_times, -1000, 8000, colors='red', linestyles='dashed', label='Marqueurs')
        for t, v in zip(event_times, event_values):
            plt.text(t, 8000, str(int(v)), color='red', fontsize=12)

    plt.title("Données EEG Brutes et Marqueurs")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude (uV) + Offset")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Filtre passe-bande simple."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def notch_filter(data, freq, fs, quality=30):
    """Filtre Notch stable avec iirnotch."""
    # quality (Q) = freq / bandwidth. Q=30 est bien pour 50Hz.
    b, a = iirnotch(freq, quality, fs)
    y = lfilter(b, a, data)
    return y

def plot_psd(data, eeg_channels):
    """Affiche la densité spectrale de puissance (PSD)."""
    plt.figure(figsize=(10, 6))
    
    for i, ch in enumerate(eeg_channels):
        signal = data.iloc[:, ch]
        
        # 1. Filtre Notch pour tuer le 50Hz secteur
        signal_notch = notch_filter(signal, 50.0, SAMPLING_RATE)
        
        # 2. Filtre Passe-bande 1-40Hz (On garde Alpha/Beta, on vire le bruit HF)
        signal_filtered = bandpass_filter(signal_notch, 1.0, 40.0, SAMPLING_RATE)
        
        f, Pxx = welch(signal_filtered, fs=SAMPLING_RATE, nperseg=SAMPLING_RATE*2)
        plt.semilogy(f, Pxx, label=f'Ch {i+1}')
        
    plt.title("Densité Spectrale (PSD) - Nettoyée (Notch 50Hz + Bandpass 1-40Hz)")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Puissance (uV^2/Hz)")
    plt.xlim(0, 40) # Focus sur la zone utile (Mu/Beta)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # 1. Charger les données
    try:
        raw_data = load_data(FILE_PATH)
    except FileNotFoundError:
        print(f"Erreur: Le fichier {FILE_PATH} est introuvable.")
        return

    # 2. Identifier les canaux
    eeg_channels, marker_channel = get_eeg_channels()
    print(f"Canaux EEG (indices): {eeg_channels}")
    print(f"Canal Marqueur (indice): {marker_channel}")

    # 3. Visualiser signal brut + marqueurs
    print("Affichage des signaux bruts...")
    plot_raw_data(raw_data, eeg_channels, marker_channel)

    # 4. Visualiser le spectre fréquentiel (PSD)
    print("Calcul et affichage de la PSD...")
    plot_psd(raw_data, eeg_channels)

if __name__ == "__main__":
    main()

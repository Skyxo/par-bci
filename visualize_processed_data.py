import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import os

# Configuration
DATA_DIR = 'data/processed'
X_PATH = os.path.join(DATA_DIR, 'X.npy')
Y_PATH = os.path.join(DATA_DIR, 'y.npy')
SFREQ = 250
CH_NAMES = ['Cz', 'FCz', 'P3', 'Pz', 'C3', 'C4', 'O1', 'P4']
CLASS_MAP = {1: 'Left Hand', 2: 'Right Hand', 3: 'Feet', 10: 'Rest'}

def load_data():
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        print(f"Erreur: Les fichiers {X_PATH} ou {Y_PATH} n'existent pas.")
        print("Veuillez lancer 'preprocess_eeg.py' d'abord.")
        return None, None
    
    print(f"Chargement de {X_PATH} et {Y_PATH}...")
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    return X, y

def plot_class_distribution(y):
    unique, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(8, 5))
    bars = plt.bar([str(u) for u in unique], counts, color=['#3498db', '#e74c3c', '#2ecc71', '#95a5a6'])
    
    # Add labels
    for bar, label, count in zip(bars, unique, counts):
        height = bar.get_height()
        name = CLASS_MAP.get(label, str(label))
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({name})',
                ha='center', va='bottom')
                
    plt.title('Distribution des classes (y)')
    plt.xlabel('Classe')
    plt.ylabel("Nombre d'essais")
    plt.grid(axis='y', alpha=0.3)
    plt.show()

def plot_average_signals(X, y):
    """Affiche la moyenne temporelle pour C3 et C4 par classe"""
    classes = np.unique(y)
    
    # Indices pour C3 et C4
    if 'C3' in CH_NAMES and 'C4' in CH_NAMES:
        c3_idx = CH_NAMES.index('C3')
        c4_idx = CH_NAMES.index('C4')
    else:
        print("Canaux C3/C4 non trouvés, utilisation des deux premiers canaux.")
        c3_idx, c4_idx = 0, 1

    time_axis = np.arange(X.shape[2]) / SFREQ
    
    fig, axes = plt.subplots(len(classes), 2, figsize=(12, 3*len(classes)), sharex=True, sharey=True)
    if len(classes) == 1: axes = axes.reshape(1, -1)
    
    fig.suptitle('Moyenne des signaux (Time Domain Average - ERP) par Classe', fontsize=14)
    
    for i, cls in enumerate(classes):
        # Indices des essais pour cette classe
        idx = np.where(y == cls)[0]
        label_name = CLASS_MAP.get(cls, f"Class {cls}")
        
        # Moyenne sur les essais
        mean_signal_c3 = np.mean(X[idx, c3_idx, :], axis=0)
        mean_signal_c4 = np.mean(X[idx, c4_idx, :], axis=0)
        
        # Plot C3
        axes[i, 0].plot(time_axis, mean_signal_c3, color='#3498db')
        axes[i, 0].set_title(f'{label_name} - C3 (Right Hand Area)')
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_ylabel('Amplitude (V)')
        
        # Plot C4
        axes[i, 1].plot(time_axis, mean_signal_c4, color='#e74c3c')
        axes[i, 1].set_title(f'{label_name} - C4 (Left Hand Area)')
        axes[i, 1].grid(True, alpha=0.3)
    
    axes[-1, 0].set_xlabel('Temps (s)')
    axes[-1, 1].set_xlabel('Temps (s)')
    plt.tight_layout()
    plt.show()

def plot_psd(X, y):
    """Affiche la densité spectrale de puissance (PSD) moyenne"""
    classes = np.unique(y)
    
    c3_idx = CH_NAMES.index('C3')
    c4_idx = CH_NAMES.index('C4')
    
    plt.figure(figsize=(12, 6))
    
    for cls in classes:
        idx = np.where(y == cls)[0]
        label_name = CLASS_MAP.get(cls, f"Class {cls}")
        
        # Concaténer tous les essais pour faire un grand calcul de Welch ou faire la moyenne des Welch
        # Ici on fait la moyenne des Welch des essais individuels
        freqs, psds = welch(X[idx, c3_idx, :], fs=SFREQ, nperseg=SFREQ) # 1 sec window
        mean_psd = np.mean(psds, axis=0)
        
        plt.semilogy(freqs, mean_psd, label=f'{label_name} - C3')
        
    plt.title('PSD Moyenne sur C3 (Zone Main Droite)')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('PSD (V**2/Hz)')
    plt.xlim(0, 40) # Focus sur 0-40 Hz (Mu/Beta)
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.show()

    plt.figure(figsize=(12, 6))
    for cls in classes:
        idx = np.where(y == cls)[0]
        label_name = CLASS_MAP.get(cls, f"Class {cls}")
        
        freqs, psds = welch(X[idx, c4_idx, :], fs=SFREQ, nperseg=SFREQ)
        mean_psd = np.mean(psds, axis=0)
        
        plt.semilogy(freqs, mean_psd, label=f'{label_name} - C4')
        
    plt.title('PSD Moyenne sur C4 (Zone Main Gauche)')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('PSD (V**2/Hz)')
    plt.xlim(0, 40)
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.show()

def main():
    X, y = load_data()
    if X is None:
        return

    print("\n--- Infos Données ---")
    print(f"Forme de X (Essais, Canaux, Temps): {X.shape}")
    print(f"Forme de y (Labels): {y.shape}")
    print(f"Canaux: {CH_NAMES}")
    
    print("\nVisualisation des graphiques...")
    plot_class_distribution(y)
    plot_average_signals(X, y)
    plot_psd(X, y)

if __name__ == "__main__":
    main()

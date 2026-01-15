import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import glob

# Trouver le fichier le plus recent
def get_latest_session():
    files = glob.glob("EEG_Session_*.csv")
    if not files: return None
    return max(files, key=os.path.getctime)

FILENAME = get_latest_session()

if not FILENAME:
    print("❌ Aucun fichier de session trouvé !")
    exit()

# 1. Chargement & Nettoyage
print(f"Analyse de {FILENAME}...")
try:
    df = pd.read_csv(FILENAME, sep='\t', header=None)
except:
    df = pd.read_csv(FILENAME, sep=',', header=None)

# Convertir en Volts (Si > 100, c'est surement des uV, typiquement OpenBCI en uV sort ~45000)
# Ajustement robuste
raw_values = df.iloc[:, 1:9].values.T
if np.mean(np.abs(raw_values)) > 1.0:
    print("Détection uV -> Conversion en V")
    eeg = raw_values * 1e-6
else:
    eeg = raw_values

markers = df.iloc[:, 23].values

# Création MNE - Channels Utilisateur
ch_names = ['FC3', 'FC4', 'CP3', 'Cz', 'C3', 'C4', 'Pz', 'CP4']
info = mne.create_info(ch_names, 250, 'eeg')
raw = mne.io.RawArray(eeg, info)

# 2. Filtrage "Violent" pour ne garder que le cerveau
# On enlève le 50Hz et on garde 8-30Hz (Ondes motrices)
raw.notch_filter(50.0, verbose=False) 
raw.filter(8., 30., fir_design='firwin', verbose=False)

# 3. Découpage
diff_markers = np.diff(markers, prepend=0)
idx = np.where(np.isin(markers, [1, 2, 3]) & (diff_markers != 0))[0]
vals = markers[idx].astype(int)
events = np.column_stack((idx, np.zeros_like(idx), vals))

# Epoching
# G=1, D=2, P=3
epochs = mne.Epochs(raw, events, {'G':1, 'D':2, 'P':3}, tmin=-0.5, tmax=2.5, 
                    baseline=(-0.5, 0), verbose=False)

# 4. LE VERDICT VISUEL : C3 vs C4
# Indices: FC3(0), FC4(1), CP3(2), Cz(3), C3(4), C4(5), Pz(6), CP4(7)
idx_c3 = 4
idx_c4 = 5

if 'G' in epochs.event_id and 'D' in epochs.event_id:
    evoked_left = epochs['G'].average()
    evoked_right = epochs['D'].average()

    # On trace C3 et C4
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # C3 (Contralatéral à Droite)
    ax[0].set_title(f"Activité sur {ch_names[idx_c3]} (C3) - Doit baisser pour Main DROITE")
    ax[0].plot(evoked_left.times, evoked_left.data[idx_c3], label='Pense GAUCHE', color='blue')
    ax[0].plot(evoked_right.times, evoked_right.data[idx_c3], label='Pense DROITE', color='red', linestyle='--')
    ax[0].legend()
    ax[0].set_ylabel("Amplitude (Volts)")
    ax[0].grid(True)

    # C4 (Contralatéral à Gauche)
    ax[1].set_title(f"Activité sur {ch_names[idx_c4]} (C4) - Doit baisser pour Main GAUCHE")
    ax[1].plot(evoked_left.times, evoked_left.data[idx_c4], label='Pense GAUCHE', color='blue')
    ax[1].plot(evoked_right.times, evoked_right.data[idx_c4], label='Pense DROITE', color='red', linestyle='--')
    ax[1].legend()
    ax[1].set_xlabel("Temps (s)")
    ax[1].grid(True)

    plt.tight_layout()
    out_file = "erp_analysis_judge.png"
    plt.savefig(out_file)
    print(f"✅ Verdict visuel disponible : {out_file}")
    
    # 5. Export des données brutes pour analyse LLM
    export_df = pd.DataFrame({
        'Time_s': evoked_left.times,
        'C3_Left_V': evoked_left.data[idx_c3],
        'C3_Right_V': evoked_right.data[idx_c3],
        'C4_Left_V': evoked_left.data[idx_c4],
        'C4_Right_V': evoked_right.data[idx_c4]
    })
    
    csv_out = "erp_data_export.csv"
    export_df.to_csv(csv_out, index=False)
    print(f"📄 Données brutes exportées : {csv_out}")
    
    plt.show() # Tente d'afficher si l'environnement le permet
else:
    print("❌ Pas assez d'essais Gauche/Droite pour faire une moyenne.")

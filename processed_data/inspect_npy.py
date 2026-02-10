import numpy as np
import matplotlib.pyplot as plt

# REMPLACEZ PAR VOTRE FICHIER
FILENAME = "processed_data/X.npy" 

try:
    # 1. Chargement
    data = np.load(FILENAME, allow_pickle=True)
    
    print(f"=== INSPECTION DE {FILENAME} ===")
    
    # 2. Structure
    print(f"Type : {type(data)}")
    if isinstance(data, np.ndarray):
        print(f"Forme (Shape) : {data.shape}")  # Très important !
        print(f"Type de données : {data.dtype}")
        
        # 3. Statistiques (Pour vérifier si c'est des Volts ou Microvolts)
        print(f"Min : {np.min(data):.5f}")
        print(f"Max : {np.max(data):.5f}")
        print(f"Moyenne : {np.mean(data):.5f}")
        
        # 4. Aperçu du contenu brut
        print("\n--- Aperçu des données ---")
        print(data)

        # 5. BONUS : Visualisation si c'est un signal
        # Si la forme ressemble à (Essais, Canaux, Temps) ou (Canaux, Temps)
        if data.ndim >= 2:
            print("\nAffichage du premier signal...")
            plt.figure(figsize=(10, 4))
            
            # On essaie de tracer la dernière dimension (le temps)
            # On prend le premier élément de tout le reste
            if data.ndim == 3: # Ex: (126, 8, 750)
                plt.plot(data[0, 0, :]) 
                plt.title(f"Premier Essai, Premier Canal (Shape: {data.shape})")
            elif data.ndim == 4: # Ex: (126, 1, 8, 750) pour EEGNet
                plt.plot(data[0, 0, 0, :])
                plt.title(f"Premier Essai, Premier Canal (Shape: {data.shape})")
            else:
                plt.plot(data[0])
                
            plt.show()

    else:
        # Si c'est un dictionnaire ou autre objet sauvé
        print("Ce n'est pas un tableau simple, c'est un objet complexe :")
        print(data)

except Exception as e:
    print(f"Erreur de lecture : {e}")
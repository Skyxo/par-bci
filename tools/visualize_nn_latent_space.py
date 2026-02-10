import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.manifold import MDS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import sys
import glob

# Add EEGnet path to allow importing the class
sys.path.append(os.path.join(os.path.dirname(__file__), "../EEGnet"))
from pretrain_eegnet import EEGNet

# --- CONFIGURATION ---
# 4 Classes: Gauche, Droite, Pieds, Repos
CLASSES = {0: 'Gauche', 1: 'Droite', 2: 'Pieds', 3: 'Repos'} # NOTE: Model uses 0-3 indices, script might need mapping check
COLORS = {0: 'blue', 1: 'red', 2: 'green', 3: 'orange'}
CH_NAMES = ['FC3', 'FC4', 'CP3', 'Cz', 'C3', 'C4', 'Pz', 'CP4']
SFREQ = 250
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "presentation_figures_2")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed_data")

# MODEL PATH (Hardcoded as requested, or flexible)
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "EEGnet/runs/finetune_2026-01-28_12-59-57/eegnet_user.pth")

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
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    
    # REMAP Y to 0-3 range if necessary (Original: 1,2,3,10)
    # The Model was trained on mapped labels, so we must map data to match
    mask = np.isin(y, [1, 2, 3, 10])
    X = X[mask]
    y = y[mask]
    
    y_new = np.zeros_like(y)
    y_new[y == 1] = 0 # Left
    y_new[y == 2] = 1 # Right
    y_new[y == 3] = 2 # Feet
    y_new[y == 10] = 3 # Rest
    
    return X, y_new

def extract_features(model, x):
    """
    Manually run the forward pass up to the Flatten layer.
    Excludes the final 'dense' classification layer.
    """
    x = model.conv1(x)
    x = model.batchnorm1(x)
    x = model.conv2(x)
    x = model.batchnorm2(x)
    x = model.elu(x)
    x = model.avgpool1(x)
    x = model.dropout1(x)
    x = model.conv3_depth(x)
    x = model.conv3_point(x)
    x = model.batchnorm3(x)
    x = model.elu(x)
    x = model.avgpool2(x)
    x = model.dropout2(x)
    x = model.flatten(x)
    # STOP HERE (Do not run self.dense)
    return x

def main():
    print("🧠 Visualisation de l'Espace Latent du Réseau de Neurones...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    X_numpy, y = load_data()
    if X_numpy is None:
        print("❌ Erreur chargement données.")
        return

    # Trim to 750 samples if needed (Standard for this model)
    if X_numpy.shape[2] > 750:
         X_numpy = X_numpy[:, :, :750]
    
    # Reshape for PyTorch (N, 1, 8, 750)
    X_numpy = X_numpy[:, np.newaxis, :, :]
    
    # Convert to Tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.Tensor(X_numpy).to(device)
    
    # 2. Load Model
    print(f"📥 Chargement du modèle : {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Modèle introuvable ! Vérifiez le chemin : {MODEL_PATH}")
        return

    model = EEGNet(nb_classes=4, Chans=8, Samples=750).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # 3. Extract Features
    print("⚙️ Extraction des features (Forward pass partiel)...")
    batch_size = 32
    features_list = []
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch_x = X_tensor[i:i+batch_size]
            feats = extract_features(model, batch_x)
            features_list.append(feats.cpu().numpy())
            
    features = np.concatenate(features_list)
    print(f"   Shape Features: {features.shape} (N_epochs, N_features)")
    
    # 3b. QUANTIFY PERFORMANCE (Logistic Regression on Features)
    print("📊 Calcul de la Précision sur les Features (Validation Croisée)...")
    clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
    scores = cross_val_score(clf, features, y, cv=5)
    mean_acc = scores.mean() * 100
    std_acc = scores.std() * 100
    print(f"   ✅ Précision EEGNet (Features) : {mean_acc:.1f}% (+/- {std_acc:.1f}%)")
    
    # Generate Confusion Matrix (Normalized)
    y_pred = cross_val_predict(clf, features, y, cv=5)
    cm = confusion_matrix(y, y_pred, normalize='true')
    
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[CLASSES[i] for i in range(4)])
    disp.plot(cmap='Blues', ax=ax_cm, values_format='.2f')
    ax_cm.set_title(f"Matrice de Confusion EEGNet (Normalisée)\nAcc: {mean_acc:.1f}%")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "slide_8_eegnet_confusion_matrix.png"))
    plt.close()
    print("   -> Matrice de confusion sauvegardée : slide_8_eegnet_confusion_matrix.png")
    
    # 4. MDS 3D (Dimensionality Reduction)
    print("📉 Réduction de dimension (MDS 3D) sur les features...")
    # Using Euclidean distance in the Feature Space
    mds = MDS(n_components=3, random_state=42, n_init=2, max_iter=100)
    X_3d = mds.fit_transform(features)
    
    # 5. Plotting (3 Views)
    views = [
        (30, -60, "Vue Standard"),
        (90, -90, "Vue de Haut (2D)"),
        (10, 45, "Vue de Côté")
    ]
    
    for v_idx, (elev, azim, v_name) in enumerate(views):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for c in CLASSES:
            mask_c = (y == c)
            # Scatter Points (Opaque alpha=1)
            ax.scatter(X_3d[mask_c, 0], X_3d[mask_c, 1], X_3d[mask_c, 2],
                        c=COLORS[c], s=20, alpha=1.0, label=CLASSES[c] if v_idx==0 else "")
            
            # Centroid
            centroid = np.mean(X_3d[mask_c], axis=0)
            ax.scatter(centroid[0], centroid[1], centroid[2],
                        c=COLORS[c], s=400, marker='X', 
                        edgecolors='black', linewidth=2, label=f"Moyenne {CLASSES[c]}" if v_idx==0 else "")
            
            ax.text(centroid[0], centroid[1], centroid[2], CLASSES[c], fontsize=12, fontweight='bold', color='black')

        # Global Center
        global_center = np.mean(X_3d, axis=0)
        ax.scatter(global_center[0], global_center[1], global_center[2], c='black', s=200, marker='o')

        ax.set_title(f"ESPACE LATENT DU RÉSEAU DE NEURONES (EEGNet)\n{v_name} | Acc: {mean_acc:.1f}%", fontsize=14, fontweight='bold')
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('Latent Dim 1')
        ax.set_ylabel('Latent Dim 2')
        ax.set_zlabel('Latent Dim 3')
        
        if v_idx == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)
        
        filename = f"slide_7_nn_latent_space_view{v_idx+1}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        plt.close()
        print(f"      -> Sauvegardé {filename}")
        
    print("✅ Figures Latent Space Terminées.")

if __name__ == "__main__":
    main()

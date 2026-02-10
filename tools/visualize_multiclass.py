import numpy as np
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.distance import distance_riemann
import os
import glob

# --- CONFIGURATION ---
CLASSES = {1: 'Gauche', 2: 'Droite', 3: 'Pieds'}
COLORS = {1: 'blue', 2: 'red', 3: 'green'}
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed_data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "presentation_figures")
SFREQ = 250

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

def barycentric_coords(distances):
    """
    Convert 3 distances to barycentric coordinates using Softmax.
    This 'zooms in' on the differences.
    """
    dists = np.array(distances)
    # Beta controls the spread. Higher = more push to corners.
    # Since Riemannian distances can be ~10-20, we normalize relative to mean
    dists_centered = dists - np.mean(dists) 
    
    # We want MIN distance to have MAX weight -> Negative sign
    # Scale factor to sharpen the view
    beta = 10.0 
    
    # Softmax
    exps = np.exp(-beta * dists_centered)
    weights = exps / np.sum(exps)
    
    return weights

def plot_ternary(ax, weights_list, labels_list, accuracy):
    """
    Plots points on a ternary triangle.
    Top: Class 3 (Pieds)
    Bottom Left: Class 1 (Gauche)
    Bottom Right: Class 2 (Droite)
    """
    # Vertices of the triangle
    # A = (0, 1)    -> Pieds (Idx 2 in weights)
    # B = (-0.866, -0.5) -> Gauche (Idx 0 in weights)
    # C = (0.866, -0.5)  -> Droite (Idx 1 in weights)
    
    A = np.array([0, 1])
    B = np.array([-0.866, -0.5])
    C = np.array([0.866, -0.5])
    
    # Plot Triangle
    ax.plot([A[0], B[0]], [A[1], B[1]], 'k-', lw=1, alpha=0.3)
    ax.plot([B[0], C[0]], [B[1], C[1]], 'k-', lw=1, alpha=0.3)
    ax.plot([C[0], A[0]], [C[1], A[1]], 'k-', lw=1, alpha=0.3)
    
    # Labels for vertices
    ax.text(A[0], A[1]+0.1, "PIEDS", ha='center', fontweight='bold', color='green', fontsize=12)
    ax.text(B[0]-0.1, B[1]-0.1, "GAUCHE", ha='center', fontweight='bold', color='blue', fontsize=12)
    ax.text(C[0]+0.1, C[1]-0.1, "DROITE", ha='center', fontweight='bold', color='red', fontsize=12)
    
    # Project points
    # P = w_G * B + w_D * C + w_P * A
    for w, label in zip(weights_list, labels_list):
        # w is [w_G, w_D, w_P]
        pos = w[0] * B + w[1] * C + w[2] * A
        
        color = COLORS[label]
        # Decide marker shape based on correctness? No, just scatter is fine.
        ax.scatter(pos[0], pos[1], c=color, alpha=0.6, edgecolors='w', s=50)
        
    ax.axis('off')
    ax.set_title(f"Visualisation Ternaire (Simplex)\nPrécision MDM Multi-Classe : {accuracy:.1%}", fontsize=14)
    # Center slightly
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.8, 1.3)

def main():
    print("🔺 Visualisation Multi-Classe (Ternaire)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    X, y = load_data()
    mask = np.isin(y, [1, 2, 3])
    X = X[mask]
    y = y[mask]
    
    # 2. Filter Mu & Crop
    X_filt = mne.filter.filter_data(X, SFREQ, 8, 13, verbose=False)
    X_crop = X_filt[:, :, int(1.0*SFREQ):int(3.5*SFREQ)]
    
    # 3. Covariances
    cov_est = Covariances(estimator='lwf')
    covs = cov_est.fit_transform(X_crop)
    
    # 4. Means
    mean_G = mean_covariance(covs[y==1], metric='riemann')
    mean_D = mean_covariance(covs[y==2], metric='riemann')
    mean_P = mean_covariance(covs[y==3], metric='riemann')
    
    # 5. Calculate Distances and Weights
    weights_list = []
    correct = 0
    total = len(y)
    
    for i in range(len(covs)):
        C_i = covs[i]
        d_G = distance_riemann(C_i, mean_G)
        d_D = distance_riemann(C_i, mean_D)
        d_P = distance_riemann(C_i, mean_P)
        
        # Classification prediction (MDM)
        dists = [d_G, d_D, d_P]
        pred_idx = np.argmin(dists) # 0=G, 1=D, 2=P
        pred_label = pred_idx + 1   # 1, 2, 3
        
        if pred_label == y[i]:
            correct += 1
            
        # Barycentric weights
        w = barycentric_coords([d_G, d_D, d_P])
        weights_list.append(w)
        
    acc = correct / total
    
    # 6. Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_ternary(ax, weights_list, y, acc)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Vrai: Gauche', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Vrai: Droite', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Vrai: Pieds', markersize=10)
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    save_path = os.path.join(OUTPUT_DIR, "slide_5_multiclass_simplex.png")
    plt.savefig(save_path)
    print(f"✅ Image générée : {save_path}")

if __name__ == "__main__":
    main()

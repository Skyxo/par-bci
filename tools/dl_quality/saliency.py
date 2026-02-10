import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

def butter_bandstop_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    y = lfilter(b, a, data, axis=-1)
    return y

def compute_frequency_saliency(model, X, y, device='cpu'):
    """
    Compute importance of frequency bands by masking them (Band-Stop)
    and measuring accuracy drop.
    
    Bands:
    - Delta/Theta (1-8 Hz)
    - Mu/Alpha (8-13 Hz)
    - Beta (13-30 Hz)
    - Gamma/EMG (30-100 Hz)
    """
    model.eval()
    X = X.to(device) # (N, 1, Ch, T)
    y = y.to(device)
    
    # Baseline Accuracy
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)
        baseline_acc = (predicted == y).sum().item() / y.size(0)
        
    bands = {
        'Delta_Theta (1-8Hz)': (1, 8),
        'Mu_Alpha (8-13Hz)': (8, 13),
        'Beta (13-30Hz)': (13, 30),
        'Gamma_EMG (30-100Hz)': (30, 99) # Assuming 250Hz -> Nyquist 125
    }
    
    importance = {}
    
    X_cpu = X.cpu().numpy() # (N, Ch, T)
    
    for name, (low, high) in bands.items():
        # Perturb Data
        X_perturbed_np = butter_bandstop_filter(X_cpu, low, high, fs=250)
        X_perturbed = torch.from_numpy(X_perturbed_np).float().to(device)
        
        with torch.no_grad():
            outputs = model(X_perturbed)
            _, predicted = torch.max(outputs.data, 1)
            acc = (predicted == y).sum().item() / y.size(0)
            
        drop = baseline_acc - acc
        importance[name] = drop
        
    return importance, baseline_acc

def plot_saliency(importance_dict, plot_path=None):
    names = list(importance_dict.keys())
    values = list(importance_dict.values())
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names, values, color='salmon')
    ax.axvline(0, color='black', linestyle='--')
    ax.set_title('Feature Importance (Accuracy Drop when Masked)')
    ax.set_xlabel('Accuracy Drop')
    
    if plot_path:
        plt.savefig(plot_path)
    plt.close()

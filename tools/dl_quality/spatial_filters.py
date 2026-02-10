import torch
import numpy as np
import matplotlib.pyplot as plt
import mne
from models import EEGNetv4

def plot_spatial_filters(model_path, plot_path=None):
    """
    Visualize spatial filters from Conv2d layer of EEGNet.
    
    Args:
        model_path (str): Path to .pth model file.
        plot_path (str): Path to save the plot.
    """
    # Load Model
    # We need to know the architecture parameters to load state dict
    # Assume default parameters or load from args if saved (not saved in this simple script)
    # Default: 8 chans, 1000 times (doesn't affect weights loading if shapes match)
    model = EEGNetv4(n_classes=3, n_channels=8, n_times=1000)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Extract Weights
    # Shape: (F1*D, 8)
    weights = model.get_spatial_weights() 
    n_filters = weights.shape[0]
    
    # Setup MNE Info for Topomaps
    ch_names = ['FC3', 'FC4', 'CP3', 'Cz', 'C3', 'C4', 'Pz', 'CP4']
    info = mne.create_info(ch_names, 250, 'eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)
    
    # Plot
    fig, axes = plt.subplots(1, n_filters, figsize=(n_filters * 2, 3))
    
    if n_filters == 1:
        axes = [axes]
        
    for i in range(n_filters):
        im, _ = mne.viz.plot_topomap(weights[i], info, axes=axes[i], show=False, contours=0)
        axes[i].set_title(f'Filter {i+1}')
        
    plt.tight_layout()
    if plot_path:
        plt.savefig(plot_path)
        print(f"Spatial filters saved to {plot_path}")
    plt.close()

if __name__ == "__main__":
    # Example usage
    # plot_spatial_filters('models/eegnet_bnci_pretrained.pth', 'spatial_filters.png')
    pass

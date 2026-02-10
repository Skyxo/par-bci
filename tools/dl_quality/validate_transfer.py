import torch
import os
import sys
import numpy as np
from torch.utils.data import DataLoader

# Add tools to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dl_quality.models import EEGNetv4
from dl_quality.data_loader import load_openbci_data, EEGDataset
from dl_quality.spatial_filters import plot_spatial_filters
from dl_quality.saliency import compute_frequency_saliency, plot_saliency

def validate_transfer():
    # Paths
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, 'saved_models', 'eegnet_bnci_pretrained.pth')
    data_dir = r"C:\Users\charl\Desktop\par-bci\data_markiv"
    output_dir = os.path.join(r"C:\Users\charl\Desktop\par-bci", 'dl_quality_report')
    os.makedirs(output_dir, exist_ok=True)
    
    file_paths = [
        os.path.join(data_dir, "EEG_Session_2026-02-05_18-02.csv"),
        os.path.join(data_dir, "EEG_Session_2026-02-03_19-21.csv")
    ]
    
    # 1. Load User Data
    print("Loading User Data for Zero-Shot Validation...")
    X, y = load_openbci_data(file_paths)
    if X is None:
        print("No user data found!")
        return
        
    dataset = EEGDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 2. Load Pre-trained Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Model from {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run train_bnci.py first.")
        return
        
    # Check time dimension (BNCI training size might differ from local epoching)
    # BNCI loader uses tmax=4.0 -> 1000 samples @ 250Hz. Reference training uses this.
    # User data: tmax=4.0 -> 1000 samples. Should match.
    n_times = X.shape[2]
    model = EEGNetv4(n_classes=3, n_channels=8, n_times=n_times).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 3. Evaluate (Zero-Shot)
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Remove duplicates in labels for class names
    classes = ['Left', 'Right', 'Feet']
    
    acc = 100 * correct / total
    print(f"Zero-Shot Accuracy on OpenBCI Data: {acc:.2f}% (Chance: 33.3%)")
    
    # 4. Spatial Filters Visualization
    print("Generating Spatial Filters Plot...")
    plot_spatial_filters(model_path, os.path.join(output_dir, 'spatial_filters.png'))
    
    # 5. Saliency Analysis
    print("Computing Saliency (Frequency Sensitivity)...")
    # Use full batch for robust stats
    X_full = torch.from_numpy(dataset.X).float() # (N, Ch, T)
    # Re-apply transform manually matching Dataset
    # But dataset.X is not standardized. dataset.__getitem__ does it.
    # We should use the dataloader inputs or standardize manually.
    # Let's standardize manually Z-score per epoch
    mean = X_full.mean(axis=2, keepdims=True)
    std = X_full.std(axis=2, keepdims=True)
    X_full = (X_full - mean) / (std + 1e-4) # (N, Ch, T)
    
    import_dict, baseline_acc = compute_frequency_saliency(model, X_full, torch.from_numpy(dataset.y).long(), device)
    plot_saliency(import_dict, os.path.join(output_dir, 'saliency_map.png'))
    
    # 6. Generate Report
    report_content = f"""# Deep Learning Quality Report

## Zero-Shot Transfer Performance
- **Model**: EEGNet v4 (Pre-trained on BNCI2014001)
- **Test Data**: {len(X)} epochs from OpenBCI sessions.
- **Accuracy**: **{acc:.2f}%** (Chance Level: 33.3%)
- **Interpretation**:
    - **< 35%**: Random guessing. Likely noise or domain shift (electrode placement mismatch).
    - **35% - 45%**: Weak transfer. Signal contains some motor features.
    - **> 45%**: Strong transfer. High quality, physiological signal.

## Physiological Plausibility
### Spatial Filters (Sensory Motor Cortex)
![Spatial Filters](spatial_filters.png)
*Check if filters focus on C3, C4, Cz (Central Electrodes) or Fp/O (Artifacts).*

### Frequency Saliency (Band Importance)
![Saliency](saliency_map.png)
*Check if Mu/Alpha (8-13Hz) and Beta (13-30Hz) drops cause accuracy loss. High frequency importance usually means EMG noise.*
"""
    
    with open(os.path.join(output_dir, "DL_Quality_Report.md"), "w") as f:
        f.write(report_content)
    print(f"Report generated at {os.path.join(output_dir, 'DL_Quality_Report.md')}")

if __name__ == "__main__":
    validate_transfer()

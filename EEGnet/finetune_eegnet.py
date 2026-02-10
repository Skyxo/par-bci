import numpy as np
import pandas as pd
import mne
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import os
import glob
import sys
import datetime
import matplotlib.pyplot as plt

# Import EEGNet architecture
sys.path.append(os.path.dirname(__file__))
from pretrain_eegnet import EEGNet

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_SOURCE = "Physionet 4-Class (Left, Right, Feet, Rest)"

# PRE-TRAINED MODEL CONFIGURATION
# ⚠️ UPDATE THIS PATH TO YOUR SPECIFIC PRE-TRAINING RUN ⚠️
# Example: os.path.join(BASE_DIR, "EEGnet", "runs", "pretrain_2026-01-28_12-00-00", "eegnet_physionet_weights.pth")
WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "eegnet_physionet_weights.pth") # Default fallback
SFREQ = 250
CH_NAMES = ['FC3', 'FC4', 'CP3', 'Cz', 'C3', 'C4', 'Pz', 'CP4']


# ------------------------------------------------------------------
# DATA LOADING: Use Preprocessed .npy files (Same as Riemannian)
# ------------------------------------------------------------------
def load_all_sessions():
    print("📥 Loading Preprocessed Data from 'processed_data'...")
    try:
        # Glob ALL processed files
        X_files = glob.glob(os.path.join(BASE_DIR, "processed_data", "X_*.npy"))
        if not X_files:
            print("❌ No .npy files found.")
            return None, None
            
        X_list = []
        y_list = []
        
        for f in X_files:
            # Find matching Y
            base = os.path.basename(f).replace("X_", "y_")
            y_f = os.path.join(BASE_DIR, "processed_data", base)
            
            if os.path.exists(y_f):
                print(f"   -> Loading {os.path.basename(f)}...")
                X_part = np.load(f) # (N, 8, T)
                y_part = np.load(y_f) # (N,)
                
                # Check for NaNs
                if np.isnan(X_part).any():
                    print("     ⚠️ NaN detected, replacing with 0")
                    X_part = np.nan_to_num(X_part)
                
                X_list.append(X_part)
                y_list.append(y_part)
        
        if not X_list: return None, None
        
        X = np.concatenate(X_list)
        y = np.concatenate(y_list)

        # MAPPINGS: 1(L), 2(R), 3(F), 10(Rest)
        # We filter to keep only ACTIVE classes (1, 2, 3) + Rest (10)
        
        mask_valid = (y == 1) | (y == 2) | (y == 3) | (y == 10)
        X = X[mask_valid]
        y = y[mask_valid]
        
        y_new = np.zeros_like(y)
        y_new[y == 1] = 0 # Left
        y_new[y == 2] = 1 # Right
        y_new[y == 3] = 2 # Feet
        y_new[y == 10] = 3 # Rest (NEW)
        y = y_new
        
        # Trim time dimension to match Fixed Input Size (750)
        # Assuming X is (N, 8, T)
        if X.shape[2] > 750:
            X = X[:, :, :750]
        elif X.shape[2] < 750:
             # Pad with zeros if too short
             pad_len = 750 - X.shape[2]
             X = np.pad(X, ((0,0), (0,0), (0, pad_len)), mode='constant')

        return X, y
    except Exception as e:
        print(f"❌ Error loading .npy files: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        
    avg_loss = total_loss / len(loader)
    acc = 100 * correct / total
    return avg_loss, acc

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    acc = 100 * correct / total if total > 0 else 0
    return avg_loss, acc

def main():
    print("=== EEGNet FINE-TUNING (Transfer Learning) ===")
    
    # 1. Load User Data (ALL SESSIONS)
    X, y = load_all_sessions()
    
    # OUTPUT DIRS
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(os.path.dirname(__file__), "runs", f"finetune_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"📂 Output Directory: {run_dir}")
    
    if X is None or len(X) == 0:
        print("❌ No valid epochs found.")
        return

    # Debug Scaling
    mean_val = np.mean(np.abs(X))
    if mean_val > 1.0: # Detect uV
         print("   ⚠️ Data > 1.0 (uV likely), Converting to Volts...")
         X = X * 1e-6

    # Reshape for PyTorch: (N, 1, Chans, Time)
    X = X[:, np.newaxis, :, :]
    
    # Data Validation
    unique, counts = np.unique(y, return_counts=True)
    print(f"✅ Data Ready: {X.shape} samples. Classes: {unique} (Counts: {counts})")
    
    # COMPUTE CLASS WEIGHTS
    class_weights = compute_class_weight('balanced', classes=unique, y=y)
    class_weights_tensor = torch.Tensor(class_weights)
    print(f"⚖️ Class Weights: {class_weights}")
    
    # 2. Load Pre-Trained Model (Target: 4 Classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}")
    
    model = EEGNet(nb_classes=4, Chans=8, Samples=750).to(device) 
    
    # Check if weights file exists
    if os.path.exists(WEIGHTS_FILE):
        print(f"✅ Loading Pre-trained Weights from: {WEIGHTS_FILE}")
        try:
            state = torch.load(WEIGHTS_FILE)
            model_state = model.state_dict()
            pretrained_dict = {k: v for k, v in state.items() if k in model_state and v.size() == model_state[k].size()}
            model.load_state_dict(pretrained_dict, strict=False)
            
            if 'fc.bias' in pretrained_dict:
                print(f"   ✅ Perfect Match! Loaded 4-class Pre-trained Weights.")
            else:
                print(f"   ⚠️ Class mismatch. Loaded feature extractor only. Final layer reset.")
            
        except RuntimeError as e:
            print(f"⚠️ Error loading weights: {e}")
            print("Training from scratch.")
    else:
        print("⚠️ No pre-trained weights found. Training from scratch.")

    # 4. Train/Val Split (Stratified)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_dataset = TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.Tensor(X_val), torch.LongTensor(y_val))
    
    # INCREASED BATCH SIZE
    BATCH_SIZE = 16
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    
    # HISTORY TRACKING
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    best_val_bal_acc = 0.0
    best_epoch = -1
    best_model_path = os.path.join(MODELS_DIR, "eegnet_best.pth")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # ------------------------------------------------------------------
    # PHASE 1: FROZEN FEATURE EXTRACTOR (Warmup)
    # ------------------------------------------------------------------
    print("\n❄️ PHASE 1: FREEZING FEATURES (Training Classifier Only)...")
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze only the classification layer (dense)
    for param in model.dense.parameters():
        param.requires_grad = True
        
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    WARMUP_EPOCHS = 20
    for epoch in range(WARMUP_EPOCHS):
        t_loss, t_acc = train_epoch(model, loader, criterion, optimizer, device)
        v_loss, v_acc = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)
        
        print(f"[Warmup {epoch+1}/{WARMUP_EPOCHS}] Loss: {t_loss:.3f} | Acc: {t_acc:.1f}% | Val Loss: {v_loss:.3f} | Val Acc: {v_acc:.1f}%")

    # ------------------------------------------------------------------
    # PHASE 2: FINE-TUNING (All Layers)
    # ------------------------------------------------------------------
    print("\n🔓 PHASE 2: UNFREEZING ALL LAYERS (Fine-Tuning)...")
    for param in model.parameters():
        param.requires_grad = True
        
    # Lower Learning Rate for Fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=0.0001) 
    
    # Scheduler: Monitor Val Loss instead of Acc
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6, verbose=True)
    
    patience = 30 # Increased patience for long training
    trigger_times = 0
    n_epochs = 1000 # Increased to 1000 as requested
    
    best_val_loss = float('inf')

    try:
        for epoch in range(n_epochs):
            t_loss, t_acc = train_epoch(model, loader, criterion, optimizer, device)
            v_loss, v_acc = evaluate(model, val_loader, criterion, device)
            
            history['train_loss'].append(t_loss)
            history['val_loss'].append(v_loss)
            history['train_acc'].append(t_acc)
            history['val_acc'].append(v_acc)
            
            # Step Scheduler on Loss
            scheduler.step(v_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1}/{n_epochs} | Loss: {t_loss:.4f} | Acc: {t_acc:.1f}% | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.1f}% | LR: {current_lr:.1e}")

            # SAVE BEST MODEL (Based on VAL LOSS - Early Stopping)
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_epoch = len(history['val_loss']) # Correct global epoch count
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> ⭐ New Best Val Loss! Saved to eegnet_best.pth")
                trigger_times = 0
            else:
                trigger_times += 1
                print(f"  -> No improvement ({trigger_times}/{patience})")

            # PLOTTING
            plt.figure(figsize=(10, 5))
            
            # Loss
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Train', color='red', alpha=0.6)
            plt.plot(history['val_loss'], label='Val', color='orange')
            if best_epoch != -1:
                 plt.plot(best_epoch-1, best_val_loss, marker='*', color='purple', markersize=15, label=f'Best: {best_val_loss:.4f}')
            plt.title(f'Loss (Best: {best_val_loss:.4f})')
            plt.legend()
            plt.grid(True)
            plt.axvline(x=WARMUP_EPOCHS, color='black', linestyle='--', alpha=0.5, label='Phase 2 Start')

            # Accuracy
            plt.subplot(1, 2, 2)
            plt.plot(history['train_acc'], label='Train', color='green', alpha=0.6)
            plt.plot(history['val_acc'], label='Val', color='blue')
            plt.title('Accuracy')
            plt.legend()
            plt.grid(True)
            plt.axvline(x=WARMUP_EPOCHS, color='black', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            save_path = os.path.join(run_dir, 'finetune_results.png')
            plt.savefig(save_path)
            plt.close()
            
            if trigger_times >= patience:
                print(f"🛑 Early stopping! No improvement for {patience} epochs.")
                break

    except KeyboardInterrupt:
        print("\n\n⚠️ INTERRUPTED BY USER (Ctrl+C)")
        print("   Saving current state before exiting...")

    # Save Fine-Tuned Model (Final state)
    user_model_path = os.path.join(run_dir, "eegnet_user.pth")
    torch.save(model.state_dict(), user_model_path)
    print(f"✅ Final model saved: {user_model_path}")
    print(f"🏆 Best model (used for replay): {best_model_path}")

if __name__ == "__main__":
    main()

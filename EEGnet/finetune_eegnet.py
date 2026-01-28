import numpy as np
import pandas as pd
import mne
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
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
WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "eegnet_physionet_weights.pth")
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
        # We filter to keep only ACTIVE classes (1, 2, 3)
        # Marker 10 (Rest) is excluded as per recent decision to focus on active tasks
        
        mask_valid = (y == 1) | (y == 2) | (y == 3)
        X = X[mask_valid]
        y = y[mask_valid]
        
        y_new = np.zeros_like(y)
        y_new[y == 1] = 0 # Left
        y_new[y == 2] = 1 # Right
        y_new[y == 3] = 2 # Feet
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
    
    print(f"✅ Data Ready: {X.shape} samples. Classes: {np.unique(y, return_counts=True)}")
    
    # 2. Load Pre-Trained Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # CHANGED TO 3 CLASSES
    model = EEGNet(nb_classes=3, Chans=8, Samples=750).to(device) 
    
    if os.path.exists(WEIGHTS_FILE):
        print("✅ Loading Physionet Weights...")
        try:
            # We must handle the mismatch of the final layer (3 classes vs 4 classes)
            state = torch.load(WEIGHTS_FILE)
            model_state = model.state_dict()
            
            # Filter out mismatching keys
            state = {k: v for k, v in state.items() if k in model_state and v.size() == model_state[k].size()}
            
            model.load_state_dict(state, strict=False)
            print("   (Partial load successful - Classification layer reset)")
        except RuntimeError:
            print("⚠️ Weights mismatch too severe. Training from scratch.")
    else:
        print("⚠️ No pre-trained weights found. Training from scratch.")

    # 3. UNFREEZE ALL LAYERS (Full Fine-Tuning)
    print("🔓 Unfreezing all layers allow full adaptation (Low LR)...")
    for param in model.parameters():
        param.requires_grad = True
    
    # 4. Train/Val Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.Tensor(X_val), torch.LongTensor(y_val))
    
    loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    # LOWER LEARNING RATE for full fine-tuning
    # LOWER LEARNING RATE for full fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # SCHEDULER: Reduce LR if Val Loss plateaus
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10, min_lr=1e-6)
    
    print("Fine-tuning on User Data (Full Network)...")
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    best_val_acc = 0.0
    best_epoch = -1
    best_model_path = os.path.join(run_dir, "eegnet_best.pth")

    best_model_path = os.path.join(MODELS_DIR, "eegnet_best.pth")

    n_epochs = 500
    for epoch in range(n_epochs):
        # TRAIN
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
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
            
        epoch_loss = total_loss/len(loader)
        epoch_acc = 100*correct/total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # VALIDATE
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
             for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss/len(val_loader) if len(val_loader) > 0 else 0
        val_epoch_acc = 100*val_correct/val_total if val_total > 0 else 0
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)
        
        # UPDATE SCHEDULER
        scheduler.step(val_epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.1f}% | Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.1f}% | LR: {current_lr:.1e}")

        # SAVE BEST MODEL (Based on ACCURACY)
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> ⭐ New Best Val Acc! Saved to eegnet_best.pth")

        # ==========================================
        # LIVE PLOTTING (Every Epoch)
        # ==========================================
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(train_losses)+1), train_losses, label='Train', marker='o', color='red', alpha=0.6)
        plt.plot(range(1, len(val_losses)+1), val_losses, label='Val', marker='x', color='orange')
        # Mark Best
        if best_epoch != -1:
            # We want to show where the best model (by Accuracy) is located on the Loss graph
            loss_at_best = val_losses[best_epoch-1]
            plt.plot(best_epoch, loss_at_best, marker='*', color='purple', markersize=15, label=f'Best Acc: Ep {best_epoch}')
        
        plt.title(f'Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(train_accs)+1), train_accs, label='Train', marker='o', color='green', alpha=0.6)
        plt.plot(range(1, len(val_accs)+1), val_accs, label='Val', marker='x', color='blue')
        # Mark Best
        if best_epoch != -1:
            plt.plot(best_epoch, best_val_acc, marker='*', color='purple', markersize=15, label=f'Best: {best_val_acc:.1f}%')
            
        plt.title(f'Accuracy (Best: {best_val_acc:.1f}%)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(run_dir, 'finetune_results.png')
        plt.savefig(save_path)
        plt.close()
        
        if False: # Early stopping removed
            print(f"� Early stopping! No improvement for {patience} epochs.")
            break

    # Save Fine-Tuned Model (Final state)
    user_model_path = os.path.join(run_dir, "eegnet_user.pth")
    torch.save(model.state_dict(), user_model_path)
    print(f"✅ Final model saved: {user_model_path}")
    print(f"🏆 Best model (used for replay): {best_model_path}")

if __name__ == "__main__":
    main()

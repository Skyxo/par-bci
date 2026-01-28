import numpy as np
import datetime
import os
import mne
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mne.datasets import eegbci
import os
import matplotlib.pyplot as plt

# Suppress MNE info messages
mne.set_log_level('WARNING')

class EEGNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=128, 
                 dropoutRate=0.5, kernLength=64, F1=8, 
                 D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNet, self).__init__()
        self.Chans = Chans
        self.Samples = Samples

        # Layer 1
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        
        # Layer 2
        self.conv2 = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropoutRate)
        
        # Layer 3
        self.conv3_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.conv3_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropoutRate)
        
        self.flatten = nn.Flatten()
        
        # Calculate output size
        # Input: (1, 8, 750)
        # Conv1: (8, 8, 750) 
        # Conv2: (16, 1, 750)
        # Pool1: (16, 1, 187)
        # Conv3: (16, 1, 187)
        # Pool2: (16, 1, 23)
        # Flatten: 16 * 23 = 368
        
        out_samples = Samples // 32
        self.dense = nn.Linear(F2 * out_samples, nb_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        x = self.conv3_depth(x)
        x = self.conv3_point(x)
        x = self.batchnorm3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def main():
    print("=== EEGNet PRE-TRAINING (Physionet 4-Class) ===")
    
    # CACHE PATH
    # DATASET (Frozen 4-Class)
    CACHE_FILE = os.path.join(os.path.dirname(__file__), "PRETRAIN_DATABASE_4CLASS.npz")
    
    # OUTPUT DIRS
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(os.path.dirname(__file__), "runs", f"pretrain_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"📂 Output Directory: {run_dir}")
    
    if os.path.exists(CACHE_FILE):
        print(f"🚀 Loading FROZEN Database from {CACHE_FILE}...")
        data = np.load(CACHE_FILE)
        X = data['X']
        y = data['y']
    else:
        print(f"❌ Database not found: {CACHE_FILE}")
        print("   Please run 'python tools/build_physionet_4class.py' first.")
        return
    
    # Validation of shape
    print(f"   Loaded Shape: {X.shape}")
    
    # Reshape for PyTorch: (Batch, 1, Channels, Time)
    X = X.reshape(X.shape[0], 1, 8, 750)
    
    print(f"Data Loaded: {X.shape} samples. Classes: {np.unique(y, return_counts=True)}")
    
    # ==========================================
    # 3. TRAINING LOOP
    # ==========================================
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    train_dataset = TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # Larger batch size
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # 4 CLASSES NOW
    model = EEGNet(nb_classes=4, Chans=8, Samples=750).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # SCHEDULER (User Request)
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10, min_lr=1e-6)
    
    n_epochs = 1000
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_bal_accs, val_bal_accs = [], []
    
    best_val_loss = float('inf')
    best_epoch = -1
    patience = 1000
    trigger_times = 0
    
    try:
        for epoch in range(n_epochs):
            # TRAIN
            model.train()
            running_loss = 0.0
            all_preds = []
            all_targets = []
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
            
            train_loss = running_loss/len(train_loader)
            train_acc = 100 * accuracy_score(all_targets, all_preds)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # VALIDATION
            model.eval()
            val_running_loss = 0.0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())
            
            val_loss = val_running_loss/len(test_loader)
            val_acc = 100 * accuracy_score(val_targets, val_preds)

            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # UPDATE SCHEDULER
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1}/{n_epochs} | Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | LR: {current_lr:.1e}")

            # Save Best Model & Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_model_path = os.path.join(run_dir, "eegnet_physionet_best.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> ⭐ New Best Val Loss! Saved to {best_model_path}")
                trigger_times = 0
            else:
                trigger_times += 1
                print(f"  -> No improvement ({trigger_times}/{patience})")
                
            # ==========================================
            # LIVE PLOTTING (Every Epoch)
            # ==========================================
            plt.figure(figsize=(10, 5))
            
            # Loss
            plt.subplot(1, 2, 1)
            plt.plot(range(1, len(train_losses)+1), train_losses, label='Train', marker='o', alpha=0.6)
            plt.plot(range(1, len(val_losses)+1), val_losses, label='Val', marker='x')
            if best_epoch != -1:
                plt.plot(best_epoch, best_val_loss, marker='*', color='purple', markersize=15, label=f'Best: {best_val_loss:.4f}')
            plt.title(f'Loss (Best: {best_val_loss:.4f})')
            plt.legend()
            plt.grid(True)
            
            # Accuracy
            plt.subplot(1, 2, 2)
            plt.plot(range(1, len(train_accs)+1), train_accs, label='Train', color='green', alpha=0.6)
            plt.plot(range(1, len(val_accs)+1), val_accs, label='Val', color='blue')
            plt.title('Accuracy')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            save_path = os.path.join(run_dir, 'pretrain_metrics.png')
            plt.savefig(save_path)
            plt.close() # Close to free memory
            
            # Check early stopping AFTER plotting so we capture the last state
            if trigger_times >= patience:
                print(f"🛑 Early stopping! No improvement for {patience} epochs.")
                break
                
    except KeyboardInterrupt:
        print("\n\n⚠️ INTERRUPTED BY USER (Ctrl+C)")
        print("   Saving current state before exiting...")
        
    # Save Final Weights (Run Directory)
    model_path = os.path.join(run_dir, "eegnet_physionet_weights.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Final Model saved to {model_path}")

    # FINAL MESSAGE
    print(f"\n✅ Pre-training Complete!")
    print(f"   Shape: {X.shape}")
    print(f"   Best Model: {best_model_path}")
    print(f"   Final Model: {model_path}")
    print(f"👉 To Finetune: Update 'WEIGHTS_FILE' in finetune_eegnet.py with the path above.")

if __name__ == "__main__":
    main()

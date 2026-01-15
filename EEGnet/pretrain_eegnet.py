import numpy as np
import mne
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
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
        self.batchnorm1 = nn.BatchNorm2d(F1, False)
        
        # Layer 2
        self.conv2 = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D, False)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropoutRate)
        
        # Layer 3
        self.conv3_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.conv3_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2, False)
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
    print("=== EEGNet PRE-TRAINING (Physionet) ===")
    
    # Use ALL subjects for maximum rigor (1 to 109)
    subjects = list(range(1, 46))
    # subjects = list(range(1, 4)) # Debug mode
    
    # User Channels (Target)
    target_channels = ['FC3', 'FC4', 'CP3', 'CZ', 'C3', 'C4', 'PZ', 'CP4']
    
    X_total = []
    y_total = []
    
    print(f"Downloading/Loading data for {len(subjects)} subjects (Real + Imaginary)...")

    # Mapping of Runs to Tasks:
    runs_hands = [3, 7, 11, 4, 8, 12] # Real + Imag
    runs_feet =  [5, 9, 13, 6, 10, 14] # Real + Imag

    for subject in subjects:
        try:
            # --- PART 1: HANDS (Left vs Right) ---
            fnames_hands = eegbci.load_data(subject, runs_hands)
            raws_hands = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames_hands]
            raw_hands = mne.concatenate_raws(raws_hands)
            
            # --- PART 2: FEET (Feet) ---
            fnames_feet = eegbci.load_data(subject, runs_feet)
            raws_feet = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in fnames_feet]
            raw_feet = mne.concatenate_raws(raws_feet)
            
            # Combine logic
            for raw_tmp, task_type in [(raw_hands, 'hands'), (raw_feet, 'feet')]:
                
                # 1. Standardize Channels
                mne.rename_channels(raw_tmp.info, lambda x: x.strip('.').upper()) 
                
                # Check availability
                available_chs = raw_tmp.ch_names
                missing = [ch for ch in target_channels if ch not in available_chs]
                if missing:
                    continue

                # 2. Select Channels
                raw_tmp.pick(target_channels)
                
                # 3. Filter & Resample
                raw_tmp.filter(2., 40., fir_design='firwin', verbose=False)
                if raw_tmp.info['sfreq'] != 250:
                    raw_tmp.resample(250, npad="auto", verbose=False)
                
                # 4. Epoching
                events, event_id_dict = mne.events_from_annotations(raw_tmp, verbose=False)
                
                # Define mapping
                current_event_id = {}
                if task_type == 'hands':
                    # T1=Left(0), T2=Right(1)
                    if 'T1' in event_id_dict: current_event_id[event_id_dict['T1']] = 0
                    if 'T2' in event_id_dict: current_event_id[event_id_dict['T2']] = 1
                else:
                    # T2=Feet(2)
                    if 'T2' in event_id_dict: current_event_id[event_id_dict['T2']] = 2
                
                if not current_event_id: continue
                
                valid_events = []
                valid_labels = []
                for ev in events:
                    code = ev[2]
                    if code in current_event_id:
                        valid_events.append(ev)
                        valid_labels.append(current_event_id[code])
                
                if not valid_events: continue
                
                valid_events = np.array(valid_events)
                
                epochs = mne.Epochs(raw_tmp, valid_events, event_id=None, tmin=0, tmax=3.0, 
                                    proj=False, baseline=None, verbose=False)
                
                data = epochs.get_data() # (N, 8, 751)
                
                # Crop to 750 (3s)
                if data.shape[2] > 750:
                    data = data[:, :, :750]
                elif data.shape[2] < 750:
                    continue # Skip incomplete
                
                X_total.append(data)
                y_total.append(np.array(valid_labels))
            
            print(f"  Subject {subject}: Processed.")

        except Exception as e:
            print(f"  ⚠️ Skipping Subject {subject}: {e}")
            continue

    if len(X_total) == 0:
        print("❌ Failed to load any data.")
        return

    X = np.concatenate(X_total)
    y = np.concatenate(y_total)
    
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
    
    model = EEGNet(nb_classes=3, Chans=8, Samples=750).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    n_epochs = 150
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = -1
    
    # Early Stopping
    patience = 20
    trigger_times = 0
    
    for epoch in range(n_epochs):
        # TRAIN
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss/len(train_loader)
        train_losses.append(train_loss)
        
        # VALIDATION
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
        
        val_loss = val_running_loss/len(test_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save Best Model & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            trigger_times = 0 # Reset
            best_model_path = os.path.join(os.path.dirname(__file__), "eegnet_physionet_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> ⭐ New Best Val Loss! Saved to eegnet_physionet_best.pth")
        else:
            trigger_times += 1
            print(f"  -> No improvement (patience: {trigger_times}/{patience})")
            
        # ==========================================
        # LIVE PLOTTING (Every Epoch)
        # ==========================================
        plt.figure()
        plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', marker='o', alpha=0.6)
        plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss', marker='x')
        
        # Mark Best
        if best_epoch != -1:
            plt.plot(best_epoch, best_val_loss, marker='*', color='purple', markersize=15, label=f'Best: Ep {best_epoch}')
            
        plt.title(f'Pre-training Loss (Best: {best_val_loss:.4f} @ Ep {best_epoch})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(os.path.dirname(__file__), 'pretrain_loss.png')
        plt.savefig(save_path)
        plt.close() # Close to free memory
        
        # Check early stopping AFTER plotting so we capture the last state
        if trigger_times >= patience:
            print(f"🛑 Early stopping! No improvement for {patience} epochs.")
            break
        
    # Save Final Weights (Optional)
    model_path = os.path.join(os.path.dirname(__file__), "eegnet_physionet_weights.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Final Model saved to {model_path}")

if __name__ == "__main__":
    main()

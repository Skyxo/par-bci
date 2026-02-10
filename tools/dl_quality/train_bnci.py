import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import load_bnci_data, EEGDataset, load_openbci_data
from models import EEGNetv4
import os
import argparse
import numpy as np

def train_bnci(epochs=20, batch_size=64, lr=0.001):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data (All subjects for robust pre-training)
    print("Loading BNCI Data (Training Set)...")
    # subject_ids can be range(1, 10).
    X_train, y_train = load_bnci_data(subject_ids=range(1, 10))
    print(f"Loaded {X_train.shape[0]} trials.")
    
    # Create Dataset
    dataset = EEGDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model Init
    # X_train shape: (N, 8, 1000)
    n_times = X_train.shape[2]
    # Classes: Left, Right, Feet (3)
    model = EEGNetv4(n_classes=3, n_channels=8, n_times=n_times).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train Loop
    print("Starting Training...")
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs: (Batch, 8, 1000)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(dataloader):.4f} | Acc: {100*correct/total:.2f}%")
        
    # Save Model
    save_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'eegnet_bnci_pretrained.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_bnci()

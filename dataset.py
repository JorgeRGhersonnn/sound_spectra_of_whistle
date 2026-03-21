#!/usr/bin/env python3
'''
Data Science Student Fellowship -- Jorge R. Gherson
Project: Visualization of Sound Spectra of a Whistle.
-----------------------------------------------------
File Description 
This file converts the raw NumPy arrays into PyTorch tensors and wrapping them in a DataLoader. 
'''

import os 
import numpy as np 
import torch 
from torch.utils.data import Dataset, DataLoader 
from sklearn.model_selection import train_test_split 
import config 

class WhistleDataset(Dataset): 
    def __init__(self, features, labels): 
        """
        Custom PyTorch dataset for whistle data. 
        """
        # Convert NumPy arrays to tensors -- unsqueeze (add extra dimension) so shape becomes (Samples, Channels, Frequencies)
        # CNNs expect a channel dimension. Here, Channel = 1.
        self.x = torch.tensor(features, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def get_dataloaders(batch_size=16, test_size=0.2, random_state=42): 
    # Loads processed data, splits it, and returns PyTorch dataloaders 
    print("Loading preprocessed data for Pytorch...")
    features_path = os.path.join(config.PROCESSED_DIR, "psd_features.npy")
    labels_path = os.path.join(config.PROCESSED_DIR, "mass_flow_labels.npy")

    X = np.load(features_path)
    y = np.load(labels_path)

    # split into trai and test sets 
    X_train, X_test, y_train, y_test = train_test_split( 
        X,
        y,
        test_size=test_size, 
        random_state=random_state
    )

    # Initialize the datasets
    train_dataset = WhistleDataset(X_train, y_train)
    test_dataset = WhistleDataset(X_test, y_test)

    # Create dataloaders 
    # note: shuffle=True prevents the model from learning the order of the data. 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print(f"Train Loader: {len(train_loader)} batches of size {batch_size}")
    print(f"Test Loader: {len(test_loader)} batches of sizze {batch_size}")

    return train_loader, test_loader

# quick test to make sure it works 
if __name__ == "__main__": 
    train_loader, test_loader = get_dataloaders()

    # fetch one batch and inspect shape
    features, labels = next(iter(train_loader))
    print(f"\nBatch Features Shape: {features.shape}")      # Should be [16, 1, 1698]
    print(f"Batch Labels Shape: {labels.shape}")            # Should be [16, 1]
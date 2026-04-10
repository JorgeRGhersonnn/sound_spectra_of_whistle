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
        Custom PyTorch dataset for whistle data (1D PSD). 
        """
        # Convert NumPy arrays to tensors -- unsqueeze (add extra dimension) so shape becomes (Samples, Channels, Frequencies)
        # CNNs expect a channel dimension. Here, Channel = 1.
        self.x = torch.tensor(features, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class WhistleDataset2D(Dataset):
    def __init__(self, features, labels):
        """
        Custom PyTorch dataset for whistle STFT data (2D spectrogram).
        Input features shape: (Samples, Freq_bins, Time_frames)
        Output shape: (Samples, Channels=1, Freq_bins, Time_frames)
        """
        # Add channel dimension for 2D CNN: (Samples, Freq, Time) -> (Samples, 1, Freq, Time)
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

def get_dataloaders_2d(batch_size=16, test_size=0.2, random_state=42):
    """
    Loads 2D STFT spectrogram data, splits it, and returns PyTorch dataloaders.
    """
    print("Loading 2D STFT spectrograms for PyTorch...")
    stft_features_path = os.path.join(config.PROCESSED_DIR, "stft_features.npy")
    labels_path = os.path.join(config.PROCESSED_DIR, "mass_flow_labels.npy")

    X = np.load(stft_features_path)
    y = np.load(labels_path)

    print(f"2D STFT shape: {X.shape}")  # Should be (Samples, Freq, Time)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    # Initialize the datasets
    train_dataset = WhistleDataset2D(X_train, y_train)
    test_dataset = WhistleDataset2D(X_test, y_test)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print(f"Train Loader: {len(train_loader)} batches of size {batch_size}")
    print(f"Test Loader: {len(test_loader)} batches of size {batch_size}")

    return train_loader, test_loader

# quick test to make sure it works 
if __name__ == "__main__": 
    print("Testing 1D PSD DataLoader...")
    train_loader, test_loader = get_dataloaders()

    # fetch one batch and inspect shape
    features, labels = next(iter(train_loader))
    print(f"1D Batch Features Shape: {features.shape}")      # Should be [16, 1, 1698]
    print(f"1D Batch Labels Shape: {labels.shape}")            # Should be [16, 1]
    
    print("\nTesting 2D STFT DataLoader...")
    train_loader_2d, test_loader_2d = get_dataloaders_2d()
    
    # fetch one batch and inspect shape
    features_2d, labels_2d = next(iter(train_loader_2d))
    print(f"2D Batch Features Shape: {features_2d.shape}")  # Should be [16, 1, Freq, Time]
    print(f"2D Batch Labels Shape: {labels_2d.shape}")      # Should be [16, 1]
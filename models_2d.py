#!/usr/bin/env python3
'''
Data Science Student Fellowship -- Jorge R. Gherson
Project: Visualization of Sound Spectra of a Whistle.
-----------------------------------------------------
File Description 
2D CNN model architecture for learning from STFT spectrograms.
Input shape: (Batch_size, Channels=1, Freq_bins, Time_frames)
'''

import torch
import torch.nn as nn


class WhistleCNN2D(nn.Module):
    """2D Convolutional Neural Network for STFT spectrogram analysis."""
    
    def __init__(self, input_freq_bins=212, input_time_frames=392):
        super(WhistleCNN2D, self).__init__()

        # -- 2D CONVOLUTIONAL FEATURE EXTRACTOR --
        # Input shape: (Batch, 1, Freq_bins, Time_frames)
        self.conv_layers = nn.Sequential(
            # Block 1: Extract basic spectrotemporal patterns
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5), 
                     stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # Block 2: Capture hierarchical features
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5),
                     stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # Block 3: More abstract patterns
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3),
                     stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # Block 4: High-level representation
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                     stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16))  # Compress to fixed size
        )

        # Calculate flattened size: 128 channels * 16 * 16
        self.fc_input_size = 128 * 16 * 16

        # -- FULLY CONNECTED REGRESSOR --
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            
            nn.Linear(64, 1)  # Single output: Mass Flow Rate
        )

    def forward(self, x):
        """
        Forward pass through the 2D CNN.
        Args:
            x: Input tensor of shape (Batch, 1, Freq, Time)
        Returns:
            output: Predicted mass flow rate (Batch, 1)
        """
        # Convolutional feature extraction
        x = self.conv_layers(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Regression head
        x = self.fc_layers(x)

        return x


class WhistleCNN1D(nn.Module):
    """
    1D Convolutional Neural Network for PSD feature analysis.
    (Kept here for parallel comparison with 2D model)
    Input shape: (Batch_size, Channels=1, Frequencies)
    """
    
    def __init__(self):
        super(WhistleCNN1D, self).__init__()

        # -- 1D CONVOLUTIONAL FEATURE EXTRACTOR --
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, 
                     stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5,
                     stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5,
                     stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # ADAPTIVE POOLING: Forces output to be exactly 32 frequency bins
            nn.AdaptiveAvgPool1d(32)
        )

        # -- FULLY CONNECTED REGRESSOR --
        # 64 channels * 32 bins = 2048 features
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 32, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            
            nn.Linear(128, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1)  # Final output: Mass Flow Rate
        )

    def forward(self, x):
        """
        Forward pass through the 1D CNN.
        Args:
            x: Input tensor of shape (Batch, 1, Frequencies)
        Returns:
            output: Predicted mass flow rate (Batch, 1)
        """
        # Convolutional feature extraction
        x = self.conv_layers(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Regression head
        x = self.fc_layers(x)

        return x


if __name__ == "__main__":
    # Test 2D model
    print("Testing 2D CNN...")
    model_2d = WhistleCNN2D()
    dummy_2d = torch.randn(16, 1, 212, 392)  # (Batch, Channels, Freq, Time)
    output_2d = model_2d(dummy_2d)
    print(f"2D CNN Input: {dummy_2d.shape} -> Output: {output_2d.shape}")
    print(f"2D CNN Parameters: {sum(p.numel() for p in model_2d.parameters()):,}")
    print()

    # Test 1D model
    print("Testing 1D CNN...")
    model_1d = WhistleCNN1D()
    dummy_1d = torch.randn(16, 1, 1698)  # (Batch, Channels, Frequencies)
    output_1d = model_1d(dummy_1d)
    print(f"1D CNN Input: {dummy_1d.shape} -> Output: {output_1d.shape}")
    print(f"1D CNN Parameters: {sum(p.numel() for p in model_1d.parameters()):,}")

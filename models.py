#!/usr/bin/env python3
'''
Data Science Student Fellowship -- Jorge R. Gherson
Project: Visualization of Sound Spectra of a Whistle.
-----------------------------------------------------
File Description 
'''

import torch 
import torch.nn as nn 

class WhistleCNN(nn.Module):
    def __init__(self): 
        super(WhistleCNN, self).__init__()

        # -- 1D CONVOLUTIONAL FEATURE EXTRACTOR -- 
        # Input shape expected: (Batch_size, Channels=1, Frequencies)
        self.conv_layers = nn.Sequential(
            # Block 1 
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=2, stride=2), 

            # Block 2
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=2, stride=2), 

            # Block 3 
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=2, stride=2), 

            # ADAPTIVE POOLING: Forces output to be exactly 32 frequency bins per channel 
            nn.AdaptiveAvgPool1d(32)
        )

        # -- FULLY CONNECTED REGRESSOR --
        # we have 64 channels * 32 bins = 2048 features 
        self.fc_layers = nn.Sequential( 
            nn.Linear(64*32, 128),
            nn.ReLU(), 
            nn.Dropout(p=0.3),      # prevent overfitting
            nn.Linear(128, 32), 
            nn.ReLU(), 
            nn.Linear(32, 1)        # Final output is a single continuous value (Mass Flow Rate)
        )
    
    def forward(self, x): 
        # Defines how the data flows through the network. 
        # 1. Pass through convolutional filters. 
        x = self.conv_layers(x)

        # 2. Flatten the 3D tensor to a 2D tensor for the Linear layers 
        # shape goes from (Batch, 64, 32) -> (Batch, 2048)
        x = x.view(x.size(0), -1)

        # 3. Pass through the fully connected regressor 
        x = self.fc_layers(x)

        return x 
    
# quick test to verify shapes
if __name__ == "__main__": 
    # create a dummy tensor that looks like one of the batches 
    dummy_input = torch.randn(16, 1, 1698)

    # Initialize model 
    model = WhistleCNN()

    # pass dummy through model 
    output = model(dummy_input)

    print(f"Model successfully built!")
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape} -> (Batch Size, Predicted Mass Flow Rate)")
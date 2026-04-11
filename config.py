#!/usr/bin/env python3
'''
Data Science Student Fellowship -- Jorge R. Gherson
Project: Visualization of Sound Spectra of a Whistle.
-----------------------------------------------------
File Description 
This file stores hyperparameters and relevant data for the project. 
'''

import os 

# FILE PATHS & BASE DIRECTORIES
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Visualization_Dataset_copy1.mat")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")

os.makedirs(PROCESSED_DIR, exist_ok=True) # ensure the directory exists. 

# GLOBAL ACOUSTIC HYPERPARAMETERS 
fs = 48000          # Sampling Frequency (Hz)
f_min = 100         # Minimum Frequency for Extraction (Hz)
f_max = 20000       # Maximum Frequency for Extraction (Hz)
epsilon = 1e-12     # Small constant to prevent log(0)

# SPECTROGRAM PARAMETERS 
spec_nperseg = 1024         # Window size for spectrogram generation. Larger windows give better frequency resolution but worse time resolution.
spec_noverlap = 512         # Overlap between windows for spectrogram generation.
spec_window = 'hann'        # Window type for spectrogram generation 
spec_vmin_db = -120         # Dynamic range for noise floor clipping. 

# WELCH'S METHOD PARAMETERS
welch_npserseg = 4096 
welch_window = 'hann'
welch_vmin_db = -120 

# STFT PARAMETERS (for 2D representation)
stft_nperseg = 512      # Window size for STFT
stft_noverlap = 256     # Overlap between windows
stft_window = 'hann'
stft_vmin_db = -60       # Dynamic range for noise floor clipping (improved visibility)
stft_percentile = 99     # Percentile for robust normalization across samples


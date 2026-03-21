#!/usr/bin/env python3
'''
Data Science Student Fellowship -- Jorge R. Gherson
Project: Visualization of Sound Spectra of a Whistle.
-----------------------------------------------------
File Description 
This file used config.py to unpack the dataset, process the acoustic signals into 2D matrices, and save them as neat .npy array. 
'''

import numpy as np 
import scipy as sp 
import scipy.io 
import scipy.signal 
import os 
import config 

def load_raw_data(filepath): 
    # Load the raw .mat file and extracts necessary variables.
    print(f"Loading dataset from: {filepath}")
    matdata = sp.io.loadmat(filepath, squeeze_me=True, struct_as_record=True)

    # Extract and flatten signals 
    P_all = matdata["P_all"]
    frequency = matdata["frequency"]
    signal_all = np.vstack([sig.flatten() for sig in matdata["signal_all"]])
    amplitude_all = np.vstack([amp.flatten() for amp in matdata["amplitude_all"]])
    m_dot_all = matdata["m_dot_all"]

    print(f"Successfully loaded {len(m_dot_all)} samples.")
    return P_all, frequency, signal_all, amplitude_all, m_dot_all

def generate_welch_psd(signal_matrix, fs): 
    # Processes raw signals into Power Spectral Density (PSD) arrays using Welch's method. 
    print("Generating Welch PSDs...")
    Pxx_list = []

    for sig in signal_matrix: 
        f, Pxx = sp.signal.welch(
            sig, 
            fs = fs, 
            nperseg = config.welch_npserseg, 
            window = config.welch_window
        )
        Pxx_list.append(Pxx)
    
    Pxx_all = np.array(Pxx_list)

    # Convert signals to dB and normalize. 
    psd_db = 10 * np.log10(Pxx_all + config.epsilon)
    psd_norm = psd_db - np.max(psd_db)

    # Clip the noise floor
    psd_norm = np.clip(psd_norm, a_min=config.welch_vmin_db, a_max = 0)

    return f, psd_norm

def process_and_save(): 
    # Main pipeline to load, process, and save the data for CNN. 
    # 1. Load data 
    P_all, frequency, signals, amplitudes, mass_flows = load_raw_data(config.DATA_PATH)

    # 2. Process Data 
    freqs, psd_features = generate_welch_psd(signals, config.fs)

    # filter frequency range of interest 
    mask = (freqs >= config.f_min) & (freqs <= config.f_max)
    freqs_cropped = freqs[mask]
    psd_features_cropped = psd_features[:, mask]

    print(f"Final feature shape (Samples, Frequencies): {psd_features_cropped.shape}")

    # 3. Save data as NumPy arrays for ML pipeline 
    features_path = os.path.join(config.PROCESSED_DIR, "psd_features.npy")
    labels_path = os.path.join(config.PROCESSED_DIR, "mass_flow_labels.npy")
    freqs_path = os.path.join(config.PROCESSED_DIR, "frequencies.npy")

    np.save(features_path, psd_features_cropped)
    np.save(labels_path, mass_flows)
    np.save(freqs_path, freqs_cropped)

    print(f"Data successfully processed and saved to {config.PROCESSED_DIR}")

if __name__ == "__main__": 
    process_and_save()
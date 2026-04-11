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

def generate_stft_spectrograms(signal_matrix, fs):
    # Generates Short-Time Fourier Transform spectrograms for 2D representation.
    # Output shape: (num_samples, num_freq_bins, num_time_frames)
    print("Generating STFT Spectrograms...")
    stft_list = []
    
    for sig in signal_matrix:
        # Compute STFT
        f, t, Zxx = sp.signal.stft(
            sig,
            fs=fs,
            nperseg=config.stft_nperseg,
            noverlap=config.stft_noverlap,
            window=config.stft_window
        )
        
        # Convert to magnitude and then dB
        Zxx_mag = np.abs(Zxx)
        Zxx_db = 20 * np.log10(Zxx_mag + config.epsilon)
        
        stft_list.append(Zxx_db)
    
    # Stack all spectrograms: (num_samples, num_freq_bins, num_time_frames)
    stft_all = np.array(stft_list)
    
    # Normalize by percentile to balance weak and strong signals
    # This prevents high-flow samples from being washed out by low-flow peaks
    percentile_value = np.percentile(stft_all, config.stft_percentile)
    stft_norm = stft_all - percentile_value
    
    # Clip the noise floor
    stft_norm = np.clip(stft_norm, a_min=config.stft_vmin_db, a_max=0)
    
    return f, t, stft_norm

def process_and_save(): 
    # Main pipeline to load, process, and save the data for CNN. 
    # 1. Load data 
    P_all, frequency, signals, amplitudes, mass_flows = load_raw_data(config.DATA_PATH)

    # 2. Process 1D PSD Data (Welch's Method)
    freqs, psd_features = generate_welch_psd(signals, config.fs)

    # filter frequency range of interest 
    mask = (freqs >= config.f_min) & (freqs <= config.f_max)
    freqs_cropped = freqs[mask]
    psd_features_cropped = psd_features[:, mask]

    print(f"1D PSD feature shape (Samples, Frequencies): {psd_features_cropped.shape}")

    # 3. Process 2D STFT Data  
    stft_freqs, stft_times, stft_features = generate_stft_spectrograms(signals, config.fs)
    
    # Filter frequency range
    freq_mask = (stft_freqs >= config.f_min) & (stft_freqs <= config.f_max)
    stft_freqs_cropped = stft_freqs[freq_mask]
    stft_features_cropped = stft_features[:, freq_mask, :]
    
    print(f"2D STFT feature shape (Samples, Frequencies, Times): {stft_features_cropped.shape}")

    # 4. Save data as NumPy arrays for ML pipeline 
    # 1D PSD data
    psd_features_path = os.path.join(config.PROCESSED_DIR, "psd_features.npy")
    labels_path = os.path.join(config.PROCESSED_DIR, "mass_flow_labels.npy")
    freqs_path = os.path.join(config.PROCESSED_DIR, "frequencies.npy")

    np.save(psd_features_path, psd_features_cropped)
    np.save(labels_path, mass_flows)
    np.save(freqs_path, freqs_cropped)
    
    # 2D STFT data
    stft_features_path = os.path.join(config.PROCESSED_DIR, "stft_features.npy")
    stft_freqs_path = os.path.join(config.PROCESSED_DIR, "stft_frequencies.npy")
    stft_times_path = os.path.join(config.PROCESSED_DIR, "stft_times.npy")
    
    np.save(stft_features_path, stft_features_cropped)
    np.save(stft_freqs_path, stft_freqs_cropped)
    np.save(stft_times_path, stft_times)

    print(f"Data successfully processed and saved to {config.PROCESSED_DIR}")

if __name__ == "__main__": 
    process_and_save()
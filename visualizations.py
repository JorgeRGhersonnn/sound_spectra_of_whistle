#!/usr/bin/env python3
'''
Data Science Student Fellowship -- Jorge R. Gherson
Project: Visualization of Sound Spectra of a Whistle.
-----------------------------------------------------
File Description 
This file takes the numpy arrays from processed_data to create plots. 

Structure: 

1. Global Configuration 
2. Colormaps 
3. Graphics
'''
import numpy as np 
import os 
import matplotlib.pyplot as plt 
import config 

# -- GLOBAL STYLE CONFIG -- 
def set_plot_style(): 
    plt.rcParams.update({
        'font.size': 12, 
        'axes.titlesize': 14, 
        'axes.labelsize': 12,
        'xtick.labelsize': 10, 
        'ytick.labelsize': 10,
        'legend.fontsize': 10, 
        'figure.dpi': 300, 
        'axes.grid': True, 
        'grid.alpha': 0.3, 
        'grid.linestyle': '--',
        'font.family': 'sans-serif', 
        'axes.spines.top': False, 
        'axes.spines.right': False
    })

# -- LOAD DATA -- 
def load_data(): 
    features_path = os.path.join(config.PROCESSED_DIR, "psd_features.npy")
    labels_path = os.path.join(config.PROCESSED_DIR, "mass_flow_labels.npy")
    freqs_path = os.path.join(config.PROCESSED_DIR, "frequencies.npy")

    psd_features = np.load(features_path)
    mass_flows = np.load(labels_path)
    freqs = np.load(freqs_path)

    return freqs, psd_features, mass_flows

# -- VISUALIZATION FUNCTIONS -- 

def plot_comparative_spectra(freqs, psd, mass_flows):
    # find indices of lowest and highest mass flow rates 
    idx_min = np.argmin(mass_flows)
    idx_max = np.argmax(mass_flows)

    plt.figure(figsize=(10,6))

    plt.plot(freqs, psd[idx_min], label=f"Low Flow ({mass_flows[idx_min]})", 
             color = "#0460a1", linewidth=1.5, alpha=0.9)
    plt.plot(freqs, psd[idx_max], label=f'High Flow ({mass_flows[idx_max]})', 
             color = "#700c0c", linewidth=1.5, alpha=0.9)
    
    plt.title("Acoustic Signature Comparison: Low vs. High Mass Flow")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (dB)")
    plt.xscale('log')
    plt.xlim(config.f_min, config.f_max)

    plt.legend()
    plt.tight_layout()

    # save the figure
    os.makedirs("plots", exist_ok=True)
    plt.savefig("./plots/spectra_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_global_spectrogram(freqs, psd, mass_flows): 
    # sort data by mass flow rate so Y-axis is ordered 
    sort_indices = np.argsort(mass_flows)
    sorted_flows = mass_flows[sort_indices]
    sorted_psd = psd[sort_indices]

    plt.figure(figsize=(10, 6))
    mesh = plt.pcolormesh(freqs, sorted_flows, sorted_psd,
                          shading='auto', cmap='magma',
                          vmin=config.welch_vmin_db, vmax=0)
    
    plt.title("Global Whistle Spectrum Across All Mass Flow Rates")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Mass Flow Rate (kg/hr)")
    plt.xscale('log')
    plt.xlim(config.f_min, config.f_max)

    # add a colorbar
    cbar = plt.colorbar(mesh)
    cbar.set_label("Relative Power Density (dB)", rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig("plots/global_spectrogram.png", dpi=300, bbox_inches='tight')
    plt.show()

# -- Main Execution --
if __name__ == "__main__":
    set_plot_style()
    freqs, psd, mass_flows = load_data()
    
    print("Generating Comparative Spectra Plot...")
    plot_comparative_spectra(freqs, psd, mass_flows)
    
    print("Generating Global Spectrogram...")
    plot_global_spectrogram(freqs, psd, mass_flows)
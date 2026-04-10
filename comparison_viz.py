#!/usr/bin/env python3
'''
Data Science Student Fellowship -- Jorge R. Gherson
Project: Visualization of Sound Spectra of a Whistle.
-----------------------------------------------------
File Description 
This file creates batched comparison visualizations between 1D PSD and 2D STFT
representations across percentile-selected samples (5th, 25th, 50th, 75th, 95th).
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os
import config


class PercentileSampler:
    """Samples data points based on percentile values of mass flow rate."""
    
    def __init__(self, mass_flows):
        self.mass_flows = mass_flows
        self.percentiles = [5, 25, 50, 75, 95]
    
    def get_percentile_indices(self):
        """Returns indices corresponding to specified percentiles."""
        indices = {}
        for p in self.percentiles:
            value = np.percentile(self.mass_flows, p)
            # Find closest index to this value
            idx = np.argmin(np.abs(self.mass_flows - value))
            indices[p] = idx
        return indices
    
    def get_contrast_pairs(self):
        """Returns pairs of extreme contrasts: (low-flow index, high-flow index)."""
        pairs = []
        percentile_indices = self.get_percentile_indices()
        
        # Low vs. High pairs
        pairs.append({
            'low': percentile_indices[5],
            'high': percentile_indices[95],
            'label': '5th vs 95th Percentile'
        })
        pairs.append({
            'low': percentile_indices[25],
            'high': percentile_indices[75],
            'label': '25th vs 75th Percentile'
        })
        
        return pairs


def set_plot_style():
    """Configure matplotlib style for publication-quality plots."""
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'axes.grid': True,
        'grid.alpha': 0.2,
        'grid.linestyle': '--',
        'font.family': 'sans-serif',
        'axes.spines.top': False,
        'axes.spines.right': False
    })


def load_all_data():
    """Load both 1D PSD and 2D STFT features."""
    psd_path = os.path.join(config.PROCESSED_DIR, "psd_features.npy")
    stft_path = os.path.join(config.PROCESSED_DIR, "stft_features.npy")
    labels_path = os.path.join(config.PROCESSED_DIR, "mass_flow_labels.npy")
    freqs_path = os.path.join(config.PROCESSED_DIR, "frequencies.npy")
    stft_freqs_path = os.path.join(config.PROCESSED_DIR, "stft_frequencies.npy")
    stft_times_path = os.path.join(config.PROCESSED_DIR, "stft_times.npy")

    psd_features = np.load(psd_path)
    stft_features = np.load(stft_path)
    mass_flows = np.load(labels_path)
    freqs = np.load(freqs_path)
    stft_freqs = np.load(stft_freqs_path)
    stft_times = np.load(stft_times_path)

    return {
        'psd': psd_features,
        'stft': stft_features,
        'mass_flows': mass_flows,
        'freqs': freqs,
        'stft_freqs': stft_freqs,
        'stft_times': stft_times
    }


def plot_percentile_grid(data, output_dir="plots"):
    """
    Create a comprehensive 5x2 grid showing:
    - Row: Each percentile (5th, 25th, 50th, 75th, 95th)
    - Col 1: 1D PSD spectrum
    - Col 2: 2D STFT spectrogram heatmap
    """
    os.makedirs(output_dir, exist_ok=True)
    sampler = PercentileSampler(data['mass_flows'])
    percentile_indices = sampler.get_percentile_indices()
    
    fig = plt.figure(figsize=(14, 16))
    gs = GridSpec(5, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    for row, p in enumerate(sampler.percentiles):
        idx = percentile_indices[p]
        mass_flow = data['mass_flows'][idx]
        
        # --- Column 1: 1D PSD ---
        ax1 = fig.add_subplot(gs[row, 0])
        psd = data['psd'][idx]
        ax1.plot(data['freqs'], psd, color='#0460a1', linewidth=1.5, alpha=0.8)
        ax1.set_ylabel("Power (dB)")
        ax1.set_title(f"{p}th Percentile (m={mass_flow:.2f} kg/hr) - 1D PSD")
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        
        # --- Column 2: 2D STFT ---
        ax2 = fig.add_subplot(gs[row, 1])
        stft = data['stft'][idx]  # shape: (freq_bins, time_frames)
        im = ax2.pcolormesh(data['stft_times'], data['stft_freqs'], stft,
                            shading='auto', cmap='magma',
                            vmin=config.stft_vmin_db, vmax=0)
        ax2.set_ylabel("Frequency (Hz)")
        ax2.set_title(f"{p}th Percentile (m={mass_flow:.2f} kg/hr) - 2D STFT")
        ax2.set_yscale('log')
        
        # Only add colorbar to rightmost column
        if row == 0:
            cbar = plt.colorbar(im, ax=ax2, label="Magnitude (dB)")
    
    # Common x-label for bottom row
    fig.add_subplot(gs[4, 0]).set_xlabel("Frequency (Hz)")
    fig.add_subplot(gs[4, 1]).set_xlabel("Time (s)")
    
    plt.suptitle("Percentile-Based Comparison: 1D PSD vs 2D STFT Representations",
                 fontsize=14, y=0.995)
    
    plt.savefig(os.path.join(output_dir, "percentile_grid_1d_vs_2d.png"),
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved percentile grid to {output_dir}/percentile_grid_1d_vs_2d.png")
    plt.close()


def plot_contrast_pairs(data, output_dir="plots"):
    """
    Create side-by-side comparisons for extreme contrast pairs:
    - Top row: Low mass flow
    - Bottom row: High mass flow
    Each pair shows 1D PSD and 2D STFT.
    """
    os.makedirs(output_dir, exist_ok=True)
    sampler = PercentileSampler(data['mass_flows'])
    pairs = sampler.get_contrast_pairs()
    
    for pair_idx, pair in enumerate(pairs):
        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
        
        for row, flow_type in enumerate(['low', 'high']):
            idx = pair[flow_type]
            mass_flow = data['mass_flows'][idx]
            
            # --- Column 1: 1D PSD ---
            ax1 = fig.add_subplot(gs[row, 0])
            psd = data['psd'][idx]
            color = '#0460a1' if flow_type == 'low' else '#700c0c'
            ax1.plot(data['freqs'], psd, color=color, linewidth=2, alpha=0.85)
            ax1.set_ylabel("Power (dB)")
            flow_label = f"Low Flow - {mass_flow:.2f} kg/hr" if flow_type == 'low' else f"High Flow - {mass_flow:.2f} kg/hr"
            ax1.set_title(f"{flow_label} - 1D PSD")
            ax1.set_xscale('log')
            ax1.grid(True, alpha=0.3)
            
            # --- Column 2: 2D STFT ---
            ax2 = fig.add_subplot(gs[row, 1])
            stft = data['stft'][idx]
            im = ax2.pcolormesh(data['stft_times'], data['stft_freqs'], stft,
                                shading='auto', cmap='magma',
                                vmin=config.stft_vmin_db, vmax=0)
            ax2.set_ylabel("Frequency (Hz)")
            ax2.set_title(f"{flow_label} - 2D STFT")
            ax2.set_yscale('log')
            plt.colorbar(im, ax=ax2, label="Magnitude (dB)")
        
        # Common x-labels
        fig.add_subplot(gs[1, 0]).set_xlabel("Frequency (Hz)")
        fig.add_subplot(gs[1, 1]).set_xlabel("Time (s)")
        
        plt.suptitle(f"Contrast Comparison: {pair['label']}", fontsize=14, y=0.995)
        
        filename = f"contrast_pair_{pair_idx + 1}_{pair['label'].replace(' ', '_').replace('th', '')}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        print(f"✓ Saved {pair['label']} contrast pair to {output_dir}/{filename}")
        plt.close()


def plot_all_percentiles_overlay(data, output_dir="plots"):
    """
    Create overlaid 1D spectra for all 5 percentiles to show progression.
    """
    os.makedirs(output_dir, exist_ok=True)
    sampler = PercentileSampler(data['mass_flows'])
    percentile_indices = sampler.get_percentile_indices()
    
    # Create two figures: one with linear scale, one with log scale
    for scale_type, x_scale in [('log', 'log'), ('linear', 'linear')]:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.RdYlBu_r(np.linspace(0, 1, 5))
        
        for i, p in enumerate(sampler.percentiles):
            idx = percentile_indices[p]
            mass_flow = data['mass_flows'][idx]
            psd = data['psd'][idx]
            
            ax.plot(data['freqs'], psd, label=f"{p}th percentile ({mass_flow:.1f} kg/hr)",
                   color=colors[i], linewidth=2, alpha=0.8)
        
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (dB)")
        ax.set_xscale(x_scale)
        ax.set_title(f"Percentile Progression: 1D PSD Spectra ({x_scale.capitalize()} Scale)")
        ax.legend(loc='best', framealpha=0.95)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"percentile_overlay_{scale_type}_scale.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        print(f"✓ Saved percentile overlay ({x_scale} scale) to {output_dir}/{filename}")
        plt.close()


def plot_stft_heatmap_progression(data, output_dir="plots"):
    """
    Create a single figure showing all 5 percentile STFT heatmaps in a row.
    """
    os.makedirs(output_dir, exist_ok=True)
    sampler = PercentileSampler(data['mass_flows'])
    percentile_indices = sampler.get_percentile_indices()
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for col, p in enumerate(sampler.percentiles):
        idx = percentile_indices[p]
        mass_flow = data['mass_flows'][idx]
        stft = data['stft'][idx]
        
        im = axes[col].pcolormesh(data['stft_times'], data['stft_freqs'], stft,
                                  shading='auto', cmap='magma',
                                  vmin=config.stft_vmin_db, vmax=0)
        axes[col].set_title(f"{p}th Percentile\n({mass_flow:.1f} kg/hr)")
        axes[col].set_yscale('log')
        axes[col].set_ylabel("Frequency (Hz)" if col == 0 else "")
        axes[col].set_xlabel("Time (s)")
        plt.colorbar(im, ax=axes[col], label="dB")
    
    plt.suptitle("STFT Spectrogram Progression Across Percentiles", fontsize=14, y=1.02)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "stft_percentile_heatmaps.png"),
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved STFT percentile heatmaps to {output_dir}/stft_percentile_heatmaps.png")
    plt.close()


def generate_all_visualizations(output_dir="plots"):
    """Main execution: generate all comparison visualizations."""
    print("Loading data...")
    data = load_all_data()
    
    print(f"Data shapes:")
    print(f"  1D PSD: {data['psd'].shape}")
    print(f"  2D STFT: {data['stft'].shape}")
    print(f"  Mass flows: {data['mass_flows'].shape}")
    print()
    
    set_plot_style()
    
    print("Generating visualizations...")
    plot_percentile_grid(data, output_dir)
    plot_contrast_pairs(data, output_dir)
    plot_all_percentiles_overlay(data, output_dir)
    plot_stft_heatmap_progression(data, output_dir)
    
    print("\n✓ All visualizations generated successfully!")


if __name__ == "__main__":
    generate_all_visualizations()

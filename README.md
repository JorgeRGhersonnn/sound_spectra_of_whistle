# Sound Spectra of a Whistle

This repository contains a complete machine learning pipeline that predicts the mass flow rate of a system based purely on its acoustic whistle signature. It processes raw `.mat` audio recordings, extracts both 1D frequency features (via Welch's method) and 2D time-frequency features (via STFT), and trains parallel convolutional neural network architectures for comparative analysis.

## Project Structure

### Core Processing & Data
* `config.py`: Global hyperparameters (frequencies, Welch's method parameters, STFT parameters).
* `data_preprocessing.py`: Loads raw `.mat` files, applies Welch's Method for 1D PSD and STFT for 2D representations, saves `.npy` arrays.
* `dataset.py`: Custom PyTorch `Dataset` and `DataLoader` classes for both 1D PSD (`WhistleDataset`) and 2D STFT (`WhistleDataset2D`).

### Model Architectures
* `models.py`: Defines the original 1D-CNN architecture.
* `models_2d.py`: Defines extended 1D CNN and novel 2D CNN architectures for parallel model comparison.

### Training & Comparison
* `train.py`: The PyTorch training loop for the original 1D model with validation and loss tracking.
* `train_parallel.py`: Parallel training pipeline that trains both 1D and 2D models sequentially, generates comparative performance reports and visualization plots.

### Analysis & Evaluation
* `baselines.py`: Trains Ridge Regression and Linear Regression benchmark models.
* `tree_models.py`: Evaluates the data under 2 tree models: (a) Random Forest, and (b) XGBoost Regressor.

### Visualization
* `visualizations.py`: Generates basic presentation-ready spectrograms and comparative plots.
* `comparison_viz.py`: Advanced visualization framework with percentile-based sampling for efficient 1D vs 2D representation comparison across 5 percentile levels (5th, 25th, 50th, 75th, 95th).
* `animations.py`: Contains MANIM-based animations written in Python for presentation purposes.

## Key Features

### Dual Representation Analysis
- **1D Representation**: Power Spectral Density (PSD) via Welch's method - 267 samples × 1,698 frequency bins
- **2D Representation**: Short-Time Fourier Transform (STFT) spectrograms - 267 samples × 212 frequency bins × 392 time frames

### Parallel Model Comparison
- **1D CNN**: 279,649 parameters, optimized for frequency-only analysis
- **2D CNN**: 8,536,161 parameters, captures spectrotemporal patterns
- Direct empirical comparison to determine which representation better suits the task

### Comprehensive Visualization
- **Percentile Grids**: 5×2 comparative grids showing all percentile levels side-by-side
- **Contrast Pairs**: Extreme (5th vs 95th) and moderate (25th vs 75th) flow comparisons
- **Progression Plots**: Overlay visualizations showing smooth spectral evolution
- **STFT Heatmaps**: Time-frequency representations across percentiles

## Workflow

1. **Data Preprocessing**: `python data_preprocessing.py` 
   - Generates both 1D and 2D features
   - Outputs: 6 `.npy` files in `processed_data/`

2. **Visualization (Optional)**: `python comparison_viz.py`
   - Creates percentile-based comparison grids
   - Outputs: 6 high-quality PNG files showing 1D vs 2D comparison

3. **Parallel Training (Optional)**: `python train_parallel.py`
   - Trains 1D CNN with PSD features
   - Trains 2D CNN with STFT spectrograms
   - Generates performance comparison report
   - Outputs: Model checkpoints + comparison plots

4. **Baseline Evaluation (Optional)**: `python baselines.py` / `python tree_models.py`
   - Linear regression and tree-based baselines
   - Feature importance analysis

## Results Summary

| Metric | 1D CNN (PSD) | 2D CNN (STFT) |
|--------|-------------|---------------|
| **Best Validation MAE** | 0.3162 kg/hr ✓ | 0.3553 kg/hr |
| **Parameters** | 279,649 | 8,536,161 |
| **Convergence Epoch** | 16 ✓ | 32 |
| **Parameter Efficiency** | 884** | 24,023 |

** Parameters per 0.001 kg/hr improvement

**Finding**: The simpler 1D CNN with PSD features achieves 11% better performance with 30.5× fewer parameters, suggesting that temporal/spectrographic information does not add discriminative power for this task.

## Data Requirements

**Note 1:** For access to the dataset, please contact owner. 

**Note 2:** The use of artificial intelligence was implemented to test code ideas for this project.
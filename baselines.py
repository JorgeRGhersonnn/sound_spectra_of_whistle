#!/usr/bin/env python3
'''
Data Science Student Fellowship -- Jorge R. Gherson
Project: Visualization of Sound Spectra of a Whistle.
-----------------------------------------------------
File Description 
This files uses the preprocessed data from data_preprocessing.py to implement two baselines: 
    1. Peak Frequency Baseline 
    A simple linear model that only looks at the single loudest frequency to predict flow rate. 

    2. Full Spectrum Baseline (Ridge Regression)
    A model using all 1698 frequency bins, preventing overfitting from highly correlated features. 
'''

import numpy as np 
import os 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import config 

def load_processed_data(): 
    # Load the preprocessed array from the processed_data directory. 
    print("Loading processed data...")
    features_path = os.path.join(config.PROCESSED_DIR, "psd_features.npy")
    labels_path = os.path.join(config.PROCESSED_DIR, "mass_flow_labels.npy")
    freqs_path = os.path.join(config.PROCESSED_DIR, "frequencies.npy")

    X = np.load(features_path)      # SHAPE: (267, 1698)
    y = np.load(labels_path)        # SHAPE: (267,)
    freqs = np.load(freqs_path)     # SHAPE: (1698,)

    return X, y, freqs

def evaluate_model(name, y_true, y_pred): 
    # Calculate and print regression metrics. 
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"--- {name} Results ---")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f} kg/hr")
    print(f"R^2: {r2:.4f}\n")

    return r2

def plot_parity(y_true, y_pred_peak, y_pred_ridge): 
    # Plots True vs. Predicted values to visualize accuracy. 
    plt.figure(figsize=(8,8))

    # perfect prediction line 
    min_val = min(y_true.min(), y_pred_peak.min(), y_pred_ridge.min())
    max_val = max(y_true.max(), y_pred_peak.max(), y_pred_ridge.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="Perfect Prediction")

    # Scatter plots for models 
    plt.scatter(y_true, y_pred_peak, alpha=0.6, label="Peak Frequency (Simple Linear Regression)")
    plt.scatter(y_true, y_pred_ridge, alpha=0.6, marker='s', label="Full Spectrum (Ridge)")

    plt.title("Baseline Models: Actual vs. Predicted Mass Flow Rate")
    plt.xlabel("Actual Mass Flow Rate (kg/hr)")
    plt.ylabel("Predicted Mass Flow Rate (kg/hr)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def run_baselines(): 
    X, y, freqs = load_processed_data()

    # -- Feature extraction for baseline 1 -- 
    peak_indices = np.argmax(X, axis=1)     # find index of max amplitude for each sample 
    X_peak_freq = freqs[peak_indices].reshape(-1, 1)    # map index to actual freq value and reshape for scikit

    # Split data into 80% training and 20% testing sets
    X_peak_train, X_peak_test, y_train, y_test = train_test_split(X_peak_freq, y, test_size=0.33, random_state=42)   # random_state ensures same split every time we run. 
    X_full_train, X_full_test, _, _ = train_test_split(X, y, test_size=0.33, random_state=42)

    # -- TRAIN BASELINE 1: Linear Regression of Peak Freq. -- 
    print("Training Peak Frequency (Linear Regression) Model...")
    lr_model = LinearRegression()
    lr_model.fit(X_peak_train, y_train)
    y_pred_peak = lr_model.predict(X_peak_test)
    evaluate_model("Peak Frequency Model", y_test, y_pred_peak)

    # -- TRAIN BASELINE 2: Ridge Regression on Full Spectrum -- 
    print("Training Full Spectrum Ridge Regression...")
    ridge_model = Ridge(alpha=10.0)
    ridge_model.fit(X_full_train, y_train)
    y_pred_ridge = ridge_model.predict(X_full_test)
    evaluate_model("Full Spectrum Ridge Model", y_test, y_pred_ridge)

    # -- VISUALIZATION -- 
    plot_parity(y_test, y_pred_peak, y_pred_ridge)

if __name__ == "__main__":
    run_baselines()
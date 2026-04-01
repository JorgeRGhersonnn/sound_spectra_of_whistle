#!/usr/bin/env python3
'''
Data Science Student Fellowship -- Jorge R. Gherson
Project: Visualization of Sound Spectra of a Whistle.
-----------------------------------------------------
File Description 
This script will train both a Random Forest and XGBoost Regressor. 
'''

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb 
import config 

def load_processed_data():
    features_path = os.path.join(config.PROCESSED_DIR, "psd_features.npy")
    labels_path = os.path.join(config.PROCESSED_DIR, "mass_flow_labels.npy")
    freqs_path = os.path.join(config.PROCESSED_DIR, "frequencies.npy")

    X = np.load(features_path)
    y = np.load(labels_path)
    freqs = np.load(freqs_path)

    return X, y, freqs

def evaluate_model(name, y_true, y_pred): 
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"-- {name} --")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f} kg/hr")
    print(f"R^2: {r2:.4f}\n")
    return r2, mae

def plot_feature_importance(model, freqs, title, top_n=10):
    importances = model.feature_importances_ 
    indices = np.argsort(importances)[::-1]

    top_freqs = freqs[indices][:top_n]
    top_importances = importances[indices][:top_n]

    plt.figure(figsize=(10, 5))
    plt.bar(range(top_n), top_importances, align='center', color='#2ca02c') 
    plt.xticks(range(top_n), [f"{f:.1f} Hz" for f in top_freqs], rotation=45)
    plt.xlim([-1, top_n])
    plt.ylabel("Relative Importance")
    plt.title(f"Top {top_n} Most Important Frequencies ({title})")
    plt.tight_layout()
    plt.show()

def run_tree_model(): 
    X, y, freqs = load_processed_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # MODEL 1: RANDOM FOREST
    print("Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    evaluate_model("Random Forest Regressor", y_test, y_pred_rf)

    # MODEL 2: XGBoost 
    xgb_model = xgb.XGBRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth = 4, 
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    evaluate_model("XGBoost Regressor:", y_test, y_pred_xgb)

    # Visualizations 
    plot_feature_importance(xgb_model, freqs, "XGBoost")

if __name__ == "__main__": 
    run_tree_model()
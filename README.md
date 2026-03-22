# Sound Spectra of a Whistle

This repository contains a complete machine learning pipeline that predicts the mass flow rate of a system based purely on its acoustic whistle signature. It processes raw `.mat` audio recordings, extracts frequency features, and trains both linear baselines and a 1D Convolutional Neural Network (1D-CNN) using PyTorch.

## Project Structure

* `config.py`: Global hyperparameters (frequencies, Welch's method parameters).
* `data_preprocessing.py`: Loads raw `.mat` files, applies Welch's Method for PSD, and saves `.npy` arrays.
* `baselines.py`: Trains Ridge Regression and simple Linear Regression benchmark models.
* `dataset.py`: Custom PyTorch `Dataset` and `DataLoader` for batching.
* `models.py`: Defines the 1D-CNN architecture.
* `train.py`: The PyTorch training loop with validation and loss tracking.
* `visualizations.py`: Generates presentation-ready spectrograms and comparative plots.

**Note:** For access to the dataset, please contact owner. 
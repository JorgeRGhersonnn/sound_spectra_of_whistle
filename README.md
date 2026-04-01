# Sound Spectra of a Whistle

This repository contains a complete machine learning pipeline that predicts the mass flow rate of a system based purely on its acoustic whistle signature. It processes raw `.mat` audio recordings, extracts frequency features, and trains both linear baselines and a 1D Convolutional Neural Network (1D-CNN) using PyTorch.

## Project Structure

* `config.py`: Global hyperparameters (frequencies, Welch's method parameters).
* `data_preprocessing.py`: Loads raw `.mat` files, applies Welch's Method for PSD, and saves `.npy` arrays.
* `baselines.py`: Trains Ridge Regression and simple Linear Regression benchmark models.
* `dataset.py`: Custom PyTorch `Dataset` and `DataLoader` for batching.
* `models.py`: Defines the 1D-CNN architecture.
* `train.py`: The PyTorch training loop with validation and loss tracking.
* `tree_models.py`: Evaluates the data under 2 tree models: (a) Random Forest, and (b) XGBoost Regressor.
* `visualizations.py`: Generates presentation-ready spectrograms and comparative plots.

**Note 1:** For access to the dataset, please contact owner. 
**Note 2:** The use of artificial intelligence was implemented to test code ideas for this project.
#!/usr/bin/env python3
'''
Data Science Student Fellowship -- Jorge R. Gherson
Project: Visualization of Sound Spectra of a Whistle.
-----------------------------------------------------
File Description 
'''
import torch 
import torch.nn as nn 
import torch.optim as optim 
import matplotlib.pyplot as plt 
import os 

from dataset import get_dataloaders 
from models import WhistleCNN 
import config 

def train_model(): 
    # 1. Hardware setup (Use GPU if available, Apple Silicon MPS, or fallback to CPU) -- coded with gemini ai 
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 2. Load Data 
    train_loader, test_loader = get_dataloaders(batch_size=16)

    # 3. Initialize Model, Loss Function, and Optimizer 
    model = WhistleCNN().to(device)

    # L1Loss is MAE, matches metric from baseline 
    criterion = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Training Hyperparameters 
    epochs = 50 
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    print("Starting training...")

    # 5. Training loop 
    for epoch in range(epochs): 
        # -- TRAINING PHASE -- 
        model.train()                   # tell torch we are in training mode 
        running_train_loss = 0.0

        for features, labels in train_loader: 
            features, labels = features.to(device), labels.to(device)

            # a. clear old gradients 
            optimizer.zero_grad()

            # b. make guess 
            predictions = model(features)

            # c. calculate loss (i.e., error)
            loss = criterion(predictions, labels)

            # d. backward pass (i.e., calculate gradients)
            loss.backward()

            # e. update model weigths. 
            optimizer.step()

            running_train_loss += loss.item() * features.size(0)
        
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # -- VALIDATION PHASE -- 
        model.eval()                    # tell torch we are in evaluation mode
        running_val_loss = 0.0
        
        with torch.no_grad():
            for features, labels in test_loader: 
                features, labels = features.to(device), labels.to(device)

                predictions = model(features)
                loss = criterion(predictions, labels)

                running_val_loss += loss.item() * features.size(0)
        
        epoch_val_loss = running_val_loss / len(test_loader.dataset)
        val_losses.append(epoch_val_loss)

        # print progress every 5 epochs 
        if (epoch + 1) % 5 == 0 or epoch == 0: 
            print(f"Epoch [{epoch+1}/{epochs}] | Train MAE: {epoch_train_loss:.4f} | Val MAE: {epoch_val_loss:.4f}")
        
        # save model if best one so far 
        if epoch_val_loss < best_val_loss: 
            best_val_loss = epoch_val_loss
            os.makedirs("saved_models", exist_ok=True)
            torch.save(model.state_dict(), "saved_models/best_whistle_cnn.pth")
        
        print(f"\nTraining complete! Best Validation MAE: {best_val_loss:.4f}")
    
    # 6. Plot Learning Curves 
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss (Mean Absolute Error)")
    plt.plot(val_losses, label="Validation Loss (Mean Absolute Error)")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Absolute Error (kg/hr)")
    plt.title("CNN Training Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    train_model()
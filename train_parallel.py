#!/usr/bin/env python3
'''
Data Science Student Fellowship -- Jorge R. Gherson
Project: Visualization of Sound Spectra of a Whistle.
-----------------------------------------------------
File Description 
Parallel training pipeline for 1D and 2D CNN models to compare performance.
Trains both models independently and generates comparative analysis plots.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

from dataset import get_dataloaders, get_dataloaders_2d
from models import WhistleCNN
from models_2d import WhistleCNN2D, WhistleCNN1D
import config


class ModelTrainer:
    """Handles training and validation of a single model."""
    
    def __init__(self, model, device, model_name="model"):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        
        # Count parameters
        self.param_count = sum(p.numel() for p in model.parameters())
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        
        for features, labels in train_loader:
            features, labels = features.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            predictions = self.model(features)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * features.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss
    
    def validate(self, val_loader, criterion):
        """Validate on validation set."""
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                predictions = self.model(features)
                loss = criterion(predictions, labels)
                running_loss += loss.item() * features.size(0)
        
        epoch_loss = running_loss / len(val_loader.dataset)
        return epoch_loss
    
    def train_full(self, train_loader, val_loader, epochs=50, lr=0.001):
        """Complete training loop."""
        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"{'='*60}")
        print(f"Model Parameters: {self.param_count:,}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs} | Learning Rate: {lr}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            val_loss = self.validate(val_loader, criterion)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_checkpoint()
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                      f"Train MAE: {train_loss:.4f} | "
                      f"Val MAE: {val_loss:.4f} | "
                      f"Best: {self.best_val_loss:.4f} (Epoch {self.best_epoch+1})")
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Validation MAE: {self.best_val_loss:.4f} (Epoch {self.best_epoch+1})")
        print(f"{'='*60}\n")
    
    def save_checkpoint(self):
        """Save best model checkpoint."""
        os.makedirs("saved_models", exist_ok=True)
        checkpoint_path = f"saved_models/best_{self.model_name}.pth"
        torch.save(self.model.state_dict(), checkpoint_path)
    
    def load_checkpoint(self):
        """Load best model checkpoint."""
        checkpoint_path = f"saved_models/best_{self.model_name}.pth"
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"Loaded checkpoint: {checkpoint_path}")


def train_1d_model(epochs=50, batch_size=16):
    """Train 1D CNN model."""
    device = get_device()
    
    train_loader, val_loader = get_dataloaders(batch_size=batch_size)
    model = WhistleCNN1D()
    
    trainer = ModelTrainer(model, device, model_name="whistle_cnn_1d")
    trainer.train_full(train_loader, val_loader, epochs=epochs, lr=0.001)
    
    return trainer


def train_2d_model(epochs=50, batch_size=16):
    """Train 2D CNN model."""
    device = get_device()
    
    train_loader, val_loader = get_dataloaders_2d(batch_size=batch_size)
    model = WhistleCNN2D()
    
    trainer = ModelTrainer(model, device, model_name="whistle_cnn_2d")
    trainer.train_full(train_loader, val_loader, epochs=epochs, lr=0.001)
    
    return trainer


def get_device():
    """Get available device (GPU, MPS, or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    return device


def plot_training_comparison(trainer_1d, trainer_2d, output_dir="plots"):
    """Create side-by-side training curves comparison."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Individual training curves
    epochs_1d = range(1, len(trainer_1d.train_losses) + 1)
    epochs_2d = range(1, len(trainer_2d.train_losses) + 1)
    
    axes[0].plot(epochs_1d, trainer_1d.train_losses, 'o-', label="1D Train", alpha=0.7, markersize=4)
    axes[0].plot(epochs_1d, trainer_1d.val_losses, 's-', label="1D Val", alpha=0.7, markersize=4)
    axes[0].axvline(trainer_1d.best_epoch + 1, color='blue', linestyle='--', alpha=0.5, label=f"Best 1D (Epoch {trainer_1d.best_epoch+1})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Mean Absolute Error (kg/hr)")
    axes[0].set_title("1D CNN Training Curve")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs_2d, trainer_2d.train_losses, 'o-', label="2D Train", alpha=0.7, markersize=4)
    axes[1].plot(epochs_2d, trainer_2d.val_losses, 's-', label="2D Val", alpha=0.7, markersize=4)
    axes[1].axvline(trainer_2d.best_epoch + 1, color='red', linestyle='--', alpha=0.5, label=f"Best 2D (Epoch {trainer_2d.best_epoch+1})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean Absolute Error (kg/hr)")
    axes[1].set_title("2D CNN Training Curve")
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_comparison_individual.png"), dpi=300, bbox_inches='tight')
    print(f"✓ Saved individual training curves to {output_dir}/training_comparison_individual.png")
    plt.close()


def plot_overlay_comparison(trainer_1d, trainer_2d, output_dir="plots"):
    """Create overlaid comparison of both models."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Normalize epochs to common range for overlay
    epochs_1d = np.arange(len(trainer_1d.val_losses))
    epochs_2d = np.arange(len(trainer_2d.val_losses))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(epochs_1d, trainer_1d.val_losses, 'o-', label="1D CNN (PSD)", 
            color='#0460a1', linewidth=2.5, markersize=5, alpha=0.8)
    ax.plot(epochs_2d, trainer_2d.val_losses, 's-', label="2D CNN (STFT)", 
            color='#700c0c', linewidth=2.5, markersize=5, alpha=0.8)
    
    # Mark best validation losses
    ax.plot(trainer_1d.best_epoch, trainer_1d.best_val_loss, '*', 
            color='#0460a1', markersize=20, label=f"1D Best: {trainer_1d.best_val_loss:.4f} kg/hr")
    ax.plot(trainer_2d.best_epoch, trainer_2d.best_val_loss, '*', 
            color='#700c0c', markersize=20, label=f"2D Best: {trainer_2d.best_val_loss:.4f} kg/hr")
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation MAE (kg/hr)", fontsize=12)
    ax.set_title("1D CNN vs 2D CNN: Validation Performance Comparison", fontsize=13)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_comparison_overlay.png"), dpi=300, bbox_inches='tight')
    print(f"✓ Saved overlay comparison to {output_dir}/training_comparison_overlay.png")
    plt.close()


def generate_performance_report(trainer_1d, trainer_2d, output_dir="plots"):
    """Generate a text report comparing model performance."""
    os.makedirs(output_dir, exist_ok=True)
    
    report = f"""
{'='*70}
PARALLEL 1D vs 2D CNN TRAINING REPORT
{'='*70}

TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*70}
MODEL ARCHITECTURE COMPARISON
{'='*70}

1D CNN (PSD-based):
  - Input Shape: (Batch, 1, Frequencies)
  - Feature Representation: 1D Power Spectral Density
  - Total Parameters: {trainer_1d.param_count:,}
  - Architecture: Conv1d blocks with BatchNorm + ReLU

2D CNN (STFT-based):
  - Input Shape: (Batch, 1, Freq_bins, Time_frames)
  - Feature Representation: 2D Time-Frequency Spectrogram
  - Total Parameters: {trainer_2d.param_count:,}
  - Architecture: Conv2d blocks with BatchNorm + ReLU

Parameter Ratio (2D/1D): {trainer_2d.param_count / trainer_1d.param_count:.1f}x

{'='*70}
TRAINING PERFORMANCE
{'='*70}

1D CNN (PSD):
  - Best Validation MAE: {trainer_1d.best_val_loss:.4f} kg/hr
  - Best Epoch: {trainer_1d.best_epoch + 1}
  - Final Validation MAE: {trainer_1d.val_losses[-1]:.4f} kg/hr
  - Training MAE (final): {trainer_1d.train_losses[-1]:.4f} kg/hr
  - Total Epochs Trained: {len(trainer_1d.val_losses)}

2D CNN (STFT):
  - Best Validation MAE: {trainer_2d.best_val_loss:.4f} kg/hr
  - Best Epoch: {trainer_2d.best_epoch + 1}
  - Final Validation MAE: {trainer_2d.val_losses[-1]:.4f} kg/hr
  - Training MAE (final): {trainer_2d.train_losses[-1]:.4f} kg/hr
  - Total Epochs Trained: {len(trainer_2d.val_losses)}

{'='*70}
COMPARATIVE ANALYSIS
{'='*70}

Performance Improvement (2D vs 1D):
  - Absolute MAE Difference: {abs(trainer_2d.best_val_loss - trainer_1d.best_val_loss):.4f} kg/hr
  - Relative Improvement: {((trainer_1d.best_val_loss - trainer_2d.best_val_loss) / trainer_1d.best_val_loss * 100):+.2f}%
  - Better Model: {'2D CNN' if trainer_2d.best_val_loss < trainer_1d.best_val_loss else '1D CNN'}

Efficiency Analysis:
  - 1D Parameters per 0.001 kg/hr improvement: {trainer_1d.param_count / (trainer_1d.best_val_loss / 0.001):.0f}
  - 2D Parameters per 0.001 kg/hr improvement: {trainer_2d.param_count / (trainer_2d.best_val_loss / 0.001):.0f}

Convergence Speed:
  - 1D Converged at: Epoch {trainer_1d.best_epoch + 1}
  - 2D Converged at: Epoch {trainer_2d.best_epoch + 1}
  - Faster Convergence: {'2D' if trainer_2d.best_epoch < trainer_1d.best_epoch else '1D'} CNN

{'='*70}
RECOMMENDATIONS
{'='*70}

"""
    
    if trainer_2d.best_val_loss < trainer_1d.best_val_loss:
        improvement_pct = ((trainer_1d.best_val_loss - trainer_2d.best_val_loss) / 
                          trainer_1d.best_val_loss * 100)
        report += f"""
1. The 2D CNN (STFT-based) model achieves {improvement_pct:.2f}% better performance.
   
2. The 2D representation captures temporal dynamics in the spectrogram that the 
   1D PSD representation cannot encode.
   
3. Despite having {trainer_2d.param_count / trainer_1d.param_count:.1f}x more parameters, 
   the 2D model demonstrates improved generalization.
   
4. RECOMMENDATION: Use 2D CNN with STFT spectrograms for production deployment.

5. Future optimization: Consider input resolution reduction or model compression
   to maintain 2D benefits with fewer parameters.
"""
    else:
        improvement_pct = ((trainer_2d.best_val_loss - trainer_1d.best_val_loss) / 
                          trainer_2d.best_val_loss * 100)
        report += f"""
1. The 1D CNN (PSD-based) model achieves {improvement_pct:.2f}% better performance
   with significantly fewer parameters ({trainer_1d.param_count:,} vs {trainer_2d.param_count:,}).
   
2. The simpler 1D representation is sufficient for this task, suggesting that
   temporal/spectrographic details do not provide additional discriminative power.
   
3. RECOMMENDATION: Use 1D CNN with PSD features for production deployment due to
   better parameter efficiency and comparable/superior performance.
   
4. The 1D model is more computationally efficient and easier to deploy on
   resource-constrained devices.
"""
    
    report += f"""
{'='*70}
SAVED ARTIFACTS
{'='*70}

Checkpoints:
  - saved_models/best_whistle_cnn_1d.pth
  - saved_models/best_whistle_cnn_2d.pth

Visualizations:
  - plots/training_comparison_individual.png (separate training curves)
  - plots/training_comparison_overlay.png (overlay comparison)
  - plots/performance_report.txt (this report)

Data:
  - processed_data/psd_features.npy (1D features)
  - processed_data/stft_features.npy (2D features)
  - plots/percentile_grid_1d_vs_2d.png (visual comparison)

{'='*70}
"""
    
    report_path = os.path.join(output_dir, "performance_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"✓ Saved performance report to {report_path}")


def main(epochs=50, batch_size=16):
    """Main execution: train both models in sequence and compare."""
    
    print("\n" + "="*70)
    print("PARALLEL 1D + 2D CNN TRAINING PIPELINE")
    print("="*70 + "\n")
    
    # Train 1D model
    trainer_1d = train_1d_model(epochs=epochs, batch_size=batch_size)
    
    # Train 2D model
    trainer_2d = train_2d_model(epochs=epochs, batch_size=batch_size)
    
    # Generate comparison plots and report
    plot_training_comparison(trainer_1d, trainer_2d)
    plot_overlay_comparison(trainer_1d, trainer_2d)
    generate_performance_report(trainer_1d, trainer_2d)


if __name__ == "__main__":
    # Adjust epochs and batch_size as needed
    main(epochs=50, batch_size=16)

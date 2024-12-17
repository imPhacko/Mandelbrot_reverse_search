import torch
from src.train import train_model
from src.data_loader import train_loader, val_loader
from src.evaluate import plot_training_history, visualize_predictions
from src.utils import create_run_dir
import os

def main():
    # Create run directory
    run_dir = create_run_dir()
    print(f"Saving results to: {run_dir}")
    
    # Train model
    trained_model, train_losses, val_losses = train_model(train_loader, val_loader)
    
    # Plot and save training history
    plot_training_history(train_losses, val_losses, run_dir)
    
    # Visualize and save predictions
    visualize_predictions(trained_model, val_loader, run_dir)
    
    # Save the trained model
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses
    }, os.path.join(run_dir, 'mandelbrot_model.pth'))

if __name__ == "__main__":
    main()
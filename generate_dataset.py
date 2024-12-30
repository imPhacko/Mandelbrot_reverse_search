import os
import numpy as np
import matplotlib.pyplot as plt
from src.mandelbrot import mandelbrot_zoom
from pathlib import Path

def generate_dataset(num_samples, width, height, max_iter, output_dir):
    """
    Generate a large dataset of Mandelbrot set images.
    
    Parameters:
        num_samples: Number of images to generate
        width: Image width in pixels
        height: Image height in pixels
        max_iter: Maximum iterations for Mandelbrot calculation
        output_dir: Directory to save images and metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} images in {output_dir}")
    
    for i in range(num_samples):
        # Progress update, sometimes takes long so nice to see progress
        if i % 100 == 0:
            print(f"Generated {i}/{num_samples} images")
            
        # Randomly select center and zoom
        # More focused sampling around interesting regions
        if np.random.random() < 0.7: # make it 70% chance of interesting regions (aka around the main cardiod)
            # maybe there is a better way to do this not sure
            center = complex(
                np.random.uniform(-0.7, 0.3),
                np.random.uniform(-0.5, 0.5)
            )
            zoom = np.random.uniform(0.5, 3.0)
        else:
            # Wider exploration
            center = complex(
                np.random.uniform(-2, 1),
                np.random.uniform(-1.5, 1.5)
            )
            zoom = np.random.uniform(0.1, 2.0)
        
        # Generate Mandelbrot set
        mandel = mandelbrot_zoom(height, width, max_iter, center, zoom)
        
        # Save image
        plt.imsave(
            output_dir / f"mandelbrot_{i:04d}.png",  # Zero-padded numbering
            mandel,
            cmap='magma'
        )
        
        # Save metadata
        with open(output_dir / f"mandelbrot_{i:04d}.txt", 'w') as f:
            f.write(f"Center: {center}\nZoom: {zoom}\n")

def main():
    config = {
        'num_samples': 1000,  # Generate 1000 images
        'width': 512,
        'height': 512,
        'max_iter': 100,
        'output_dir': 'data/large_dataset'
    }
    
    print("Starting dataset generation with config:")
    for key, value in config.items():
        print(f"{key}: {value}")
        
    generate_dataset(**config)
    print("\nDataset generation done")

if __name__ == "__main__":
    main()

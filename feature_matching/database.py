import numpy as np
from pathlib import Path
import cv2
from .feature_matcher import MandelbrotFeatureMatcher

def create_reference_database(image_dir, matcher, save_path=None):
    """
    Create a database of features from reference images.
    
    Purpose:
    - Process a directory of Mandelbrot set images
    - Extract and store their features
    - Save the relationship between features and coordinates
    
    Why we need this:
    - Feature matching needs a reference database to compare against
    - Each reference image represents a known location in the Mandelbrot set
    - More reference images = better chance of finding good matches
    
    Parameters:
        image_dir: Directory containing reference images and their metadata
        matcher: Instance of MandelbrotFeatureMatcher
        save_path: Optional path to save the database
        
    Returns:
        matcher: The matcher object with populated database
    """
    # Convert to Path object for easier file handling
    image_dir = Path(image_dir)
    
    # Process all PNG images in directory
    for img_path in image_dir.glob('*.png'):
        # Load image in grayscale
        # Why grayscale?
        # - Feature detection works better on grayscale images
        # - Reduces memory usage
        # - Faster processing
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        # Look for corresponding text file with coordinates
        coord_path = img_path.with_suffix('.txt')
        if coord_path.exists():
            with open(coord_path, 'r') as f:
                lines = f.readlines()
                # Parse complex number from string
                center = complex(lines[0].split(': ')[1].strip())
                # Parse zoom level
                zoom = float(lines[1].split(': ')[1].strip())
                
            # Store coordinates in dictionary format
            coordinates = {
                'x': center.real,      # Real part of complex number
                'y': center.imag,      # Imaginary part
                'zoom': zoom           # Zoom level
            }
            
            # Add image and its coordinates to database
            matcher.add_reference_image(image, coordinates)
    
    # Save database if path provided
    # Useful for:
    # - Avoiding recomputing features
    # - Sharing database between sessions
    if save_path:
        matcher.save_database(save_path)
        
    return matcher

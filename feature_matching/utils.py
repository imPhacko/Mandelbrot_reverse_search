import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_matches(query_image, reference_image, query_kp, ref_kp, matches):
    """
    Draw matching features between two images.
    
    Purpose:
    - Visualize how features in query image match reference image
    - Useful for debugging and understanding matching results
    - Helps identify why matches might be good or bad
    
    What it shows:
    - Lines connecting matching features between images
    - Location of keypoints in both images
    - How well the features align
    
    Parameters:
        query_image: The image we're trying to locate
        reference_image: The matching reference image
        query_kp: Keypoints from query image
        ref_kp: Keypoints from reference image
        matches: List of matching keypoint pairs
        
    Returns:
        visualization: Image showing the matches between images
    """
    return cv2.drawMatches(
        query_image, query_kp,          # First image and its keypoints
        reference_image, ref_kp,        # Second image and its keypoints
        matches, None,                  # Matches and output image
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS  # Only draw matching points
    )

def visualize_match(query_image, matched_coordinates, full_mandelbrot):
    """
    Visualize where the matched image is located in the full Mandelbrot set.
    
    Purpose:
    - Show the location of a matched image in context
    - Help verify if the match makes sense
    - Provide visual feedback for the matching process
    
    TODO: Implementation details
    - Could show full Mandelbrot set with highlighted region
    - Could show side-by-side comparison
    - Could add zoom animation between views
    
    Parameters:
        query_image: The image we located
        matched_coordinates: Where it was found (x, y, zoom)
        full_mandelbrot: Full Mandelbrot set image for reference
    """
    # This is a placeholder for future implementation
    # Ideas for visualization:
    # 1. Draw rectangle or circle around matched region
    # 2. Show arrows pointing to location
    # 3. Create split view showing both full set and zoomed region
    pass

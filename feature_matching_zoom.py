### MANDELBROT ###
### Kristijonas and Julio pair project Artificial Intelligence Course. 

## CNN

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
from pathlib import Path
import cv2
from feature_matching.feature_matcher import MandelbrotFeatureMatcher
from src.mandelbrot import mandelbrot_zoom  # Reusing the mandelbrot generation function

image_size = 480
number_samples = 1000

############--------------############
#            Mandelbrot              #
############--------------############
def mandelbrot(h, w, max_iter, x_min, x_max, y_min, y_max):

    y, x = np.ogrid[y_min:y_max:h*1j, x_min:x_max:w*1j]
    c = x + y*1j
    z = c
    divtime = max_iter + np.zeros(z.shape, dtype=int)
    
    for i in range(max_iter):
        z = z**2 + c
        diverge = z * np.conj(z) > 2**2
        div_now = diverge & (divtime == max_iter)
        divtime[div_now] = i
        z[diverge] = 2
        
    return divtime

############--------------############
#            Neural Network          #
############--------------############

### 2. Train the NN. 
### Using tensorflow. 


############--------------############
#            ZOOM FUNCTION           #
############--------------############

#### Zoom function

def zoom_with_feature_matching(snippet, matcher, full_width=800, full_height=600, 
                             zoom_steps=100, gif_filename="feature_matching_zoom.gif"):
    """
    Create a zoom animation using feature matching predictions.
    
    Args:
        snippet: Input image (PIL Image)
        matcher: Initialized MandelbrotFeatureMatcher with loaded database
        full_width, full_height: Output image dimensions
        zoom_steps: Number of frames in the animation
        gif_filename: Output file name
    """
    # Convert PIL image to cv2 format and find location
    cv_image = cv2.cvtColor(np.array(snippet), cv2.COLOR_RGB2BGR)
    pred_coords, match_count = matcher.find_location(cv_image)
    
    if pred_coords is None:
        print("No match found!")
        return
        
    print(f"Found location with {match_count} matches")
    print(f"Predicted coordinates: {pred_coords}")
    
    # Starting coordinates (full Mandelbrot view)
    x_start, x_end = -2, 0.8
    y_start, y_end = -1.4, 1.4
    
    # Target coordinates from prediction
    x_target = pred_coords['x']
    y_target = pred_coords['y']
    zoom_target = pred_coords['zoom']
    
    frames = []
    
    for step in range(zoom_steps):
        # Calculate interpolated coordinates
        progress = step / zoom_steps
        
        # Interpolate position and zoom
        x_min = x_start + (x_target - x_start) * progress
        x_max = x_end + (x_target + 1/zoom_target - x_end) * progress
        y_min = y_start + (y_target - y_start) * progress
        y_max = y_end + (y_target + 1/zoom_target - y_end) * progress
        
        # Generate Mandelbrot image for this step
        mandel = mandelbrot_zoom(full_height, full_width, 100, 
                               complex(x_min + (x_max-x_min)/2, 
                                     y_min + (y_max-y_min)/2),
                               1/((x_max-x_min)/2))
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(mandel, cmap='magma', extent=[x_min, x_max, y_min, y_max])
        ax.set_title(f'Zoom Step {step + 1}/{zoom_steps}')
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        plt.colorbar(ax.imshow(mandel, cmap='magma'), label='Iteration count')
        
        # Capture the plot as an image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        frames.append(image)
        plt.close(fig)
        
        if step % 10 == 0:
            print(f"Generated frame {step + 1}/{zoom_steps}")
    
    # Save as GIF
    print(f"Saving animation to {gif_filename}")
    imageio.mimsave(gif_filename, frames, duration=0.1)
    print("Animation saved!")

def main():
    # Load the feature matcher and database
    matcher = MandelbrotFeatureMatcher(use_sift=True)
    matcher.load_database('feature_matching/database.pkl')
    
    # Load input image
    input_image = Image.open("search_mandelbrot.png").convert('RGB')
    
    # Generate zoom animation
    zoom_with_feature_matching(input_image, matcher)

if __name__ == "__main__":
    main()
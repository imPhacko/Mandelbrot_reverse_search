import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def mandelbrot_zoom(h, w, max_iter, center, zoom_width):
    """
    Generate the Mandelbrot set for a specific zoom level.
    """
    y, x = np.ogrid[
        center.imag - zoom_width/2:center.imag + zoom_width/2:h*1j,
        center.real - zoom_width:center.real + zoom_width:w*1j
    ]
    c = x + y*1j
    z = c
    divtime = max_iter + np.zeros(z.shape, dtype=int)
    
    for i in range(max_iter):
        z = z**2 + c
        diverge = z*np.conj(z) > 2**2
        div_now = diverge & (divtime == max_iter)
        divtime[div_now] = i
        z[diverge] = 2
        
    return divtime

def pixel_to_complex(x, y, width, height, x_min, x_max, y_min, y_max):
    """
    Convert pixel coordinates to a complex number in the Mandelbrot set.
    """
    real = x_min + (x / width) * (x_max - x_min)
    imag = y_min + (y / height) * (y_max - y_min)
    return complex(real, imag)

def plot_mandelbrot(center, zoom, width, height, max_iter):
    """
    Plot the Mandelbrot set for a given center and zoom level.
    """
    x_min, x_max = center.real - zoom, center.real + zoom
    y_min, y_max = center.imag - zoom/2, center.imag + zoom/2
    mandel = mandelbrot_zoom(height, width, max_iter, center, zoom)
    plt.imshow(mandel, cmap='magma', extent=[x_min, x_max, y_min, y_max])
    plt.colorbar(label='Iteration count')
    plt.title(f'Mandelbrot Set at {center} with zoom {zoom}')
    plt.show()

def compare_images(input_image, center, zoom, width, height, max_iter):
    """
    Compare an input image with a generated Mandelbrot set.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot input image
    axes[0].imshow(input_image, cmap='magma')
    axes[0].set_title('Input Image')
    
    # Plot generated Mandelbrot set
    x_min, x_max = center.real - zoom, center.real + zoom
    y_min, y_max = center.imag - zoom/2, center.imag + zoom/2
    mandel = mandelbrot_zoom(height, width, max_iter, center, zoom)
    axes[1].imshow(mandel, cmap='magma', extent=[x_min, x_max, y_min, y_max])
    axes[1].set_title('Generated Mandelbrot Set')
    
    plt.show()

# Example usage
center = -0.7435 + 0.1314j  # Example center point
zoom = 0.1  # Example zoom level
width, height = 800, 600
max_iterations = 100

# Plot the Mandelbrot set
plot_mandelbrot(center, zoom, width, height, max_iterations)

# Example input image (replace with actual image data)
input_image = np.random.rand(height, width)  # Placeholder for an actual image
compare_images(input_image, center, zoom, width, height, max_iterations)
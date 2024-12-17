import numpy as np

def mandelbrot_zoom(h, w, max_iter, center, zoom_width):
    """Generate the Mandelbrot set for a specific zoom level."""
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
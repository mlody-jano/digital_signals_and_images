import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy import fftpack
from scipy.linalg import hadamard
import pywt

###################################
###         1. Utility          ###
###################################

def add_poisson_noise(image, scale = 1.0): # Adding Poisson Noise
    """Add Poisson noise to an image."""
    lam = image * scale
    noisy_image = np.random.poisson(lam).astype(np.float32) / scale
    return noisy_image

###################################
###     2. Transformations      ###
###################################

###################################
###        2.1 Anscombe         ###
###################################

def anscombe_forward(x): # forward Anscombe Transformation X -> A
    """Apply the Anscombe forward transformation."""
    return 2.0 * np.sqrt(x + 3.0/8.0)

def anscombe_inverse(a): # inverse Anscombe Transformation A -> X (approx, does not include the factor of lambda value)
    """Apply the Anscombe inverse transformation."""
    return 1 / 0.25 * np.pow(a, 2) - 1.0/8.0

def anscombe_inverse_lambda(a): # inverse Anscombe Transformation A -> X with lambda correction (includes the value of lambda)
    """Apply the Anscombe inverse transformation with lambda correction."""
    return 1 / 0.25 * np.power(a, 2) - 1.0/8.0 + 1.0/4.0 * np.sqrt(3.0/2.0) * 1.0 / a - 11.0/8.0 * np.pow(a, -2) + 5.0/8.0 * np.sqrt(3.0/2.0) * np.pow(a, -3)

###################################
###         2.2 Cosine          ###
###################################

def dct2d(block): # 2D DCT (Discrete Cosine Transform)
    """Perform a 2D Discrete Cosine Transform."""
    return fftpack.dct(fftpack.dct(block.T, norm='ortho').T, norm='ortho')

def idct2d(block): # 2D inverse DCT
    """Perform a 2D Inverse Discrete Cosine Transform."""
    return fftpack.idct(fftpack.idct(block.T, norm='ortho').T, norm='ortho')

###################################
###     2.3 Walsh-Hadamard      ###
###################################

def wht2(block): # 2D Walsh-Hadamard Transform
    """Perform a 2D Walsh-Hadamard Transform."""
    H = hadamard(block.shape[0])
    return H @ block @ H / block.shape[0]

def iwht2(block): # 2D inverse Walsh-Hadamard Transform
    """Perform a 2D Inverse Walsh-Hadamard Transform."""
    H = hadamard(block.shape[0])
    return H @ block @ H / block.shape[0]

###################################
###        2.4 Wavelet          ###
###################################

def wavelet_forward(img, wavelet = 'haar', level=None):
    """Perform a 2D Wavelet Transform."""
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
    arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    return arr, coeff_slices

def wavelet_inverse(arr, coeff_slices, wavelet = 'haar'):
    """Perform a 2D Inverse Wavelet Transform."""
    coeffs = pywt.array_to_coeffs(arr, coeff_slices, output_format='wavedec2')
    return pywt.waverec2(coeffs, wavelet=wavelet)

###################################
###       3. Thresholding       ###
###################################

def hard_threshold(arr, t):
    """Apply hard thresholding to an array."""
    out = arr.copy()
    out[np.abs(out) <= t] = 0.0
    return out

def quantize(arr, q):
    """Quantize an array."""
    if q == 0:
        return arr
    return np.floor(arr * q + 0.5) / float(q)
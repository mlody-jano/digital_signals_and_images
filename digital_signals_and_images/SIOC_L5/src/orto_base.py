import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import fftpack
import pywt

###################################
###         1. Utility          ###
###################################

def add_poisson_noise(image, scale = 1.0):
    """Add Poisson noise to an image."""
    lam = image * scale
    noisy_image = np.random.poisson(lam).astype(np.float32) / scale
    return noisy_image

def mse(a, b):
    """Mean Squared Error."""
    return np.mean((a - b) ** 2)

def mae(a, b):
    """Mean Absolute Error."""
    return np.mean(np.abs(a - b))

def psnr(a, b):
    """Peak Signal-to-Noise Ratio in dB."""
    m = mse(a, b)
    if m < 1e-10:
        return 100.0
    return 10 * np.log10((255.0 ** 2) / m)

def plot_error_summary(mse_vals, mae_vals, psnr_vals, labels=[], title_suffix=""):
    groups = ["MSE", "MAE", "PSNR"]
    data = [mse_vals, mae_vals, psnr_vals]

    x = np.arange(len(labels))
    group_width = 0.8
    bar_width = group_width / len(groups)

    plt.figure(figsize=(12, 6))
    plt.title(f"Podsumowanie błędów {title_suffix}", fontsize=14)

    for i, (group_name, group_data) in enumerate(zip(groups, data)):
        offset = (i - 1) * bar_width
        bars = plt.bar(x + offset, group_data, bar_width, label=group_name)

        for bar in bars:
            h = bar.get_height()
            plt.annotate(f"{h:.2f}",
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3),
                         textcoords='offset points',
                         ha='center', va='bottom', fontsize=8)

    plt.xticks(x, labels)
    plt.ylabel("Wartości błędów")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)

###################################
###     2. Transformations      ###
###################################

def anscombe_forward(x):
    """Apply the Anscombe forward transformation."""
    return 2.0 * np.sqrt(np.maximum(x, 0) + 3.0/8.0)

def anscombe_inverse_lambda(a):
    """Apply the Anscombe inverse transformation with lambda correction."""
    a_safe = np.maximum(a, 0.5)
    result = (0.25 * np.power(a_safe, 2) - 1.0/8.0 + 
              0.25 * np.sqrt(3.0/2.0) / a_safe - 
              11.0/8.0 / np.power(a_safe, 2) + 
              5.0/8.0 * np.sqrt(3.0/2.0) / np.power(a_safe, 3))
    return np.maximum(result, 0)

###################################
###         DCT & WHT           ###
###################################

def process_blocks(image, block_size, forward_func, inverse_func, modify_func):
    """Process image in blocks with given transform functions."""
    h, w = image.shape
    
    # Pad image to be divisible by block_size
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    
    if pad_h > 0 or pad_w > 0:
        image_padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    else:
        image_padded = image
    
    output_padded = np.zeros_like(image_padded)
    
    for i in range(0, image_padded.shape[0], block_size):
        for j in range(0, image_padded.shape[1], block_size):
            block = image_padded[i:i+block_size, j:j+block_size]
            coeffs = forward_func(block)
            coeffs_modified = modify_func(coeffs)
            reconstructed = inverse_func(coeffs_modified)
            output_padded[i:i+block_size, j:j+block_size] = reconstructed
    
    return output_padded[:h, :w]

def dct2d(block):
    """Perform a 2D Discrete Cosine Transform."""
    return fftpack.dct(fftpack.dct(block.T, norm='ortho').T, norm='ortho')

def idct2d(block):
    """Perform a 2D Inverse Discrete Cosine Transform."""
    return fftpack.idct(fftpack.idct(block.T, norm='ortho').T, norm='ortho')

def wht2(block):
    """Perform a 2D Walsh-Hadamard Transform."""
    n = block.shape[0]
    
    def hadamard_matrix(size):
        if size == 1:
            return np.array([[1.0]])
        else:
            H_half = hadamard_matrix(size // 2)
            return np.block([[H_half, H_half], [H_half, -H_half]])
    
    H = hadamard_matrix(n).astype(np.float32)
    return H @ block @ H / n

def iwht2(block):
    """Perform a 2D Inverse Walsh-Hadamard Transform."""
    return wht2(block) 

###################################
###        Wavelet              ###
###################################

def wavelet_forward(img, wavelet='haar', level=None):
    """Perform a 2D Wavelet Transform."""
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
    arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    return arr, coeff_slices

def wavelet_inverse(arr, coeff_slices, wavelet='haar'):
    """Perform a 2D Inverse Wavelet Transform."""
    coeffs = pywt.array_to_coeffs(arr, coeff_slices, output_format='wavedec2')
    return pywt.waverec2(coeffs, wavelet=wavelet)

###################################
###    Coefficient Processing   ###
###################################

def hard_threshold(arr, t):
    """Apply hard thresholding - zero out coefficients below threshold."""
    out = arr.copy()
    out[np.abs(out) <= t] = 0.0
    return out

def quantize(arr, step_size):
    """
    Uniform scalar quantization.
    step_size: quantization step (smaller = less quantization, more quality)
    
    Formula from JPEG 2000: Q(c) = sign(c) * floor(|c| / Δ) * Δ
    where Δ is the step size
    """
    if step_size <= 0:
        return arr
    
    # Quantize: divide by step, round to integer, multiply back
    sign = np.sign(arr)
    quantized = np.floor(np.abs(arr) / step_size) * step_size * sign
    return quantized

###################################
###       Main Pipeline         ###
###################################

if __name__ == "__main__":
    original = np.array(cv2.imread('images/lena_img.png', cv2.IMREAD_GRAYSCALE)).astype(np.float32)
    
    # Parametry dostosowane do współczynników po Anscombe (zakres ~0-32)
    # Dla kwantyzacji: mniejszy step_size = lepsza jakość
    # Dla progowania: mniejszy próg = więcej zachowanych współczynników
    
    Q_step_sizes = [0.5, 1.0, 2.0]  # Kroki kwantyzacji (JPEG 2000 style)
    T_thresholds = [0.5, 1.0, 2.0]  # Progi hard thresholding
    
    # Wszystkie transformaty
    Transforms = {
        'DCT': None,
        'WHT': None,
        'FWT-Haar': 'haar',
        'FWT-Bior 5.3': 'bior2.2',
        'FWT-Bior 9.7': 'bior4.4'
    }
    
    BLOCK_SIZE = 8

    for transform_name, wavelet_type in Transforms.items():
        print(f"\n{'='*70}")
        print(f"Processing with {transform_name} transform")
        print(f"{'='*70}")
        
        # Generate noisy image
        np.random.seed(42)  # For reproducibility
        image_noisy = add_poisson_noise(original, scale=1.0)
        psnr_noisy = psnr(original, image_noisy)
        
        # Anscombe forward transform
        image_anscombe = anscombe_forward(image_noisy)
        
        print(f"Original image range: [{original.min():.1f}, {original.max():.1f}]")
        print(f"Noisy image PSNR: {psnr_noisy:.2f} dB (baseline)")
        print(f"After Anscombe range: [{image_anscombe.min():.2f}, {image_anscombe.max():.2f}]")
        
        results_quant = []
        results_thresh = []
        stats_quant = []
        stats_thresh = []
        
        # ===== QUANTIZATION =====
        print(f"\n--- Quantization ---")
        for step in Q_step_sizes:
            if transform_name in ['DCT', 'WHT']:
                forward_func = dct2d if transform_name == 'DCT' else wht2
                inverse_func = idct2d if transform_name == 'DCT' else iwht2
                
                img_reconstructed = process_blocks(
                    image_anscombe, 
                    BLOCK_SIZE,
                    forward_func,
                    inverse_func,
                    lambda coeffs: quantize(coeffs, step)
                )
                
            else:  # Wavelet
                coeffs, slices = wavelet_forward(image_anscombe, wavelet=wavelet_type)
                
                if step == Q_step_sizes[0]:
                    print(f"  Wavelet coeffs range: [{coeffs.min():.2f}, {coeffs.max():.2f}]")
                    print(f"  Wavelet coeffs std: {coeffs.std():.2f}")
                
                coeffs_q = quantize(coeffs, step)
                nonzero = np.count_nonzero(coeffs_q)
                total = coeffs_q.size
                
                img_reconstructed = wavelet_inverse(coeffs_q, slices, wavelet=wavelet_type)
            
            img_final = anscombe_inverse_lambda(img_reconstructed)
            img_final = np.clip(img_final, 0, 255)
            
            results_quant.append(img_final)
            
            psnr_val = psnr(original, img_final)
            mse_val = mse(original, img_final)
            improvement = psnr_val - psnr_noisy
            
            if transform_name.startswith('FWT'):
                print(f"  Step={step:.1f}: PSNR={psnr_val:.2f} dB (Δ={improvement:+.2f}), "
                      f"MSE={mse_val:.2f}, Nonzero={nonzero}/{total} ({100*nonzero/total:.1f}%)")
            else:
                print(f"  Step={step:.1f}: PSNR={psnr_val:.2f} dB (Δ={improvement:+.2f}), MSE={mse_val:.2f}")
        
        # ===== THRESHOLDING =====
        print(f"\n--- Thresholding ---")
        for thresh in T_thresholds:
            if transform_name in ['DCT', 'WHT']:
                forward_func = dct2d if transform_name == 'DCT' else wht2
                inverse_func = idct2d if transform_name == 'DCT' else iwht2
                
                img_reconstructed = process_blocks(
                    image_anscombe,
                    BLOCK_SIZE,
                    forward_func,
                    inverse_func,
                    lambda coeffs: hard_threshold(coeffs, thresh)
                )
                
            else:  # Wavelet
                coeffs, slices = wavelet_forward(image_anscombe, wavelet=wavelet_type)
                coeffs_t = hard_threshold(coeffs, thresh)
                
                nonzero = np.count_nonzero(coeffs_t)
                total = coeffs_t.size
                
                img_reconstructed = wavelet_inverse(coeffs_t, slices, wavelet=wavelet_type)
            
            img_final = anscombe_inverse_lambda(img_reconstructed)
            img_final = np.clip(img_final, 0, 255)
            
            results_thresh.append(img_final)
            
            psnr_val = psnr(original, img_final)
            mse_val = mse(original, img_final)
            improvement = psnr_val - psnr_noisy
            
            if transform_name.startswith('FWT'):
                print(f"  Thresh={thresh:.1f}: PSNR={psnr_val:.2f} dB (Δ={improvement:+.2f}), "
                      f"MSE={mse_val:.2f}, Nonzero={nonzero}/{total} ({100*nonzero/total:.1f}%)")
            else:
                print(f"  Thresh={thresh:.1f}: PSNR={psnr_val:.2f} dB (Δ={improvement:+.2f}), MSE={mse_val:.2f}")
        
        # Calculate metrics
        mse_vals_quant = [mse(original, img) for img in results_quant]
        mae_vals_quant = [mae(original, img) for img in results_quant]
        psnr_vals_quant = [psnr(original, img) for img in results_quant]
        
        mse_vals_thresh = [mse(original, img) for img in results_thresh]
        mae_vals_thresh = [mae(original, img) for img in results_thresh]
        psnr_vals_thresh = [psnr(original, img) for img in results_thresh]
        
        # Display results
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Denoising results → {transform_name}', fontsize=16, fontweight='bold')
        
        axes[0, 0].imshow(original, cmap='gray', vmin=0, vmax=255)
        axes[0, 0].set_title('Original', fontsize=11, fontweight='bold')
        axes[0, 0].axis('off')
        
        for i, (img, step) in enumerate(zip(results_quant, Q_step_sizes)):
            axes[0, i+1].imshow(img, cmap='gray', vmin=0, vmax=255)
            delta = psnr_vals_quant[i] - psnr_noisy
            axes[0, i+1].set_title(
                f'Quant Δ={step:.1f}\nPSNR={psnr_vals_quant[i]:.2f} dB\n(Δ={delta:+.2f})',
                fontsize=10
            )
            axes[0, i+1].axis('off')
        
        axes[1, 0].imshow(image_noisy, cmap='gray', vmin=0, vmax=255)
        axes[1, 0].set_title(f'Noisy\nPSNR={psnr_noisy:.2f} dB', fontsize=11, fontweight='bold')
        axes[1, 0].axis('off')
        
        for i, (img, thresh) in enumerate(zip(results_thresh, T_thresholds)):
            axes[1, i+1].imshow(img, cmap='gray', vmin=0, vmax=255)
            delta = psnr_vals_thresh[i] - psnr_noisy
            axes[1, i+1].set_title(
                f'Thresh T={thresh:.1f}\nPSNR={psnr_vals_thresh[i]:.2f} dB\n(Δ={delta:+.2f})',
                fontsize=10
            )
            axes[1, i+1].axis('off')
        
        plt.tight_layout()
        plt.show(block=False)
        
        # Plot metrics
        plot_error_summary(
            mse_vals_quant, mae_vals_quant, psnr_vals_quant,
            labels=[f'Δ={s:.1f}' for s in Q_step_sizes],
            title_suffix=f"- Quantization ({transform_name})"
        )
        
        plot_error_summary(
            mse_vals_thresh, mae_vals_thresh, psnr_vals_thresh,
            labels=[f'T={t:.1f}' for t in T_thresholds],
            title_suffix=f"- Thresholding ({transform_name})"
        )
    
    plt.show()
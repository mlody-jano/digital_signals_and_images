import numpy as np
import cv2
import matplotlib.pyplot as plt

# ===============================================================
# 1. Kernele interpolujące
# ===============================================================

def NN(x):
    return (np.abs(x) < 0.5).astype(float)

def Linear(x):
    return np.maximum(1- np.abs(x), 0)

def Cubic(x):
    x = np.abs(x)
    u = -0.5  # Mitchell–Netravali parameter

    res = np.zeros_like(x)

    mask1 = x < 1
    res[mask1] = ((u + 2)*x[mask1]**3 - (u + 3)*x[mask1]**2 + 1)

    mask2 = (x >= 1) & (x < 2)
    res[mask2] = (u*x[mask2]**3 - 5*u*x[mask2]**2 + 8*u*x[mask2] - 4*u)

    return res

# ===============================================================
# 2. Generowanie zpróbkowanego kernela 1D
# ===============================================================

def generate_sized_kernel(func, K):
    xs = np.linspace(-2, 2, K)

    temp = func(xs)
    temp /= temp.sum()
    return temp

# ===============================================================
# 3. Wygładzanie - Splot 2D
# ===============================================================

def smoothing(img, kernel):
    k = kernel.astype(np.float32)

    if img.ndim == 2:
        img32 = img.astype(np.float32)
        out = cv2.sepFilter2D(img32, -1, k, k, borderType=cv2.BORDER_REFLECT_101)
        return out

    elif img.ndim == 3:
        img32 = img.astype(np.float32)
        out = np.zeros_like(img32)
        for c in range(img32.shape[2]):
            out[..., c] = cv2.sepFilter2D(img32[..., c], -1, k, k,
                                          borderType=cv2.BORDER_REFLECT_101)
        return out

    else:
        raise ValueError("Obraz musi mieć 2 lub 3 wymiary!")

# ===============================================================
# 4. SKALOWANIE
# ===============================================================

def downscale(image, new_size, interp=cv2.INTER_AREA):
    return cv2.resize(image, new_size, interpolation=interp)

def upscale(image, new_size, interp=cv2.INTER_AREA):
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

# ===============================================================
# 5. METRYKI JAKOŚCI
# ===============================================================

def mse(a, b):
    return (np.mean((a - b) ** 2))

def mae(a, b):
    return np.mean(np.abs(a - b))

def psnr(a, b):
    m = mse(a, b)
    if m == 0:
        return 99.0
    return 20 * np.log10(255.0 / np.sqrt(m))

# ===============================================================
# 6. SZUM POISSONA
# ===============================================================

def poisson_noise(image, eta):
    return (np.random.poisson(image / eta) * eta).astype(image.dtype)

# ===============================================================
# 7. Podsumowanie statystyk
# ===============================================================

def plot_error_summary(K, mse_vals, mae_vals, psnr_vals):
    labels = ["NN", "Linear", "Cubic"]

    groups = ["MSE", "MAE", "PSNR"]
    data = [mse_vals, mae_vals, psnr_vals]

    x = np.arange(len(labels))  # NN / Linear / Cubic
    group_width = 0.8
    bar_width = group_width / len(groups)

    plt.figure(figsize=(12, 6))
    plt.title(f"Podsumowanie błędów dla K={K}")

    for i, (group_name, group_data) in enumerate(zip(groups, data)):
        # przesunięcie każdej grupy błędów
        offset = (i - 1) * bar_width
        bars = plt.bar(x + offset, group_data, bar_width, label=group_name)

        # wartości nad słupkami
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
    plt.show(block=True)

# ===============================================================
# 8. GŁÓWNY PROGRAM
# ===============================================================

if __name__ == "__main__":

    original = np.array(cv2.imread("images/test_image.png", cv2.IMREAD_GRAYSCALE)).astype(np.float32)
    if original.shape[0] != 1024 or original.shape[1] != 1024:
        raise ValueError("Obraz musi mieć rozmiar 1024x1024!")
    image = original.copy()

    K_values = [6, 12, 24, 48]
    p_values = [7, 8]
    eta_values = [1,4,16,64,256]
    k_values = [1,2,3, ...]

    # OPERACJE BEZ SZUMU

    for K in K_values:
        print(f"Wykonywanie operacji dla K = {K}, bez szumu.")

        kernel_NN = generate_sized_kernel(NN, K)
        kernel_linear = generate_sized_kernel(Linear, K)
        kernel_cubic = generate_sized_kernel(Cubic, K)

        smoothed_NN = smoothing(image, kernel_NN).astype(np.float32)
        smoothed_linear = smoothing(image, kernel_linear).astype(np.float32)
        smoothed_cubic = smoothing(image, kernel_cubic).astype(np.float32)

        cv2.imwrite(f"images/no_noise/smoothed/smoothed_NN_K_{K}.png", smoothed_NN)
        cv2.imwrite(f"images/no_noise/smoothed/smoothed_linear_K_{K}.png", smoothed_linear)
        cv2.imwrite(f"images/no_noise/smoothed/smoothed_cubic_K_{K}.png", smoothed_cubic)

        downscaled_NN_256 = downscale(smoothed_NN, (256, 256)).astype(np.float32)
        downscaled_linear_256 = downscale(smoothed_linear, (256, 256)).astype(np.float32)
        downscaled_cubic_256 = downscale(smoothed_cubic, (256, 256)).astype(np.float32)

        cv2.imwrite(f"images/no_noise/downscaled/256/downscaled_NN_K_{K}.png", downscaled_NN_256)
        cv2.imwrite(f"images/no_noise/downscaled/256/downscaled_linear_K_{K}.png", downscaled_linear_256)
        cv2.imwrite(f"images/no_noise/downscaled/256/downscaled_cubic_K_{K}.png", downscaled_cubic_256)

        downscaled_NN_128 = downscale(smoothed_NN, (128, 128)).astype(np.float32)
        downscaled_linear_128 = downscale(smoothed_linear, (128, 128)).astype(np.float32)
        downscaled_cubic_128 = downscale(smoothed_cubic, (128, 128)).astype(np.float32)

        cv2.imwrite(f"images/no_noise/downscaled/128/downscaled_NN_K_{K}.png", downscaled_NN_128)
        cv2.imwrite(f"images/no_noise/downscaled/128/downscaled_linear_K_{K}.png", downscaled_linear_128)
        cv2.imwrite(f"images/no_noise/downscaled/128/downscaled_cubic_K_{K}.png", downscaled_cubic_128)

        upscaled_NN_256 = upscale(downscaled_NN_256, (1024, 1024), cv2.INTER_NEAREST).astype(np.float32)
        upscaled_linear_256 = upscale(downscaled_linear_256, (1024, 1024), cv2.INTER_LINEAR).astype(np.float32)
        upscaled_cubic_256 = upscale(downscaled_cubic_256, (1024, 1024), cv2.INTER_CUBIC).astype(np.float32)

        cv2.imwrite(f"images/no_noise/upscaled/256/upscaled_NN_K_{K}.png", upscaled_NN_256)
        cv2.imwrite(f"images/no_noise/upscaled/256/upscaled_linear_K_{K}.png", upscaled_linear_256)
        cv2.imwrite(f"images/no_noise/upscaled/256/upscaled_cubic_K_{K}.png", upscaled_cubic_256)

        upscaled_NN_128 = upscale(downscaled_NN_128, (1024, 1024), cv2.INTER_NEAREST).astype(np.float32)
        upscaled_linear_128 = upscale(downscaled_linear_128, (1024, 1024), cv2.INTER_LINEAR).astype(np.float32)
        upscaled_cubic_128 = upscale(downscaled_cubic_128, (1024, 1024), cv2.INTER_CUBIC).astype(np.float32)

        cv2.imwrite(f"images/no_noise/upscaled/128/upscaled_NN_K_{K}.png", upscaled_NN_128)
        cv2.imwrite(f"images/no_noise/upscaled/128/upscaled_linear_K_{K}.png", upscaled_linear_128)
        cv2.imwrite(f"images/no_noise/upscaled/128/upscaled_cubic_K_{K}.png", upscaled_cubic_128)

        print(f"Wykonywanie operacji dla K = {K}, bez szumu zakończone!")

        print(f"Wyniki dla K = {K}:")
        
        mse_NN_256 = mse(original, upscaled_NN_256)
        mse_linear_256 = mse(original, upscaled_linear_256)
        mse_cubic_256 = mse(original, upscaled_cubic_256)

        mae_NN_256 = mae(original, upscaled_NN_256)
        mae_linear_256 = mae(original, upscaled_linear_256)
        mae_cubic_256 = mae(original, upscaled_cubic_256)

        psnr_NN_256 = psnr(original, upscaled_NN_256)
        psnr_linear_256 = psnr(original, upscaled_linear_256)
        psnr_cubic_256 = psnr(original, upscaled_cubic_256)

        print(f"Wartość p = 8 (256x256 -> 1024x1024)")
        print(f"NN:     MSE = {mse_NN_256}, MAE = {mae_NN_256}, PSNR = {psnr_NN_256}")
        print(f"Linear:     MSE = {mse_linear_256}, MAE = {mae_linear_256}, PSNR = {psnr_linear_256}")
        print(f"Cubic:     MSE = {mse_cubic_256}, MAE = {mae_cubic_256}, PSNR = {psnr_cubic_256}")

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(10,5))
        fig.suptitle(f"Obrazy uzyskane podczas przetworzenia 256x256 -> 1024x1024")
        ax1.imshow(original, cmap='gray')
        ax2.imshow(upscaled_NN_256, cmap='gray')
        ax3.imshow(upscaled_linear_256, cmap='gray')
        ax4.imshow(upscaled_cubic_256, cmap='gray')
        ax1.set_title('Oryginał')
        ax1.axis('off')
        ax2.set_title('Kernel NN')
        ax2.axis('off')
        ax3.set_title('Kernel liniowy')
        ax3.axis('off')
        ax4.set_title('Kernel sześcienny')
        ax4.axis('off')

        plt.tight_layout()
        plt.show(block=False)

        mse_NN_128 = mse(original, upscaled_NN_128)
        mse_linear_128 = mse(original, upscaled_linear_128)
        mse_cubic_128 = mse(original, upscaled_cubic_128)

        mae_NN_128 = mae(original, upscaled_NN_128)
        mae_linear_128 = mae(original, upscaled_linear_128)
        mae_cubic_128 = mae(original, upscaled_cubic_128)

        psnr_NN_128 = psnr(original, upscaled_NN_128)
        psnr_linear_128 = psnr(original, upscaled_linear_128)
        psnr_cubic_128 = psnr(original, upscaled_cubic_128)

        plot_error_summary(
        K,
        mse_vals=[mse_NN_256, mse_linear_256, mse_cubic_256],
        mae_vals=[mae_NN_256, mae_linear_256, mae_cubic_256],
        psnr_vals=[psnr_NN_256, psnr_linear_256, psnr_cubic_256]
        )

        print(f"Wartość p = 7 (128x128 -> 1024x1024)")
        print(f"NN:     MSE = {mse_NN_128}, MAE = {mae_NN_128}, PSNR = {psnr_NN_128}")
        print(f"Linear:     MSE = {mse_linear_128}, MAE = {mae_linear_128}, PSNR = {psnr_linear_128}")
        print(f"Cubic:     MSE = {mse_cubic_128}, MAE = {mae_cubic_128}, PSNR = {psnr_cubic_128}")

        # Porównanie obrazów

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(10,5))
        fig.suptitle(f"Obrazy uzyskane podczas przetworzenia 128x128 -> 1024x1024")
        ax1.imshow(original, cmap='gray')
        ax2.imshow(upscaled_NN_128, cmap='gray')
        ax3.imshow(upscaled_linear_128, cmap='gray')
        ax4.imshow(upscaled_cubic_128, cmap='gray')
        ax1.set_title('Oryginał')
        ax1.axis('off')
        ax2.set_title('Kernel NN')
        ax2.axis('off')
        ax3.set_title('Kernel liniowy')
        ax3.axis('off')
        ax4.set_title('Kernel sześcienny')
        ax4.axis('off')

        plt.tight_layout()
        plt.show(block=False)

        plot_error_summary(
        K,
        mse_vals=[mse_NN_256, mse_linear_256, mse_cubic_256],
        mae_vals=[mae_NN_256, mae_linear_256, mae_cubic_256],
        psnr_vals=[psnr_NN_256, psnr_linear_256, psnr_cubic_256]
        )

    # OPERACJE NA DOJEBANYM SZUMIE

    for eta in eta_values:
        noisy_image = poisson_noise(image, eta)
        for K in K_values:
            print(f"Wykonywanie operacji dla K = {K}, z szumem o wartości eta = {eta}.")

            kernel_NN = generate_sized_kernel(NN, K)
            kernel_linear = generate_sized_kernel(Linear, K)
            kernel_cubic = generate_sized_kernel(Cubic, K)

            smoothed_NN = smoothing(noisy_image, kernel_NN)
            smoothed_linear = smoothing(noisy_image, kernel_linear)
            smoothed_cubic = smoothing(noisy_image, kernel_cubic)

            cv2.imwrite(f"images/noisy/smoothed/smoothed_NN_K_{K}.png", smoothed_NN)
            cv2.imwrite(f"images/noisy/smoothed/smoothed_linear_K_{K}.png", smoothed_linear)
            cv2.imwrite(f"images/noisy/smoothed/smoothed_cubic_K_{K}.png", smoothed_cubic)

            downscaled_NN_256 = downscale(smoothed_NN, (256, 256))
            downscaled_linear_256 = downscale(smoothed_linear, (256, 256))
            downscaled_cubic_256 = downscale(smoothed_cubic, (256, 256))

            cv2.imwrite(f"images/noisy/downscaled/256/downscaled_NN_K_{K}.png", downscaled_NN_256)
            cv2.imwrite(f"images/noisy/downscaled/256/downscaled_linear_K_{K}.png", downscaled_linear_256)
            cv2.imwrite(f"images/noisy/downscaled/256/downscaled_cubic_K_{K}.png", downscaled_cubic_256)

            downscaled_NN_128 = downscale(smoothed_NN, (128, 128))
            downscaled_linear_128 = downscale(smoothed_linear, (128, 128))
            downscaled_cubic_128 = downscale(smoothed_cubic, (128, 128))

            cv2.imwrite(f"images/noisy/downscaled/128/downscaled_NN_K_{K}.png", downscaled_NN_128)
            cv2.imwrite(f"images/noisy/downscaled/128/downscaled_linear_K_{K}.png", downscaled_linear_128)
            cv2.imwrite(f"images/noisy/downscaled/128/downscaled_cubic_K_{K}.png", downscaled_cubic_128)

            upscaled_NN_256 = upscale(downscaled_NN_256, (1024, 1024), cv2.INTER_NEAREST)
            upscaled_linear_256 = upscale(downscaled_linear_256, (1024, 1024), cv2.INTER_LINEAR)
            upscaled_cubic_256 = upscale(downscaled_cubic_256, (1024, 1024), cv2.INTER_CUBIC)

            cv2.imwrite(f"images/noisy/upscaled/256/upscaled_NN_K_{K}.png", upscaled_NN_256)
            cv2.imwrite(f"images/noisy/upscaled/256/upscaled_linear_K_{K}.png", upscaled_linear_256)
            cv2.imwrite(f"images/noisy/upscaled/256/upscaled_cubic_K_{K}.png", upscaled_cubic_256)

            upscaled_NN_128 = upscale(downscaled_NN_128, (1024, 1024), cv2.INTER_NEAREST)
            upscaled_linear_128 = upscale(downscaled_linear_128, (1024, 1024), cv2.INTER_LINEAR)
            upscaled_cubic_128 = upscale(downscaled_cubic_128, (1024, 1024), cv2.INTER_CUBIC)

            cv2.imwrite(f"images/noisy/upscaled/128/upscaled_NN_K_{K}.png", upscaled_NN_128)
            cv2.imwrite(f"images/noisy/upscaled/128/upscaled_linear_K_{K}.png", upscaled_linear_128)
            cv2.imwrite(f"images/noisy/upscaled/128/upscaled_cubic_K_{K}.png", upscaled_cubic_128)

            print(f"Wykonywanie operacji dla K = {K}, z szumem o wartości eta = {eta} zakończone!")

            print(f"Wyniki dla K = {K}:")
            
            mse_NN_256 = mse(original, upscaled_NN_256)
            mse_linear_256 = mse(original, upscaled_linear_256)
            mse_cubic_256 = mse(original, upscaled_cubic_256)

            mae_NN_256 = mae(original, upscaled_NN_256)
            mae_linear_256 = mae(original, upscaled_linear_256)
            mae_cubic_256 = mae(original, upscaled_cubic_256)

            psnr_NN_256 = psnr(original, upscaled_NN_256)
            psnr_linear_256 = psnr(original, upscaled_linear_256)
            psnr_cubic_256 = psnr(original, upscaled_cubic_256)

            print(f"Wartość p = 8 (256x256 -> 1024x1024)")
            print(f"NN:     MSE = {mse_NN_256}, MAE = {mae_NN_256}, PSNR = {psnr_NN_256}")
            print(f"Linear:     MSE = {mse_linear_256}, MAE = {mae_linear_256}, PSNR = {psnr_linear_256}")
            print(f"Cubic:     MSE = {mse_cubic_256}, MAE = {mae_cubic_256}, PSNR = {psnr_cubic_256}")

            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(10,5))
            fig.suptitle(f"Obrazy uzyskane podczas przetworzenia 256x256 -> 1024x1024")
            ax1.imshow(original, cmap='gray')
            ax2.imshow(upscaled_NN_256, cmap='gray')
            ax3.imshow(upscaled_linear_256, cmap='gray')
            ax4.imshow(upscaled_cubic_256, cmap='gray')
            ax1.set_title('Oryginał')
            ax1.axis('off')
            ax2.set_title('Kernel NN')
            ax2.axis('off')
            ax3.set_title('Kernel liniowy')
            ax3.axis('off')
            ax4.set_title('Kernel sześcienny')
            ax4.axis('off')

            plt.tight_layout()
            plt.show(block=False)

            mse_NN_128 = mse(original, upscaled_NN_128)
            mse_linear_128 = mse(original, upscaled_linear_128)
            mse_cubic_128 = mse(original, upscaled_cubic_128)

            mae_NN_128 = mae(original, upscaled_NN_128)
            mae_linear_128 = mae(original, upscaled_linear_128)
            mae_cubic_128 = mae(original, upscaled_cubic_128)

            psnr_NN_128 = psnr(original, upscaled_NN_128)
            psnr_linear_128 = psnr(original, upscaled_linear_128)
            psnr_cubic_128 = psnr(original, upscaled_cubic_128)

            plot_error_summary(
            K,
            mse_vals=[mse_NN_256, mse_linear_256, mse_cubic_256],
            mae_vals=[mae_NN_256, mae_linear_256, mae_cubic_256],
            psnr_vals=[psnr_NN_256, psnr_linear_256, psnr_cubic_256]
            )

            print(f"Wartość p = 7 (128x128 -> 1024x1024)")
            print(f"NN:     MSE = {mse_NN_128}, MAE = {mae_NN_128}, PSNR = {psnr_NN_128}")
            print(f"Linear:     MSE = {mse_linear_128}, MAE = {mae_linear_128}, PSNR = {psnr_linear_128}")
            print(f"Cubic:     MSE = {mse_cubic_128}, MAE = {mae_cubic_128}, PSNR = {psnr_cubic_128}")

            # Porównanie obrazów

            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(10,5))
            fig.suptitle(f"Obrazy uzyskane podczas przetworzenia 128x128 -> 1024x1024")
            ax1.imshow(original, cmap='gray')
            ax2.imshow(upscaled_NN_128, cmap='gray')
            ax3.imshow(upscaled_linear_128, cmap='gray')
            ax4.imshow(upscaled_cubic_128, cmap='gray')
            ax1.set_title('Oryginał')
            ax1.axis('off')
            ax2.set_title('Kernel NN')
            ax2.axis('off')
            ax3.set_title('Kernel liniowy')
            ax3.axis('off')
            ax4.set_title('Kernel sześcienny')
            ax4.axis('off')

            plt.tight_layout()
            plt.show(block=False)

            plot_error_summary(
            K,
            mse_vals=[mse_NN_256, mse_linear_256, mse_cubic_256],
            mae_vals=[mae_NN_256, mae_linear_256, mae_cubic_256],
            psnr_vals=[psnr_NN_256, psnr_linear_256, psnr_cubic_256]
            )
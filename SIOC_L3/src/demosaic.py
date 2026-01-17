# from scaling import nearest_neighbor, linear_interpolation, cubic_interpolation
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import cv2
import os
from scaling import mse_mae_center_crop

# ================================
# Tworzenie katalogu wynikowego
# ================================
output_dir = "images/demosaic"
os.makedirs(output_dir, exist_ok=True)

# ================================
# Funkcje interpolujące
# ================================

def nearest_neighbor(cfa):
    """Interpolacja najbliższego sąsiada dla demozaikowania"""
    h, w = cfa.shape
    result = cfa.copy()

    for i in range(h):
        for j in range(w):
            if result[i, j] == 0:  # brak koloru => interpolacja
                # znajdź najbliższy niezerowy piksel
                r_min = max(i - 1, 0)
                r_max = min(i + 2, h)
                c_min = max(j - 1, 0)
                c_max = min(j + 2, w)

                window = cfa[r_min:r_max, c_min:c_max]
                nonzero = window[window > 0]

                if nonzero.size > 0:
                    result[i, j] = nonzero.flat[0]

    return result

def linear_interpolation(cfa):
    """Interpolacja bilinearna dla demozaikowania"""
    h, w = cfa.shape
    result = cfa.copy()

    for i in range(h):
        for j in range(w):
            if result[i, j] == 0:
                values = []

                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            if cfa[ni, nj] > 0:
                                values.append(cfa[ni, nj])

                if len(values) > 0:
                    result[i, j] = np.mean(values)

    return result

def cubic_interpolation(cfa):
    """Prosta interpolacja kubiczna: średnia ważona z większego sąsiedztwa"""
    h, w = cfa.shape
    result = cfa.copy()

    for i in range(h):
        for j in range(w):
            if result[i, j] == 0:

                values = []
                weights = []

                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            if cfa[ni, nj] > 0:
                                dist = abs(di) + abs(dj)
                                wgt = 1 / (1 + dist)
                                values.append(cfa[ni, nj] * wgt)
                                weights.append(wgt)

                if len(values) > 0:
                    result[i, j] = sum(values) / sum(weights)

    return result

# ================================
# Funkcje pomocnicze
# ================================

def apply_bayer_mask(image):
    """Zwraca trzy osobne kanały CFA (R, G, B) dla filtra Bayera"""
    pattern = np.array([
        ["G", "R"],
        ["B", "G"]
    ])
    h, w, _ = image.shape
    
    R = np.zeros((h, w))
    G = np.zeros((h, w))
    B = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            color = pattern[i % 2, j % 2]
            if color == "R": R[i, j] = image[i, j, 0]
            elif color == "G": G[i, j] = image[i, j, 1]
            elif color == "B": B[i, j] = image[i, j, 2]

    return R, G, B


def apply_xtrans_mask(image):
    """Zwraca trzy osobne kanały CFA (R, G, B) dla filtra X-Trans"""
    pattern = np.array([
        ["G","B","R","G","R","B"],
        ["R","G","G","B","G","G"],
        ["B","G","G","R","G","G"],
        ["G","R","B","G","B","R"],
        ["B","G","G","R","G","G"],
        ["R","G","G","B","G","G"]
    ])
    h, w, _ = image.shape

    R = np.zeros((h, w))
    G = np.zeros((h, w))
    B = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            color = pattern[i % 6, j % 6]
            if color == "R": R[i, j] = image[i, j, 0]
            elif color == "G": G[i, j] = image[i, j, 1]
            elif color == "B": B[i, j] = image[i, j, 2]

    return R, G, B

def demozaikuj(image, pattern_type="bayer", method="linear"):
    """
    Demozaikowanie przy użyciu własnych funkcji interpolacji:
    method = 'nearest', 'linear', 'cubic'
    """
    start_time = time.time()

    # Tworzymy obraz z CFA (symulacja filtra)
    if pattern_type == "bayer":
        R_cfa, G_cfa, B_cfa = apply_bayer_mask(image)
    else:
        R_cfa, G_cfa, B_cfa = apply_xtrans_mask(image)

    # Wybór interpolatora
    if method == "nearest_neighbor":
        interp = nearest_neighbor
    elif method == "linear":
        interp = linear_interpolation
    elif method == "cubic":
        interp = cubic_interpolation
    else:
        raise ValueError("Nieznana metoda interpolacji")

    # Interpolacja każdego kanału CFA osobno
    R_full = interp(R_cfa)
    G_full = interp(G_cfa)
    B_full = interp(B_cfa)

    # Łączenie w pełny obraz RGB
    demosaiced = np.stack([R_full, G_full, B_full], axis=-1)

    exec_time = time.time() - start_time
    return demosaiced.astype(np.float32), exec_time

# ================================
# Główna część programu
# ================================
if __name__ == "__main__":
    # Wczytaj obraz testowy
    image_path = "images/lena_img.jpg"  # zmień jeśli inna ścieżka
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError("Nie znaleziono pliku testowego w katalogu projektu.")

    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB) / 255.0

    # Przykładowe uruchomienie dla każdej interpolacji
    for method in ["nearest_neighbor","linear", "cubic"]:
        print(f"\n=== DEMOZAIKOWANIE metodą {method.upper()} ===")
        demosaiced_bayer, time_bayer = demozaikuj(original, "bayer", method)
        demosaiced_xtrans, time_xtrans = demozaikuj(original, "xtrans", method)

        # Zapis wyników
        cv2.imwrite(os.path.join(output_dir, f"bayer_{method}.jpg"), cv2.cvtColor((demosaiced_bayer * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"xtrans_{method}.jpg"), cv2.cvtColor((demosaiced_xtrans * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        # Obliczenia jakości
        mse_bayer, mae_bayer = mse_mae_center_crop(original,demosaiced_bayer)
        mse_xtrans, mae_xtrans = mse_mae_center_crop(original, demosaiced_xtrans)

        print(f"Bayer ({method}):  MSE={mse_bayer:.5f}, MAE={mae_bayer:.5f}, czas={time_bayer:.4f}s")
        print(f"X-Trans ({method}): MSE={mse_xtrans:.5f}, MAE={mae_xtrans:.5f}, czas={time_xtrans:.4f}s")

        # Wykres porównawczy
        labels = ['MSE', 'MAE', 'Czas [s]']
        bayer_vals = [mse_bayer, mae_bayer, time_bayer]
        xtrans_vals = [mse_xtrans, mae_xtrans, time_xtrans]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots()
        ax.bar(x - width/2, bayer_vals, width, label='Bayer')
        ax.bar(x + width/2, xtrans_vals, width, label='X-Trans')

        ax.set_ylabel('Wartość')
        ax.set_title('Porównanie jakości demozaikowania')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        # ================================
        # DODANIE WARTOŚCI NAD SŁUPKAMI
        # ================================
        def autolabel(rects, values):
            for rect, val in zip(rects, values):
                height = rect.get_height()
                ax.annotate(f'{val:.4f}',
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),  # offset w pionie
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

        # pobierz słupki wykresu
        rects_bayer = ax.patches[0:3]    # trzy pierwsze słupki
        rects_xtrans = ax.patches[3:6]   # trzy kolejne słupki

        autolabel(rects_bayer, bayer_vals)
        autolabel(rects_xtrans, xtrans_vals)

        plt.tight_layout()
        plt.show(block=False)

        # Porównanie dla maski Bayera

        R_bayer, G_bayer, B_bayer = apply_bayer_mask(original)

        h, w = R_bayer.shape
        R_pos = np.zeros((h,w,3))
        R_pos[:,:,0] = R_bayer
        G_pos = np.zeros((h,w,3))
        G_pos[:,:,1] = G_bayer
        B_pos = np.zeros((h,w,3))
        B_pos[:,:,2] = B_bayer

        cv2.imwrite(os.path.join(output_dir, f"R_bayer_{method}.jpg"), cv2.cvtColor((R_pos * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"G_bayer_{method}.jpg"), cv2.cvtColor((G_pos * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"B_bayer_{method}.jpg"), cv2.cvtColor((B_pos * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"mosaic_bayer_{method}.jpg"), cv2.cvtColor(((R_pos + G_pos + B_pos) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,5))
        fig.suptitle(f"Interpolacja {method} -> Kanały po nałozeniu maski Bayera")
        ax1.imshow(R_pos)
        ax2.imshow(G_pos)
        ax3.imshow(B_pos)
        ax1.set_title('Kanał R')
        ax1.axis('off')
        ax2.set_title('Kanał G')
        ax2.axis('off')
        ax3.set_title('Kanał B')
        ax3.axis('off')

        plt.tight_layout()
        plt.show(block=False)

        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,5))
        fig.suptitle(f"Interpolacja {method} -> Porównanie obrazów po nałozeniu maski Bayera")
        ax1.imshow(original)
        ax2.imshow(demosaiced_bayer)
        ax1.set_title('Oryginał')
        ax1.axis('off')
        ax2.set_title('Obraz odtworzony')
        ax2.axis('off')
        ax3.imshow(R_pos + G_pos + B_pos)
        ax3.set_title('Mozaika pikseli')
        ax3.axis('off')

        plt.show(block=True)

        # Porównanie dla maski X-Trans Fuji

        R_fuji, G_fuji, B_fuji = apply_xtrans_mask(original)

        R_pos = np.zeros((h,w,3))
        R_pos[:,:,0] = R_fuji
        G_pos = np.zeros((h,w,3))
        G_pos[:,:,1] = G_fuji
        B_pos = np.zeros((h,w,3))
        B_pos[:,:,2] = B_fuji

        cv2.imwrite(os.path.join(output_dir, f"R_fuji_{method}.jpg"), cv2.cvtColor((R_pos * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"G_fuji_{method}.jpg"), cv2.cvtColor((G_pos * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"B_fuji_{method}.jpg"), cv2.cvtColor((B_pos * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"mosaic_fuji_{method}.jpg"), cv2.cvtColor(((R_pos + G_pos + B_pos) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,5))
        fig.suptitle(f"Interpolacja {method} -> Kanały po nałozeniu maski X-Trans")
        ax1.imshow(R_pos)
        ax2.imshow(G_pos)
        ax3.imshow(B_pos)
        ax1.set_title('Kanał R')
        ax1.axis('off')
        ax2.set_title('Kanał G')
        ax2.axis('off')
        ax3.set_title('Kanał B')
        ax3.axis('off')
        
        plt.tight_layout()
        plt.show(block=False)

        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,5))
        fig.suptitle(f"Interpolacja {method} -> Porównanie obrazów po nałozeniu maski X-Trans")
        ax1.imshow(original)
        ax2.imshow(demosaiced_xtrans)
        ax1.set_title('Oryginał')
        ax1.axis('off')
        ax2.set_title('Obraz odtworzony')
        ax2.axis('off')
        ax3.imshow(R_pos + G_pos + B_pos)
        ax3.set_title('Mozaika pikseli')
        ax3.axis('off')

        plt.show(block=True)

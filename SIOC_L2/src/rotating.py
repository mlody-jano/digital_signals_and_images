import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scaling import nearest_interpolation_pixel, bilinear_interpolation_pixel, bicubic_interpolation_pixel, cubic_weight
import os
import time

# === 1. Ustawienia początkowe ===
INPUT_IMAGE = "images/lena_img.jpg"  # <-- podaj nazwę swojego pliku
OUTPUT_DIR = "images/rotated"
ANGLE_STEP = 12            # obrót co 12 stopni
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 2. Wczytanie i przygotowanie obrazu ===
original_img = Image.open(INPUT_IMAGE).convert("RGB")
original_np = np.asarray(original_img)
original_cmap = 'gray'

time_list = []

# === 2.1 Funkcja obracająca

def rotate_image(img, angle_deg, method, center=None):

    h, w, channels = img.shape
    angle = np.radians(angle_deg)

    if center is None:
        cx, cy = w / 2, h / 2
    else:
        cx, cy = center

    rotated = np.zeros_like(img, dtype=np.float32)

    for y_new in range(h):
        for x_new in range(w):
            # odwrotna transformacja: z obrazu wynikowego do źródłowego
            x = np.cos(angle) * (x_new - cx) + np.sin(angle) * (y_new - cy) + cx
            y = -np.sin(angle) * (x_new - cx) + np.cos(angle) * (y_new - cy) + cy

            if 0 <= x < w and 0 <= y < h:
                for c in range(img.shape[2]):
                    if method == 'nearest':
                        rotated[y_new, x_new, c] = nearest_interpolation_pixel(img[..., c], x, y)
                    elif method == 'bilinear':
                        rotated[y_new, x_new, c] = bilinear_interpolation_pixel(img[..., c], x, y)
                    elif method == 'bicubic':
                        rotated[y_new, x_new, c] = bicubic_interpolation_pixel(img[..., c], x, y)
            else:
                rotated[y_new, x_new, 0] = 0


    return np.clip(rotated, 0, 255).astype(np.uint8)

# === 3. Pętla obrotu ===
rotated = original_np.copy()

for i in range(0, 360, ANGLE_STEP):
    start = time.time()
    rotated = rotate_image(rotated, ANGLE_STEP, "nearest")
    end = time.time()
    time_list.append(end-start)
    angle = i + ANGLE_STEP
    output_path = os.path.join(OUTPUT_DIR, f"rotated_{angle:03d}.png")
    Image.fromarray(rotated).save(output_path)
    print(f"[INFO] Zapisano: {output_path}")

# === 4. Wczytanie obrazu po pełnym obrocie (360°) ===
final_img_path = os.path.join(OUTPUT_DIR, "rotated_360.png")
final_img = Image.open(final_img_path).convert("RGB")

# Dopasowanie rozmiarów (w razie różnicy)
final_img = final_img.resize(original_img.size)
final_np = np.asarray(final_img)

def mse_mae_center_crop(img1, img2):
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    c = min(img1.shape[2], img2.shape[2]) if img1.ndim == 3 and img2.ndim == 3 else 1

    def center_crop(img, h, w):
        start_y = (img.shape[0] - h) // 2
        start_x = (img.shape[1] - w) // 2
        return img[start_y:start_y + h, start_x:start_x + w, :c] if img.ndim == 3 else img[start_y:start_y + h, start_x:start_x + w]

    img1_cropped = center_crop(img1, h, w)
    img2_cropped = center_crop(img2, h, w)

    mse = np.mean((img1_cropped - img2_cropped) ** 2)
    mae = np.mean(np.abs(img1_cropped - img2_cropped))
    return mse, mae

# === 5. Obliczanie błędów MSE i MAE ===
mse, mae = mse_mae_center_crop(original_np, final_np)
print(f"\n--- Wyniki porównania ---")
print(f"MSE (Mean Squared Error): {mse:.6f}")
print(f"MAE (Mean Absolute Error): {mae:.6f}")

# === 6. Wizualizacja wyników ===
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(original_np, original_cmap)
axes[0].set_title("Oryginalny obraz")
axes[0].axis("off")

axes[1].imshow(final_np, original_cmap)
axes[1].set_title("Po obrocie 360°")
axes[1].axis("off")

# Mapa błędu (różnica absolutna)
error_map = np.abs(original_np.astype(np.int16) - final_np.astype(np.int16))
axes[2].imshow(error_map)
axes[2].set_title("Mapa błędu (|oryginał - 360°|)")
axes[2].axis("off")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle("Statystyki interpolacji", fontsize=14)
ax1.bar(["Średni czas pracy"], np.average(time_list))
ax1.bar_label(ax1.bar(["Średni czas pracy"], np.average(time_list)))
ax2.bar(["MSE"], mse)
ax2.bar_label(ax2.bar(["MSE"], mse))
ax3.bar(["MAE"], mae)
ax3.bar_label(ax3.bar(["MAE"], mae))
plt.show(block=False)

if __name__ == ("__main__"):
    plt.tight_layout()
    plt.show()
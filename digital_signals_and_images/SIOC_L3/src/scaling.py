import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.widgets import Button
import time

# konfiguracja
global scale
fig, ax = plt.subplots()
axis_color = 'lightgoldenrodyellow'
path = "images/lena_img.jpg"
scale = 1

def load_image(path):
    img_pil = Image.open(path)
    if img_pil.mode == "L":
        img = np.asarray(img_pil)
        cmap = 'gray'
    else:
        img = np.asarray(img_pil.convert("RGB"))
        cmap = None
    return img, cmap

image, cmap = load_image(path)

# liczniki i pomiary czasu
nn_counter = 0
linear_counter = 0
cubic_counter = 0
time_list = []

fig.subplots_adjust(bottom=0.25)

# przycisk "Next"
nextax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
nextbutton = Button(nextax, 'Next', color=axis_color, hovercolor='0.975')

# funkcja porównująca obrazy
def show_summary(scaled_img):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Porównanie obrazów po interpolacji", fontsize=14)
    ax1.imshow(original_img)
    ax1.set_title("Oryginał")
    ax1.axis('off')
    ax2.imshow(scaled_img)
    ax2.set_title("Po skalowaniu")
    ax2.axis('off')
    plt.show(block=False)

def summarize_data():

    time = np.average(time_list)
    mse, mae = mse_mae_center_crop(original_img, current_img)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle("Statystyki interpolacji", fontsize=14)
    ax1.bar(["Średni czas pracy"], time)
    ax1.bar_label(ax1.bar(["Średni czas pracy"], time))
    ax2.bar(["MSE"], mse)
    ax2.bar_label(ax2.bar(["MSE"], mse))
    ax3.bar(["MAE"], mae)
    ax3.bar_label(ax3.bar(["MAE"], mae))
    plt.show(block=False)

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

# funkcja przycisku
def button_clicked(mouse_event):
    global current_img, img_display
    start = time.time()
    temp = linear_interpolation(current_img)
    end = time.time()
    time_list.append(end - start)
    if temp is None:
        return
    img_display.set_data(temp)
    current_img = temp
    plt.draw()

nextbutton.on_clicked(button_clicked)

# nearest neighbor interpolation
def nearest_neighbor(image):
    global nn_counter

    if nn_counter < 5:
        h, w, c = image.shape
        new_h, new_w = int(h * scale), int(w * scale)
        new_img = np.zeros((new_h, new_w, c), dtype=image.dtype)

        for y in range(new_h):
            for x in range(new_w):
                src_y = round(y / scale)
                src_x = round(x / scale)
                new_img[y, x] = image[min(src_y, h - 1), min(src_x, w - 1)]

        print(f"Scale: {np.pow(scale, nn_counter + 1):.2f}")
        print(f"Height: {new_h}, Width: {new_w}")

        im = Image.fromarray(new_img)
        im.save(f'images/near_neigh/enlarged Scale {np.pow(scale, nn_counter + 1):.2f}.jpg')
        nn_counter += 1
        return new_img

    elif nn_counter == 5:
        show_summary(current_img)
        nn_counter += 1
        return None

    elif nn_counter > 5 and nn_counter < 11:
        h, w, c = image.shape
        new_h, new_w = int(h / scale), int(w / scale)
        new_img = np.zeros((new_h, new_w, c), dtype=image.dtype)

        for y in range(new_h):
            for x in range(new_w):
                src_y = round(y * scale)
                src_x = round(x * scale)
                new_img[y, x] = image[min(src_y, h - 1), min(src_x, w - 1)]

        print(f"Scale: {np.pow(scale, 10 - nn_counter):.2f}")
        print(f"Height: {new_h}, Width: {new_w}")

        im = Image.fromarray(new_img)
        im.save(f'images/near_neigh/shrinked Scale {np.pow(scale, 10 - nn_counter):.2f}.jpg')
        nn_counter += 1
        return new_img

    elif nn_counter == 11:
        show_summary(image)
        summarize_data()
        nn_counter += 1
        return None
    else:
        print("Error! Out of bounds!")
        exit()

# linear interpolation function

def linear_interpolation(image):
    global linear_counter

    if linear_counter < 5:
        h, w, c = image.shape
        new_h, new_w = int(h * scale), int(w * scale)
        new_img = np.zeros((new_h, new_w, c), dtype=image.dtype)

        for y in range(new_h):
            src_y = y / scale
            y0 = int(np.floor(src_y))
            y1 = min(y0 + 1, h - 1)
            dy = src_y - y0

            for x in range(new_w):
                src_x = x / scale
                x0 = int(np.floor(src_x))
                x1 = min(x0 + 1, w - 1)
                dx = src_x - x0

                for channel in range(c):
                    top = (1 - dx) * image[y0, x0, channel] + dx * image[y0, x1, channel]
                    bottom = (1 - dx) * image[y1, x0, channel] + dx * image[y1, x1, channel]
                    new_img[y, x, channel] = (1 - dy) * top + dy * bottom

        print(f"Scale: {np.pow(scale, linear_counter + 1):.2f}")
        print(f"Height: {new_h}, Width: {new_w}")

        linear_counter += 1
        return new_img

    elif linear_counter == 5:
        show_summary(image)
        linear_counter += 1
        return None

    elif linear_counter > 5 and linear_counter < 11:
        h, w, c = image.shape
        new_h, new_w = int(h / scale), int(w / scale)
        new_img = np.zeros((new_h, new_w, c), dtype=image.dtype)

        for y in range(new_h):
            src_y = y * scale
            y0 = int(np.floor(src_y))
            y1 = min(y0 + 1, h - 1)
            dy = src_y - y0

            for x in range(new_w):
                src_x = x * scale
                x0 = int(np.floor(src_x))
                x1 = min(x0 + 1, w - 1)
                dx = src_x - x0

                for channel in range(c):
                    top = (1 - dx) * image[y0, x0, channel] + dx * image[y0, x1, channel]
                    bottom = (1 - dx) * image[y1, x0, channel] + dx * image[y1, x1, channel]
                    new_img[y, x, channel] = (1 - dy) * top + dy * bottom

        print(f"Scale: {np.pow(scale, 10 - linear_counter):.2f}")
        print(f"Height: {new_h}, Width: {new_w}")

        linear_counter += 1
        return new_img

    elif linear_counter == 11:

        im = Image.fromarray(new_img.astype(np.uint8))
        im.save(f'images/linear/enlarged Scale {np.pow(scale, linear_counter + 1):.2f}.jpg')

        show_summary(image)
        summarize_data()
        linear_counter += 1
        return None

    else:
        print("Error! Out of bounds of interpolation number!")
        exit()

# cubic weight defining func for cubic_interpolation()

def cubic_weight(x, a=-0.5):
    x = abs(x)
    if x <= 1:
        return (a + 2)*x**3 - (a + 3)*x**2 + 1
    elif x < 2:
        return a*x**3 - 5*a*x**2 + 8*a*x - 4*a
    else:
        return 0
    
# cubic interpolation function

def cubic_interpolation(image):
    global cubic_counter

    def bicubic_resize(img, scale):
        h, w, c = img.shape
        new_h, new_w = int(h * scale), int(w * scale)
        new_img = np.zeros((new_h, new_w, c), dtype=np.float32)

        for y in range(new_h):
            src_y = y / scale
            y_int = int(np.floor(src_y))
            dy = src_y - y_int

            for x in range(new_w):
                src_x = x / scale
                x_int = int(np.floor(src_x))
                dx = src_x - x_int

                for channel in range(c):
                    result = 0.0
                    for m in range(-1, 3):
                        wy = cubic_weight(m - dy)
                        yi = np.clip(y_int + m, 0, h - 1)
                        for n in range(-1, 3):
                            wx = cubic_weight(n - dx)
                            xi = np.clip(x_int + n, 0, w - 1)
                            result += img[yi, xi, channel] * wx * wy

                    new_img[y, x, channel] = np.clip(result, 0, 255)

        return new_img.astype(img.dtype)

    if cubic_counter < 5:
        new_img = bicubic_resize(image, scale)

        print(f"Scale: {np.pow(scale, cubic_counter + 1):.2f}")
        print(f"Height: {new_img.shape[0]}, Width: {new_img.shape[1]}")

        cubic_counter += 1
        return new_img

    elif cubic_counter == 5:
        show_summary(image)
        cubic_counter += 1
        return None

    elif 5 < cubic_counter < 11:
        new_img = bicubic_resize(image, scale)

        print(f"Scale: {np.pow(scale, 10 - cubic_counter):.2f}")
        print(f"Height: {new_img.shape[0]}, Width: {new_img.shape[1]}")

        cubic_counter += 1
        return new_img

    elif cubic_counter == 11:
        im = Image.fromarray(new_img.astype(np.uint8))
        im.save(f'images/cubic/enlarged Scale {np.pow(scale, cubic_counter + 1):.2f}.jpg')

        show_summary(image)
        summarize_data()
        cubic_counter += 1
        return None

    else:
        print("Error! Out of bounds of interpolation number!")
        exit()

# funkcje interpolujące dla pikseli

def nearest_interpolation_pixel(image, x, y):
    x0 = int(round(x))
    y0 = int(round(y))
    if 0 <= x0 < image.shape[1] and 0 <= y0 < image.shape[0]:
        return image[y0, x0]
    else:
        return 0
def bilinear_interpolation_pixel(image, x, y):
    """
    Interpolacja biliniowa — dla pojedynczego punktu (x, y).
    Oblicza wartość piksela jako średnią ważoną 4 sąsiadów.
    """
    h, w = image.shape[:2]

    # Współrzędne sąsiadów
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = x0 + 1, y0 + 1

    # Sprawdzenie zakresu
    if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
        return 0

    # Odległości od pikseli
    dx = x - x0
    dy = y - y0

    # Cztery sąsiady
    Q11 = image[y0, x0]
    Q21 = image[y0, x1]
    Q12 = image[y1, x0]
    Q22 = image[y1, x1]

    # Interpolacja pozioma, potem pionowa
    R1 = (1 - dx) * Q11 + dx * Q21
    R2 = (1 - dx) * Q12 + dx * Q22
    P = (1 - dy) * R1 + dy * R2

    return P
def cubic_weight(t):
    """Pomocnicza funkcja do wag interpolacji sześciennej (Catmull-Rom)."""
    a = -0.5
    t = abs(t)
    if t <= 1:
        return (a + 2) * t**3 - (a + 3) * t**2 + 1
    elif 1 < t < 2:
        return a * t**3 - 5*a * t**2 + 8*a * t - 4*a
    else:
        return 0
def bicubic_interpolation_pixel(image, x, y):
    """
    Interpolacja sześcienna (bicubic) — dla pojedynczego punktu (x, y).
    Używa 16 sąsiadów (4x4) wokół punktu.
    """
    h, w = image.shape[:2]

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))

    if x0 < 1 or y0 < 1 or x0 + 2 >= w or y0 + 2 >= h:
        # Zwracamy zero, jeśli punkt jest przy krawędzi
        return 0

    result = 0.0
    for m in range(-1, 3):
        for n in range(-1, 3):
            px = image[y0 + m, x0 + n]
            wx = cubic_weight(n - (x - x0))
            wy = cubic_weight(m - (y - y0))
            result += px * wx * wy

    return np.clip(result, 0, 255)
if __name__ == "__main__":

    original_img, cmap = load_image(path)
    current_img = np.asarray(image)
    img_display = ax.imshow(current_img, cmap=cmap)

    plt.show()
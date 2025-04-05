import cv2
import numpy as np
import os
from PIL import Image

input = "input"
output = "output"

def get_paths():
    global input
    paths = []
    for filename in os.listdir(input):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'webp')):
            image_path = os.path.join(input, filename)
            paths.append(image_path)
    return paths

def extract_contours(input_path):
    threshold = 0
    step_threshold = 25
    contours_list = []

    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    for _ in range(10):
        _, binary = cv2.threshold(image, threshold, 250, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contours_list.extend(contours)
        threshold += step_threshold
    
        #отображение прогресса
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Выделение контуров для изображения " + str(counter) + ". " + str(threshold / 250 * 100) + "%")

    output_img = np.ones_like(image) * 255
    output_path = os.path.join(output, str(counter) + "_contours.png")

    cv2.drawContours(output_img, contours_list, -1, (0), 1)
    cv2.imwrite(output_path, output_img)
    return transparent(output_path)

def extract_shadows(input_path, start_clusters=3, max_clusters=6):
    gray_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    combined_shadows = np.zeros_like(gray_image, dtype=np.uint8)

    for num_clusters in range(start_clusters, max_clusters + 1):
        #отображение прогресса
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Выделение теней для изображения " + str(counter) + ". " + str(round((num_clusters / (max_clusters)) * 100)) + "%")
    
        #формат под kmeans
        pixels = gray_image.reshape(-1, 1).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.5)
        _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 0, cv2.KMEANS_RANDOM_CENTERS)

        labels = labels.reshape(gray_image.shape)

        centers = centers.flatten()
        sorted_centers = np.argsort(centers)

        #кластер для теней (например, средний по яркости)
        shadow_cluster = sorted_centers[num_clusters // 2]

        #маска для теней текущей итерации
        shadow_mask = np.zeros_like(gray_image, dtype=np.uint8)
        shadow_mask[labels == shadow_cluster] = 255

        #тени для текущей итерации
        current_shadows = np.zeros_like(gray_image, dtype=np.uint8)
        current_shadows[labels == shadow_cluster] = gray_image[labels == shadow_cluster]

        combined_shadows = cv2.bitwise_or(combined_shadows, current_shadows)

    output_path = os.path.join(output, str(counter) + "_shadows.png")
    cv2.imwrite(output_path, combined_shadows)
    return transparent(output_path)

def transparent(input_path):
    img = Image.open(input_path).convert("RGBA")
    datas = img.getdata()

    white = (255,255,255)
    black = (0,0,0)

    new_data = []

    if "_shadows" in input_path:
        for pix in datas:
            if pix[:3] == white:
                new_data.append((0,0,0,0))
            elif pix[:3] == black:
                new_data.append((0,0,0,0))
            else:
                new_data.append(pix)
    elif "_contours" in input_path:
        for pix in datas:
            if pix[:3] == white:
                new_data.append((0,0,0,0))
            else:
                new_data.append(pix)
    
    if os.path.exists(input_path):
        output_path = os.path.join(input_path[:-4] + "_transparent.png")
        os.remove(input_path)
        img.putdata(new_data)
        img.save(output_path, "PNG")
    return output_path

def overlay_images_with_transparency(image_path1, image_path2):
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise ValueError("Не удалось загрузить одно или оба изображения")

    if img1.shape != img2.shape:
        raise ValueError("Размеры изображений не совпадают")

    img1_bgr = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_bgr = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    height, width = img1.shape
    result = np.zeros((height, width, 4), dtype=np.uint8)

    result[:, :, :3] = img1_bgr
    alpha = np.ones((height, width), dtype=np.uint8) * 255 

    common_pixels = (img1 == img2)
    alpha[common_pixels] = 0

    different_pixels = (img1 != img2)
    result[different_pixels, :3] = img2_bgr[different_pixels]

    result[:, :, 3] = alpha

    output_path = os.path.join(output, str(counter) + "_overlay.png")
    cv2.imwrite(output_path, result)

if __name__ == "__main__":
    image_list = get_paths()
    counter = 1
    for img_path in image_list:
        contour = extract_contours(img_path)
        shadow = extract_shadows(img_path)
        #overlay_images_with_transparency(contour, shadow)
        counter += 1
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"Обработано {counter - 1} изображений.")
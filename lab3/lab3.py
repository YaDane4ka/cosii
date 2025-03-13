import os
import cv2
import numpy as np
from PIL import Image
from math import sqrt


def rotate_objects(image_path, object_locations, output_path, shift_amount=50, min_aspect_ratio=0.5,
                   max_aspect_ratio=2.0):
    # Загружаем изображение
    image = cv2.imread(image_path)

    image_height, image_width = image.shape[:2]
    new_height = image_height + shift_amount
    rotated_image = np.zeros((new_height, image_width, 3), dtype=np.uint8)
    rotated_image[shift_amount:, :] = image
    rotated_image = np.zeros((new_height, image_width, 3), dtype=np.uint8)

    sycle = 0
    for index, (x1, x2, x3, x4) in enumerate(object_locations):
        sycle += 1
        delta_y = x4[1] - x1[1]
        delta_x = x4[0] - x1[0]
        h = sqrt(((x1[1] - x2[1]) ** 2) + ((x2[0] - x1[0]) ** 2))
        w = sqrt(((x3[1] - x2[1]) ** 2) + ((x3[0] - x2[0]) ** 2))
        angle = np.arctan2(delta_y, delta_x) * (180 / np.pi)  # угол в градусах
        if h < w:
            angle += 90
        x_min = min(x1[0], x2[0], x3[0], x4[0])
        y_min = min(x1[1], x2[1], x3[1], x4[1])
        x_max = max(x1[0], x2[0], x3[0], x4[0])
        y_max = max(x1[1], x2[1], x3[1], x4[1])
        object_roi = image[y_min:y_max, x_min:x_max]

        if object_roi.size == 0:
            print(f"Empty region for object {index}, skipping rotation")
            continue  # Пропускаем, если область пустая

        center = (object_roi.shape[1] // 2, object_roi.shape[0] // 2)  # Центр объекта
        rotated_object = cv2.warpAffine(object_roi, cv2.getRotationMatrix2D(center, angle, 1.0),
                                        (object_roi.shape[1] + 100, object_roi.shape[0] + 100),
                                        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        x_offset = max(0, x_min)
        y_offset = max(0, y_min)
        if sycle == 3:
            y_offset += 90
        if sycle == 4:
            x_offset += 37
        rotated_image[y_offset:y_offset + rotated_object.shape[0],
        x_offset:x_offset + rotated_object.shape[1]] = rotated_object

    cv2.imwrite(output_path, rotated_image)


def find_object_locations(image_path: str, min_area: int = 15000):
    # Загружаем изображение
    image = cv2.imread(image_path)
    output_image = image.copy()
    blue_channel = image[:, :, 0]
    green_channel = image[:, :, 1]

    # Объединяем два канала для создания единого порогового изображения
    combined_channel = cv2.bitwise_and(blue_channel, green_channel)
    # Применяем пороговую обработку
    _, threshold_combined = cv2.threshold(combined_channel, 127, 255, cv2.THRESH_BINARY)
    # Применяем операцию расширения (dilation) для увеличения объекта
    kernel = np.ones((10, 10), np.uint8)  # Размер ядра можно изменять для большей или меньшей расширенной области
    dilated_image = cv2.dilate(threshold_combined, kernel, iterations=1)
    # Находим контуры на основе расширенного изображения
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    object_locations = []  # Список для хранения координат объектов
    for contour in contours:
        # Вычисляем площадь контура
        area = cv2.contourArea(contour)
        if area > min_area:
            # Находим минимальный ограничивающий прямоугольник
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)  # Преобразуем в 4 угла
            box = np.int32(box)  # Преобразуем в целые числа
            # x1 = box[0][0]
            # y1 = box[0][1]
            # x2 = box[1][0]
            # y2 = box[1][1]
            # x3 = box[2][0]
            # y3 = box[2][1]
            # x4 = box[3][0]
            # y4 = box[3][1]

            # Добавляем объект в список
            object_locations.append(box)
            # Отображаем ограничивающий прямоугольник на изображении
            cv2.drawContours(output_image, [box], 0, (0, 255, 0), 2)

    # cv2.imwrite(image_path, output_image)
    # Возвращаем список с координатами объектов
    return object_locations


def remove_background(image_path: str, output_path: str) -> None:
    # Загрузка изображения
    image = cv2.imread(image_path)
    # Преобразование изображения в цветовую модель HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Устанавливаем расширенный диапазон для зеленого цвета
    lower_color = np.array([30, 50, 50])  # Минимальные значения для зеленого
    upper_color = np.array([90, 255, 255])  # Максимальные значения для зеленого
    # Создаём маску, которая будет выделять только указанный цвет
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # Применяем размытие для улучшения маски и уменьшения потерь
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    # Применяем маску к изображению
    result = cv2.bitwise_and(image, image, mask=mask)
    # Сохраняем результат без потерь качества
    cv2.imwrite(output_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def process_images(input_folder: str, output_folder: str) -> None:
    # Проверка существования выходной папки и её создание при необходимости
    os.makedirs(output_folder, exist_ok=True)

    # Проход по всем изображениям в входной папке
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Вызов функции удаления фона
            remove_background(input_path, output_path)
            location = find_object_locations(output_path)
            rotate_objects(output_path, location, output_path)


process_images('./Figure', './Figure_output')

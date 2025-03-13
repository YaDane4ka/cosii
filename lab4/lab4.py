import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def classify_objects(image, object_locations, model):
    results = []
    for (x, y, w, h) in object_locations:
        # Вырезаем объект из изображения
        cropped_object = image[y:y + h, x:x + w]

        # Изменяем размер до 28x28 для MNIST
        resized_object = cv2.resize(cropped_object, (28, 28))

        # Преобразуем изображение в черно-белое (одноканальное)
        resized_object = cv2.cvtColor(resized_object, cv2.COLOR_BGR2GRAY)  # Конвертируем в оттенки серого
        resized_object = np.expand_dims(resized_object, axis=-1)  # Добавляем канал

        # Нормализуем изображение
        resized_object = resized_object / 255.0
        resized_object = np.expand_dims(resized_object, axis=0)  # Добавляем размерность батча

        # Получаем предсказания
        predictions = model.predict(resized_object)

        # Находим класс с максимальной вероятностью
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class]

        # Добавляем результат
        results.append((predicted_class, confidence))

    return results


def find_object_locations(image_path: str, min_area: int = 15000):
    # Загружаем изображение
    image = cv2.imread(image_path)
    output_image = image.copy()
    # Преобразуем изображение в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Применяем размытие для уменьшения шума
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Применяем пороговую обработку
    _, threshold = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    # Применяем морфологические операции для сглаживания контуров
    kernel = np.ones((20, 12), np.uint8)
    threshold = cv2.dilate(threshold, kernel, iterations=2)
    threshold = cv2.erode(threshold, kernel, iterations=1)
    # Находим контуры объектов
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    object_locations = []  # Список для хранения координат прямоугольников
    for contour in contours:
        # Вычисляем площадь контура
        area = cv2.contourArea(contour)
        if area > min_area:  # Фильтрация по минимальной площади
            # Находим ограничивающий прямоугольник (параллельный осям)
            x, y, w, h = cv2.boundingRect(contour)
            # Добавляем координаты прямоугольника в список
            object_locations.append((x, y, w, h))
            # Отображаем прямоугольник на изображении
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Сохраняем изображение с прямоугольниками
    cv2.imwrite(image_path, output_image)

    # Возвращаем список с координатами прямоугольников
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


def process_images(input_folder: str, output_folder: str, model_path: str) -> None:
    # Проверка существования выходной папки и её создание при необходимости
    os.makedirs(output_folder, exist_ok=True)
    # Загрузка модели
    model = load_model(model_path)

    # Проход по всем изображениям в входной папке
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Вызов функции удаления фона
            remove_background(input_path, output_path)

            # Найдем расположение объектов на изображении
            object_locations = find_object_locations(output_path)

            # Загрузим изображение без фона для классификации
            image = cv2.imread(output_path)

            # Классификация объектов
            results = classify_objects(image, object_locations, model)

            # Выводим результаты и добавляем подписи
            for i, (cls, conf) in enumerate(results):
                label = f"Number {cls}, {conf * 100:.2f}%"
                (x, y, w, h) = object_locations[i]
                # Добавляем текст с подписью на изображение
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                # Отображаем ограничивающий прямоугольник
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Сохраняем изображение с подписями
            cv2.imwrite(output_path, image)

process_images('./3', './4', 'mnist_model.h5')

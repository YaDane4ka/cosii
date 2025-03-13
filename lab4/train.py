import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2


# Загрузка данных MNIST
def load_mnist_data():
    # Загружаем данные MNIST из TensorFlow
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # Нормализуем изображения в диапазоне [0, 1]
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Преобразуем изображения в форму (batch_size, 28, 28, 1) для сверточных сетей
    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)

    # Преобразуем метки в категориальные (one-hot)
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    return (train_images, train_labels), (test_images, test_labels)


# Построение и обучение модели
def train_model(dataset_path=None, model_save_path="mnist_model.h5", input_shape=(28, 28, 1), batch_size=32, epochs=20):
    # Загрузка данных MNIST
    (train_images, train_labels), (test_images, test_labels) = load_mnist_data()

    # Аугментация данных для улучшения модели
    datagen = ImageDataGenerator(
        rotation_range=20,  # Увеличение диапазона поворота
        width_shift_range=0.2,  # Увеличение сдвига по ширине
        height_shift_range=0.2,  # Увеличение сдвига по высоте
        zoom_range=0.2,  # Увеличение диапазона увеличения
        shear_range=0.2,  # Увеличение диапазона сдвига
        horizontal_flip=False,  # Поскольку это цифры, горизонтальное зеркалирование может быть неуместным
        fill_mode='nearest'
    )

    # Создание модели
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same', kernel_regularizer=l2(0.001)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)),  # Дополнительный слой
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(10, activation='softmax')  # 10 классов (цифры от 0 до 9)
    ])

    # Компиляция модели
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Более низкая скорость обучения для точной настройки
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Обучение модели с использованием аугментации данных
    model.fit(
        datagen.flow(train_images, train_labels, batch_size=batch_size),
        epochs=epochs,
        validation_data=(test_images, test_labels),
        steps_per_epoch=train_images.shape[0] // batch_size  # Количество шагов за эпоху
    )

    # Сохранение модели
    model.save(model_save_path)
    print(f"Модель сохранена в {model_save_path}")


if __name__ == "__main__":
    # Задайте путь к датасету (не используется в этом примере, так как мы загружаем данные через TensorFlow)
    model_save_path = "mnist_model.h5"
    train_model(model_save_path=model_save_path)

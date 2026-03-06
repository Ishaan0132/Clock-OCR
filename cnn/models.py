import tensorflow as tf
from tensorflow.keras import layers, models

def build_model_38(input_shape=(38, 38, 1), num_classes=144):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape, kernel_initializer='he_uniform'),
        layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform'),
        layers.MaxPooling2D(2,2),
        layers.BatchNormalization(),
        layers.Conv2D(64, (2,2), activation='relu', kernel_initializer='he_uniform'),
        layers.Conv2D(64, (2,2), activation='relu', kernel_initializer='he_uniform'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (2,2), activation='relu', kernel_initializer='he_uniform'),
        layers.Conv2D(128, (2,2), activation='relu', kernel_initializer='he_uniform'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_model_64(input_shape=(64, 64, 1), num_classes=144):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape, kernel_initializer='he_uniform'),
        layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform'),
        layers.MaxPooling2D(2,2),
        layers.BatchNormalization(),
        layers.Conv2D(64, (2,2), activation='relu', kernel_initializer='he_uniform'),
        layers.Conv2D(64, (2,2), activation='relu', kernel_initializer='he_uniform'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (2,2), activation='relu', kernel_initializer='he_uniform'),
        layers.Conv2D(128, (2,2), activation='relu', kernel_initializer='he_uniform'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
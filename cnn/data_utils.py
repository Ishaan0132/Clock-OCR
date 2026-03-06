import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(data_dir, target_size, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=10, width_shift_range=0.05,
        height_shift_range=0.05, zoom_range=0.05, validation_split=0.15
    )
    train_gen = train_datagen.flow_from_directory(
        data_dir, target_size=target_size, color_mode='grayscale',
        batch_size=batch_size, class_mode='categorical', subset='training'
    )
    val_gen = train_datagen.flow_from_directory(
        data_dir, target_size=target_size, color_mode='grayscale',
        batch_size=batch_size, class_mode='categorical', subset='validation'
    )
    # Test generator logic
    test_dir = data_dir.replace('train', 'test') # Assumes standard naming
    test_gen = None
    try:
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_gen = test_datagen.flow_from_directory(
            test_dir, target_size=target_size, color_mode='grayscale',
            batch_size=batch_size, class_mode='categorical', shuffle=False
        )
    except:
        print(f"Test directory not found at {test_dir}")
    return train_gen, val_gen, test_gen

def plot_history(hist, title):
    plt.figure(figsize=(6,4))
    plt.plot(hist.history['accuracy'], label='Train Acc')
    plt.plot(hist.history['val_accuracy'], label='Val Acc')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def show_random_predictions(model, generator, n=6):
    x_all, y_all = [], []
    for i in range(len(generator)):
        xb, yb = generator[i]
        x_all.append(xb)
        y_all.append(yb)
    x_all = np.vstack(x_all)
    y_all = np.vstack(y_all)
    
    indices = random.sample(range(len(x_all)), n)
    x_samples = x_all[indices]
    y_samples = y_all[indices]
    preds = np.argmax(model.predict(x_samples), axis=1)
    true = np.argmax(y_samples, axis=1)
    labels = list(generator.class_indices.keys())
    
    plt.figure(figsize=(12, 8))
    for i in range(n):
        plt.subplot(2, n//2, i+1)
        plt.imshow(x_samples[i].squeeze(), cmap='gray')
        plt.title(f"True: {labels[true[i]]}\nPred: {labels[preds[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
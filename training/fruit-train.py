import tensorflow as tf
import cv2
import os
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
import numpy as np


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


data_dir = 'data/fruit'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

data = tf.keras.utils.image_dataset_from_directory('data/fruit')
class_names = data.class_names  # save class names before mapping
print(f"Class names: {class_names}")
data = data.map(lambda x,y: (x/255, y))


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

def augment(image, label):
    return data_augmentation(image, training=True), label

train_size = int(len(data)*.7) 
val_size = int(len(data)*.2)+1
test_size = int(len(data)*.1)+1

"""
print(len(data))
print(train_size)
print(val_size)
print(test_size)
"""


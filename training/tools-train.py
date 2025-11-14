import tensorflow as tf
import cv2
import os
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
from tensorflow.keras.applications import MobileNetV2
import numpy as np


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


data_dir = 'data/tools'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

data = tf.keras.utils.image_dataset_from_directory('data/tools')
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

train = data.take(train_size)
train = train.map(augment)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

model_path = 'models/tools_class.h5'

if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    num_classes = len(class_names)
    
    #lightweight pre trained model
    base_model = MobileNetV2(input_shape=(256, 256, 3),
                             include_top=False,
                             #uses google trained dataset
                             weights='imagenet')
    base_model.trainable = False  
    

    #all layers of training on top of the mobilenet
    model = Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    #optimizes models learning
    model.compile('adam', 
                  loss=tf.losses.SparseCategoricalCrossentropy(), 
                  metrics=['accuracy'])

    ldir = 'logs/tools'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=ldir)
    #early stopping important to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )
    #actual training of model
    hist = model.fit(
        train, 
        epochs=50, 
        validation_data=val, 
        callbacks=[tensorboard_callback, early_stopping]
    )
    #saves model
    os.makedirs('models', exist_ok=True)
    model.save(model_path)

    print(f"accuracy: {hist.history['accuracy'][-1]*100:.2f}%")
    print(f"val acc: {hist.history['val_accuracy'][-1]*100:.2f}%")


print("test")
test_loss, test_accuracy = model.evaluate(test, verbose=0)
print(f"\ntest accuracy: {test_accuracy*100:.2f}%")
print(f"loss: {test_loss:.4f}")

# Test individual tool image
test_image_path = 'test/sdrivertest.jpg'  # Use real path
img = cv2.imread(test_image_path)


if img is not None:
    resize = tf.image.resize(img, (256, 256))
    yhat = model.predict(np.expand_dims(resize/255, 0), verbose=0)
    
    predicted_idx = np.argmax(yhat[0])
    confidence = yhat[0][predicted_idx]
    
    # Display
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"predicted tool: {class_names[predicted_idx]} ({confidence*100:.1f}%)")
    plt.axis('off')
    plt.show()
    
    print(f"\nprediction: {class_names[predicted_idx]}")
    print(f"conf: {confidence*100:.2f}%")
    print(f"\nprobs:")
    for i, name in enumerate(class_names):
        print(f"{name:10}: {yhat[0][i]*100:.1f}%")

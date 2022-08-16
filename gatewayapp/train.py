# Copyright (C) 2020 - 2022 APC, Inc.

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
# from keras.callbacks import Callback
from tensorflow.keras.callbacks import Callback

print(tf.__version__)

# Define some parameters for the dataset loader
batch_size = 32
img_height = 180*3
img_width = 180*3
# data_dir = "/data/workspace-demoapp-v2/dataset"
# data_dir = "/root/.keras/datasets/flower_photos"
data_dir = "/data/workspace-demoapp-v2/mlapp/working_directory"
saved_model = "/data/workspace-demoapp-v2/saved-model/"

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class TrainingCallback(Callback):
    def __init__(self, publish_func):
        super(TrainingCallback, self).__init__()
        self.publish_func = publish_func
    def on_epoch_end(self, epoch, logs=None):
        self.publish_func(epoch)

def setup():
    # Create data training (80%) and validation (20%)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    print(class_names)

    # Configure the dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds



def create_model(num_classes):
    # data augmentation
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                            input_shape=(img_height,
                                        img_width,
                                        3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )
    model = Sequential([
        data_augmentation,  # augmentation to reduce overfit
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),  # dropout to reduce overfit
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return model

def train(epochs, train_ds, val_ds, num_classes, publish_func):
    trainingCallback = TrainingCallback(publish_func)
    model = create_model(num_classes)
    # model.summary()
    # epochs=15
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[trainingCallback]
        # workers=4,
        # use_multiprocessing=True,
    )

    # Save the entire model to a HDF5 file.
    # The '.h5' extension indicates that the model should be saved to HDF5.
    model.save(saved_model + "my_model.h5")

    # Create plots of loss and accuracy on the training and validation sets:
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig(saved_model + '/report.png')

# example callback function
def myfunc(epoch):
    print("epoch", epoch)

def startTraining(totalEpoch, numberOfClasses, myfunc):
    train_ds, val_ds = setup()
    train(totalEpoch, train_ds, val_ds, numberOfClasses, myfunc)
    print("training is completed")

if __name__ == '__main__':
    print("start training")
    train_ds, val_ds = setup()
    
    train(15, train_ds, val_ds, 4, myfunc)
    print("training is completed")
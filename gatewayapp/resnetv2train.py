# Copyright (C) 2020 - 2022 APC, Inc.

import os
import tensorflow as tf #tf  > 2.0.0
import numpy as np
from tensorflow.keras import Model 
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
import splitfolders
import shutil
from tensorflow.keras.preprocessing import image
from timeit import default_timer as timer
from utility import get_base_dir
import json
import datetime
from timeit import default_timer as timer

# Define some parameters for the dataset loader
img_height = 256
img_width = 256

g_cancel = False
g_prev_end_time = None

class TrainingCallback(Callback):
    def __init__(self, publish_func, total_epoch):
        super(TrainingCallback, self).__init__()
        self.publish_func = publish_func
        self.total_epoch = total_epoch
    def on_train_begin(self, logs=None):
        global g_prev_end_time
        g_prev_end_time = datetime.datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        global g_cancel                
        global g_prev_end_time
        # estimate epoch time
        diff_time = int((datetime.datetime.now() - g_prev_end_time).total_seconds())
        eta_sec = diff_time * (self.total_epoch + 1 - epoch) + 30 # plus extra time
        progress = int((epoch + 1) * 100.0 / self.total_epoch)
        if (progress > 5):
            progress = progress - 5
        print("epoch", epoch, "eta_sec", eta_sec, "diff_time", diff_time, "progress", progress)

        self.publish_func(progress, eta_sec)
        g_prev_end_time = datetime.datetime.now()
        
        self.model.stop_training = g_cancel # this is how to cancel the training        

    def on_batch_end(self, batch, logs=None):
        global g_cancel
        self.model.stop_training = g_cancel # this is how to cancel the training

# see https://github.com/miladfa7/Image-Classification-Transfer-Learning/blob/master/ResNet_image_classification.ipynb
def train(epochs, classes, publish_func, img_size, batch_size, training_dir, valid_dir, saved_path):
    global g_cancel, img_height, img_width
    g_cancel = False
    img_height = img_size
    img_width = img_size
    
    numberOfClasses = len(classes)
    trainingCallback = TrainingCallback(publish_func, epochs)
    # ImageDataGenerator (in-place augmentation)
    from tensorflow.keras.preprocessing.image import ImageDataGenerator


    train_data_gen = ImageDataGenerator(rotation_range=50,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        zoom_range=0.3,
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                        fill_mode='constant',
                                        cval=0,
                                        rescale=1./255)
    valid_data_gen = ImageDataGenerator(rotation_range=45,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        zoom_range=0.3,
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                        fill_mode='constant',
                                        cval=0,
                                        rescale=1./255)
    SEED = 1234
    tf.random.set_seed(SEED) 
    print("scan train directory")
    train_gen = train_data_gen.flow_from_directory(training_dir,
                                                target_size=(img_height, img_width),
                                                batch_size=batch_size,
                                                # classes=classes,
                                                class_mode='categorical',
                                                shuffle=True,
                                                seed=SEED)  # targets are directly converted into one-hot vectors

    # Validation
    print("scan val directory")
    valid_gen = valid_data_gen.flow_from_directory(valid_dir,
                                            target_size=(img_height, img_width),
                                            batch_size=batch_size, 
                                            # classes=classes,
                                            class_mode='categorical',
                                            shuffle=False,
                                            seed=SEED)
    
    
    ### transfer learning and fine tuning model based on ResNet152V2
    # ResNet152V2 Model
    ResNet_model = tf.keras.applications.ResNet152V2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    # The last 15 layers fine tune
    for layer in ResNet_model.layers[:-15]:
        layer.trainable = False

    x = ResNet_model.output
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(units=512, activation='relu')(x)
    x = Dropout(0.3)(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    model = Model(ResNet_model.input, output)
    model.summary()
    
    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=loss, metrics= ['accuracy'])
    
    lrr = ReduceLROnPlateau(monitor='val_accuracy', 
                        patience=3, 
                        verbose=1, 
                        factor=0.4, 
                        min_lr=0.0001)


    callbacks = [lrr]

    STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
    print("train_gen.n", train_gen.n, "train_gen.batch_size", train_gen.batch_size, "STEP_SIZE_TRAIN", STEP_SIZE_TRAIN)
    STEP_SIZE_VALID=valid_gen.n//valid_gen.batch_size
    print("valid_gen.n", valid_gen.n, "valid_gen.batch_size", valid_gen.batch_size, "STEP_SIZE_VALID", STEP_SIZE_VALID)
    
    transfer_learning_history = model.fit_generator(generator=train_gen,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_gen,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=epochs,
                    callbacks=[callbacks, trainingCallback],
                    # class_weight='auto', # TODO: auto is not working maybe we have to calculate the class_weight manually                                            
    )
    
    if (g_cancel):
        print("training is cancelled, so just return without saving model")
        # remove mydataset
        return
    
    # Save the entire model to a HDF5 file.
    # The '.h5' extension indicates that the model should be saved to HDF5.
    model.save(os.path.join(saved_path, "my_model.h5"))
    
    acc = transfer_learning_history.history['accuracy']
    val_acc = transfer_learning_history.history['val_accuracy']

    loss = transfer_learning_history.history['loss']
    val_loss = transfer_learning_history.history['val_loss']

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
    
    plt.savefig(os.path.join(saved_path, 'report.png'))

def removeMyDataset(dataset_dir):
    # remove mydataset folder
    try:
        shutil.rmtree(dataset_dir)
    except OSError as e:
        if (e.errno != 2):
            print("erro file:%s errno:%s strerror:%s" % (dataset_dir,  e.errno, e.strerror))

def dummyCallback(epoch, epoch_time_sec):
    print("finished epoch", epoch, "epoch_time_sec", epoch_time_sec)  

def main(working_directory, image_size, batch_size, saved_path, epochs, dummyCallbackParam):
    # construct classes
    with open(os.path.join(working_directory, "labels.json")) as f:
        labelJson = json.load(f) 
    classes = np.array([str(a['id']) for a in labelJson])
    print("classes", classes)

    dataset_dir = os.path.join(working_directory, "restnetv2_dataset")    
    # clear dataset directory
    removeMyDataset(dataset_dir)

    # create dataset train & val
    create_dataset(working_directory, dataset_dir, "train")
    create_dataset(working_directory, dataset_dir, "val")
    
    training_dir = os.path.join(dataset_dir, "train")
    valid_dir = os.path.join(dataset_dir, "val")
    train(epochs, classes, dummyCallbackParam, image_size, batch_size, training_dir, valid_dir, saved_path)


def create_dataset(working_directory, dataset_dir, dataset_name):
    with open(os.path.join(working_directory, dataset_name + ".json")) as f:
        dataJson = json.load(f) 
    # print("dataset", dataset_name, dataJson)
    for img_info in dataJson["images"]:
        # print("img_info", img_info)
        dir_dst = os.path.join(dataset_dir, dataset_name, str(img_info["class_id"]))
        # print("dir_dst", dir_dst)
        os.makedirs(dir_dst, exist_ok=True)
        img_src = os.path.join(working_directory, "images", img_info["file_name"])
        img_dst = os.path.join(dir_dst, img_info["file_name"])
        try:
            # print("img_src", img_src, "img_dst", img_dst)         
            shutil.copy(img_src, img_dst)
            # print("copied img_src", img_src, "img_dst", img_dst)
        except OSError as e:
            print("copy error ", e.strerror)

# def inference():
#     np.set_printoptions(suppress=True)
#     mymodel = tf.keras.models.load_model(saved_model + "my_model.h5")
#     img_path = '/home/buser/project/workspace-demoapp-v1/resnet-train-classifier/Dataset/test/IMG_31.jpg' #computer-monitor
#     img = image.load_img(img_path, target_size=(img_width, img_height))
#     print("img", img)
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = tf.keras.applications.resnet_v2.preprocess_input(x)
#     preds = mymodel.predict(x)
#     print("preds", preds)

def cancelTraining():
    global g_cancel
    print("try to cancel training")
    g_cancel = True
    
if __name__ == '__main__':
    print("start training")    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024*2)])
            # tf.config.experimental.set_memory_growth(gpus[0], True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    epochs = 15
    image_size = 256 # 256 512 1024
    batch_size = 8
    working_directory = '/workspace-test-v1/mlapp/working_directory_ic2'
    saved_path = '/workspace-test-v1/saved-model'
    main(working_directory, image_size, batch_size, saved_path, epochs, dummyCallback)
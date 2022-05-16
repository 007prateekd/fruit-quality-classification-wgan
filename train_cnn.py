from lightgbm import early_stopping
import numpy as np
from regex import D
import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet152, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from preprocess import get_new_np_path
from constants import *


def load_real_images():
    new_path = get_new_np_path()
    with open(new_path, "rb") as f:
        images = np.load(f)
        labels = np.load(f)
    return images, labels


def load_fake_images(path):
    with open(path, "rb") as f:
        images = np.load(f)
        labels = np.load(f)
    return images, labels


def get_cnn_model():
    # Manual Augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
    ])
    base_model = ResNet152(
        input_shape=(256, 256, 3), 
        include_top=False, 
        weights="imagenet"
    )
    base_model.trainable = False
    global_average_layer = GlobalAveragePooling2D()
    prediction_layer = Dense(1, activation='sigmoid')
    inputs = tf.keras.Input(shape=(256, 256, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def combine_real_and_fake():
    img_real, lab_real = load_real_images()
    p = np.random.permutation(len(img_real))
    img_real, lab_real = img_real[p], lab_real[p]
    n = int(len(img_real) * 0.9)
    n = int(len(img_real) * 0.9)
    img_real_train, lab_real_train = img_real[:n], lab_real[:n]
    img_real_test, lab_real_test = img_real[n:], lab_real[n:]
    img0, lab0 = load_fake_images(PATH_TO_GEN_IMG0)
    img1, lab1 = load_fake_images(PATH_TO_GEN_IMG1)
    images = np.concatenate([img_real_train, img0, img1])
    labels = np.concatenate([lab_real_train, lab0, lab1])
    p = np.random.permutation(len(images))
    images, labels = images[p], labels[p]
    return images, labels, img_real_test, lab_real_test


def train():
    images, labels, images_test, labels_test = combine_real_and_fake()
    model = get_cnn_model()
    # Transfer learning
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=BinaryCrossentropy(),
        metrics=['accuracy']
    )
    checkpoint = ModelCheckpoint(save_best_only=True, filepath=PATH_TO_CKP)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    callbacks = [checkpoint, early_stopping];
    history = model.fit(
        images, labels, validation_split=0.2, epochs=500, 
        callbacks=callbacks
    ) 
    # Fine tuning
    model.trainable = True
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss=BinaryCrossentropy(),
        metrics=['accuracy']
    )
    history_ft = model.fit(
        images, labels, validation_split=0.2, epochs=250, 
    ) 
    model.evaluate(images_test, labels_test)

if __name__ == "__main__":
    train()
    
    
    

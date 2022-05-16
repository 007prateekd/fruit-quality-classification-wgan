from constants import *
import numpy as np
from preprocess import get_new_np_path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Reshape, Conv2D, Conv2DTranspose, LeakyReLU, Dense, GlobalMaxPooling2D


def load_data():
    new_path = get_new_np_path()
    with open(new_path, "rb") as f:
        images = np.load(f)
        labels = np.load(f)
    labels = tf.keras.utils.to_categorical(labels, 2)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return dataset


def get_discriminator():
    # 256,256,5 -> 128,128,64 -> 64,64,128 -> 32,32,256 -> 1,256 -> 1 
    discriminator = tf.keras.Sequential(
        [
            InputLayer((IMG_SIZE, IMG_SIZE, DIS_IN_CHANNELS)),
            Conv2D(64,  (3, 3), strides=2, padding="same"),
            LeakyReLU(alpha=0.2),
            Conv2D(128, (3, 3), strides=2, padding="same"),
            LeakyReLU(alpha=0.2),
            Conv2D(256, (3, 3), strides=2, padding="same"),
            LeakyReLU(alpha=0.2),
            GlobalMaxPooling2D(),
            Dense(1),
        ],
    )
    return discriminator


def get_generator():
    # 1,130 -> 32,32,130 -> 64,64,64 -> 128,128,128 -> 256,256,256 -> 256,256,3
    generator = tf.keras.Sequential(
        [
            InputLayer((GEN_IN_CHANNELS,)),
            Dense((IMG_SIZE * IMG_SIZE * GEN_IN_CHANNELS) // 64),
            LeakyReLU(alpha=0.2),
            Reshape((IMG_SIZE // 8, IMG_SIZE // 8, GEN_IN_CHANNELS)),
            Conv2DTranspose(64, (4, 4), strides=2, padding="same"),
            LeakyReLU(alpha=0.2),
            Conv2DTranspose(128, (4, 4), strides=2, padding="same"),
            LeakyReLU(alpha=0.2),
            Conv2DTranspose(256, (4, 4), strides=2, padding="same"),
            LeakyReLU(alpha=0.2),
            Conv2D(3, (7, 7), padding="same", activation="tanh"),
        ],
    )
    return generator


class WCGAN_GP(tf.keras.Model):
    def __init__(self, discriminator, generator, d_steps=5, gp_weight=10.0):
        # Discriminator Steps (d_steps > 1) -> Train D more than G
        super(WCGAN_GP, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.d_steps = d_steps
        self.gp_weight = gp_weight
        self.gen_loss_tracker = tf.keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="discriminator_loss")
    
    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]
    
    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(WCGAN_GP, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        
    def gradient_penalty(self, batch_size, real_images, fake_images):
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        real_images, one_hot_labels = data
        image_one_hot_labels = tf.repeat(
            one_hot_labels, repeats=[IMG_SIZE * IMG_SIZE]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, IMG_SIZE, IMG_SIZE, NUM_CLASSES)
        )
        batch_size = tf.shape(real_images)[0]

        for _ in range(self.d_steps):
            random_latent_vectors = tf.random.normal(shape=(batch_size, LATENT_DIM))
            random_vector_labels = tf.concat(
                [random_latent_vectors, one_hot_labels], axis=1
            )
            fake_images = self.generator(random_vector_labels, training=True)
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
            combined_images = tf.concat(
                [fake_image_and_labels, real_image_and_labels], axis=0
            )
            labels = tf.concat(
                [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
            )
            # One sided label smoothing -> Flip some labels at random
            if (np.random.rand() > 0.5):
                labels = tf.random.shuffle(labels)

            with tf.GradientTape() as tape:
                preds = self.discriminator(combined_images)
                d_loss = self.loss_fn(labels, preds)
                gp = self.gradient_penalty(batch_size, real_image_and_labels, fake_image_and_labels)
                d_loss = d_loss + gp * self.gp_weight
            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights)
            )

        random_latent_vectors = tf.random.normal(shape=(batch_size, LATENT_DIM))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )
        misleading_labels = tf.zeros((batch_size, 1))
        
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            preds = self.discriminator(fake_image_and_labels)
            g_loss = self.loss_fn(misleading_labels, preds)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {"g_loss": g_loss, "d_loss": d_loss}


def augment():
    dataset = load_data()
    discriminator = get_discriminator()
    generator = get_generator()
    # Two Time-scale Update Rule -> Different LRs for G, D
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=4e-3)
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
    gan = WCGAN_GP(discriminator, generator, 1)
    gan.compile(    
        d_optimizer, g_optimizer,
        tf.keras.losses.BinaryCrossentropy(from_logits=True)
    )
    gan.fit(dataset, epochs=1000)
    gan.save_weights(PATH_TO_MODEL)
    generator.save_weights(PATH_TO_GEN)


if __name__ == "__main__":
    augment()

from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import time
import os
import cv2
import numpy as np
from constants import *

def generate_latent_points(latent_dim, n_samples, class_to_gen, n_classes=2):
	x_input = randn(latent_dim * n_samples)
	z_input = x_input.reshape(n_samples, latent_dim)
	if class_to_gen == 1:
		labels = randint(1, n_classes, n_samples)
	if class_to_gen == 0:
		labels = randint(0, 1, n_samples)
	return [z_input, labels]


def save_plot(examples, n, full_path):
	for i in range(n * n):
		plt.subplot(n, n, 1 + i)
		plt.axis('off')
		plt.imshow(examples[i])
		filename = full_path + str(time.time()) + ".jpg"
		plt.imsave(filename, examples[i])


def generate(classToGen, full_path, weights_file):
	model = load_model(weights_file)
	for _ in range(0, 10):
		latent_points, labels = generate_latent_points(LATENT_DIM, 100, classToGen)
		X  = model.predict([latent_points, labels])
		X = (X + 1) / 2.0
		save_plot(X, 10, full_path)


def save_gen_images(classToGen, weights_file):
	full_path = "examples/" + str(classToGen) + "/"
	if not os.path.exists(full_path):
		os.makedirs(full_path)
		generate(classToGen, full_path, weights_file)
 
  
def convert_to_npy(classToGen):
    full_path = "examples/" + str(classToGen)
    images, labels = [], []
    for _, _, files in os.walk(full_path):
        for file in files:
            img = cv2.imread(os.path.join(full_path, file))
            images.append(img)
            labels.append(classToGen)
    images = np.array(images, dtype="float")
    labels = np.array(labels)
    path = PATH_TO_GEN_IMG0 if classToGen == 0 else PATH_TO_GEN_IMG1
    with open(path, "wb") as f:
        np.save(f, images)
        np.save(f, labels)


if __name__ == "__main__":
    # 0 -> unhealthy, 1 -> healthy
	weights_file = "store/wgan-gp-gen.h5"
	save_gen_images(0, weights_file)
	save_gen_images(1, weights_file)
	convert_to_npy(0)
	convert_to_npy(1)
	# To generate lemon images
	# generate(1, weights_file)

# fruit-quality-classification-wgan

## What

This repository aims to solve the task of fruit quality classification given very limited data. The <a href="https://github.com/softwaremill/lemon-dataset">dataset</a> contains 2690 images of lemons which is too less to train a deep learning model, let alone use the model practically.

## Why

Fruit quality classification of defective fruits is an important task in the fruit processing industry. An automated system that could determine fruit quality by classifying it on the basis of surface defects is needed for enhancing fruit production, reducing the dependency on manual labour and reducing losses due to fruit quality deterioration.

## How

We solve this task by first augmenting the data trying many variations of GANs - cGAN, WGAN and WGAN-GP with WGAN-GP generating the best images and giving the best results upon combining with the classification pipeline. The classification using transfer learning by trying different pretrained CNN models - VGG16, Xception, DenseNet, EfficientNet and ResNet with ResNet giving the best results of 92.11% testing accuracy. 

# fruit-quality-classification-wgan

## What

This repository aims to solve the task of fruit quality classification given very limited data. The <a href="https://github.com/softwaremill/lemon-dataset">dataset</a> contains 2690 images of lemons which is too less to train a deep learning model, let alone use it practically.

## Why

Fruit quality detection and classification of defective fruits is an important task in the fruit packaging and processing industry. An automated system that could determine fruit quality by classifying it on the basis of surface defects is needed for enhancing fruit production, reducing
the dependency on manual labour and reducing losses due to fruit quality deterioration.


## How

We do this by augmenting the data trying many variations of GANs - cGAN, WGAN, WGAN-GP - with WGAN-GP generating the best images and giving the best results upon combining with the classification pipeline. The classification is done by trying different pretrained CNN networks - ResNet, VGG19, Xception, Inception - ResNet giving the best results of 92.11% testing accuracy. 

# DESI Legacy Imaging Surveys galaxy classification

## Introduction
This project aim to classify the galaxy images collected using DESI Legacy Imaging Surveys using convolutional neural network. We used a ResNet-50 pretrained model which was trained on ImageNet dataset. The model is finetune for 50 epochs and achived 85% accuracy on test dataset. 

## Installation

### Requirements
We have trained and tested our models on `Ubuntu 18.0`, `CUDA 11.0`, `Python 3.8`.


```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Dataset preparation
Please download the dataset from [here](https://astro.utoronto.ca/~hleung/shared/Galaxy10/Galaxy10_DECals.h5) by running follwing command and put in `dataset` folder.
```
wget https://astro.utoronto.ca/~hleung/shared/Galaxy10/Galaxy10_DECals.h5
``` 

## Training


## Evaluation


## Results


## Contact
# ASL Alphabet Recognition using CNN

This repository contains a Convolutional Neural Network (CNN) model trained to recognize American Sign Language (ASL) alphabet gestures. The model is trained on a dataset consisting of ASL alphabet images and can be used to predict the corresponding ASL alphabet letter from input images or a live video stream.

## Contents

- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)

## Model Architecture

The CNN model consists of two convolutional layers followed by max-pooling layers, fully connected layers, and softmax activation for classification.

## Dataset

The dataset used for training, testing, and validation contains ASL alphabet images. The images are preprocessed using transformations such as resizing, grayscaling, and normalization.

## Usage

1. Install the required dependencies:

   ```bash
   %pip install -r requirements.txt

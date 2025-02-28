# Create-CNN-model


# 1)  MNIST Handwritten Digit Classification with CNN (TensorFlow Keras)

This repository contains a Convolutional Neural Network (CNN) model implemented in TensorFlow Keras for classifying handwritten digits from the MNIST dataset.

## Overview

This project aims to build a CNN model capable of accurately classifying handwritten digits (0-9) from the MNIST dataset. The model utilizes convolutional layers, pooling layers, and fully connected layers to learn and recognize digit patterns.

## Dataset

The dataset used is the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits.
## Usage

1.  **Run the model:**

    Execute the Python script to train and evaluate the model. The script will perform the following steps:

    * Download and load the MNIST dataset.
    * Preprocess the images (normalize and reshape).
    * Convert labels to categorical format.
    * Define the CNN model architecture.
    * Compile and train the model.
    * Evaluate the model's performance on the training and test sets.
    * Visualize predictions on sample images.

## Model Architecture

* **Type:** Convolutional Neural Network (CNN)
* **Convolutional Layers:**
    * `Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1))`
    * `Conv2D(64, kernel_size=(3, 3), activation='relu')`
* **Pooling Layers:**
    * `MaxPooling2D((2, 2))` (after each convolutional layer)
* **Flatten Layer:** Flattens the output from the convolutional layers.
* **Fully Connected Layers:**
    * `Dense(64, activation='relu')`
    * `Dropout(0.2)`
    * `Dense(10, activation='softmax')` (output layer)
* **Loss Function:** Categorical cross-entropy
* **Optimizer:** Adam

## Training Details

* **Epochs:** 3
* **Batch Size:** 32
* **Validation Split:** Test dataset is used as validation.

## Evaluation

The model's performance is evaluated on both the training and test sets, and the following metrics are reported:

* **Accuracy**

### Results

* **Training Accuracy:** 99.38%
* **Testing Accuracy:** 98.95%

## Visualizations

The script includes visualizations to display sample images from the test set along with their predicted and true labels.

# 2) MNIST Handwritten Digit Classification with CNN (PyTorch)

This repository contains a Convolutional Neural Network (CNN) model implemented in PyTorch for classifying handwritten digits from the MNIST dataset.

## Overview

This project aims to build a CNN model capable of accurately classifying handwritten digits (0-9) from the MNIST dataset. The model utilizes convolutional layers, pooling layers, and fully connected layers to learn and recognize digit patterns.

## Dataset

The dataset used is the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits.

## Usage

1.  **Run the model:**

    Execute the Python script to train and evaluate the model. The script will perform the following steps:

    * Download and load the MNIST dataset.
    * Preprocess the images (normalize).
    * Define the CNN model architecture.
    * Train the model.
    * Evaluate the model's performance on the test set.
    * Visualize predictions on sample images.

## Model Architecture

* **Type:** Convolutional Neural Network (CNN)
* **Convolutional Layers:**
    * `Conv2d(1, 32, 3, 1)`
    * `Conv2d(32, 64, 3, 1)`
* **Pooling Layers:**
    * `MaxPool2d(2)`
* **Dropout Layers:**
    * `Dropout2d(0.25)`
    * `Dropout2d(0.5)`
* **Flatten Layer:** Flattens the output from the convolutional layers.
* **Fully Connected Layers:**
    * `Linear(9216, 128)`
    * `Linear(128, 10)`
* **Activation Functions:**
    * ReLU for convolutional and first fully connected layers.
    * Log Softmax for the output layer.
* **Loss Function:** CrossEntropyLoss
* **Optimizer:** Adam (learning rate 0.001)

## Training Details

* **Epochs:** 3
* **Batch Size (Train):** 164
* **Batch Size (Test):** 1000
* **Learning Rate:** 0.001
* **Normalization:** `transforms.Normalize((0.1307,), (0.3081,))`

## Evaluation

The model's performance is evaluated on the test set, and the following metric is reported:

* **Accuracy:** 97.87%

## Results

The CNN model achieves an accuracy of 97.87% on the MNIST test set, demonstrating its effectiveness in classifying handwritten digits.

## Visualizations

The script includes visualizations to display sample images from the test set along with their predicted and true labels.

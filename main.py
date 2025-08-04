# -*- coding: utf-8 -*-
# main.py

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Load and Preprocess the Data ---

# Load the MNIST dataset, which is conveniently included with Keras.
# The dataset is already split into training and testing sets.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Data Loaded Successfully")
print(f"Initial training data shape: {x_train.shape}")
print(f"Initial test data shape: {x_test.shape}")

# Preprocessing the image data:
# CNNs expect a specific input shape. For image data, this is typically
# (batch_size, height, width, channels).
# The MNIST images are grayscale, so they have 1 color channel.

# Reshape the data to include the channel dimension.
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Normalize the pixel values. The pixel values in the images range from 0 to 255.
# Normalizing them to a range of 0 to 1 helps the model train faster and more effectively.
# We do this by dividing by the maximum pixel value (255).
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print("Data Preprocessed Successfully")
print(f"New training data shape: {x_train.shape}")
print(f"New test data shape: {x_test.shape}")

# Preprocessing the labels (the target digits 0-9):
# We need to convert the labels from single digits (e.g., 5) to a
# "one-hot encoded" vector. For 10 digits, this means a vector of length 10.
# For the digit 5, the vector would be [0, 0, 0, 0, 0, 1, 0, 0, 0, 0].
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

print(f"Example original label: {np.argmax(y_train[0])}")
print(f"Example one-hot encoded label: {y_train[0]}")

"""# --- 2. Build the Convolutional Neural Network (CNN) Model ---

"""

# We'll use a Sequential model, which is a linear stack of layers.
model = Sequential([
    # First Convolutional Layer:
    # - Conv2D: This layer creates convolution kernels that are convolved with the layer input to produce a tensor of outputs.
    # - 32 filters: The number of output filters in the convolution.
    # - (3, 3) kernel_size: The height and width of the convolution window.
    # - 'relu' activation: Rectified Linear Unit, a common activation function that helps with non-linearity.
    # - input_shape: The shape of the input data (28x28 pixels, 1 channel).
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),

    # MaxPooling Layer:
    # This layer downsamples the input along its spatial dimensions (height and width)
    # by taking the maximum value over an input window. This helps to make the
    # model more robust to variations in the position of features in the image.
    MaxPooling2D(pool_size=(2, 2)),

    # Second Convolutional Layer:
    # We add another convolutional layer to learn more complex features.
    Conv2D(64, kernel_size=(3, 3), activation='relu'),

    # Second MaxPooling Layer:
    MaxPooling2D(pool_size=(2, 2)),

    # Flatten Layer:
    # This layer flattens the 2D output from the convolutional layers into a 1D vector.
    # This is necessary to connect the convolutional layers to the dense layers.
    Flatten(),

    # Dense Layer (Fully Connected Layer):
    # This is a standard fully connected neural network layer.
    # - 128 units: The number of neurons in the layer.
    Dense(128, activation='relu'),

    # Dropout Layer:
    # Dropout is a regularization technique to prevent overfitting. It randomly sets
    # a fraction of input units to 0 at each update during training time.
    # 0.5 means 50% of the neurons will be dropped out during training.
    Dropout(0.5),

    # Output Layer:
    # - Dense layer with 10 units (one for each digit 0-9).
    # - 'softmax' activation: This activation function outputs a probability distribution
    #   over the 10 classes, making it ideal for multi-class classification.
    Dense(10, activation='softmax')
])

"""# --- 3. Compile the Model ---

"""

# Before training, we need to configure the learning process.
# - optimizer='adam': Adam is an efficient and popular optimization algorithm.
# - loss='categorical_crossentropy': This is the loss function for multi-class classification with one-hot encoded labels.
# - metrics=['accuracy']: We want to monitor the accuracy of the model during training.
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print a summary of the model's architecture
model.summary()

"""# --- 4. Train the Model ---

"""

# We now train the model using the training data.
# - x_train, y_train: The training data and labels.
# - batch_size=128: The number of samples per gradient update.
# - epochs=10: The number of times to iterate over the entire training dataset.
# - validation_split=0.1: Fraction of the training data to be used as validation data.
#   The model will not train on this data, but it will be used to evaluate the loss
#   and any model metrics at the end of each epoch.
history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=10,
                    verbose=1,
                    validation_split=0.1)

"""# --- 5. Evaluate the Model ---

"""

print("\nEvaluating the Model...")

# We evaluate the trained model on the test dataset.
score = model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss: {score[0]:.4f}')
print(f'Test accuracy: {score[1]:.4f}')

# This is the crucial new step. It saves the entire model (architecture, weights,
# optimizer state) to a single HDF5 file.
import os
model_filename = 'digit_recognizer_model.h5'
model.save(model_filename)
print(f"\nModel saved successfully as {model_filename}")
print(f"Full path: {os.path.abspath(model_filename)}")

"""# --- 6. Visualize Training History and Make Predictions ---

"""



# Plotting the training and validation accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plotting the training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Let's look at some of the model's predictions
print("\nMaking some predictions...")
predictions = model.predict(x_test)

# Display the first 10 test images, their predicted labels, and their true labels.
plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(5, 2, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    # Get the predicted digit by finding the index with the highest probability
    predicted_label = np.argmax(predictions[i])
    # Get the true digit
    true_label = np.argmax(y_test[i])
    plt.title(f"Predicted: {predicted_label}, True: {true_label}")
    plt.axis('off')
plt.tight_layout()
plt.show()


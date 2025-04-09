import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load datasets
train_df = pd.read_csv("./data/mnist_train.csv")
test_df = pd.read_csv("./data/mnist_test.csv")

# Extract labels and images
X_train = train_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0  # Normalize and reshape
y_train = train_df.iloc[:, 0].values

X_test = test_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0  # Normalize and reshape
y_test = test_df.iloc[:, 0].values

# Convert labels to categorical
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Build CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)

# Save model
model.save("cnn_digit_model.keras")

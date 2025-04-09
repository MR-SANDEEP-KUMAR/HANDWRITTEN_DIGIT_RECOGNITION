# import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"



import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# Load training dataset
print("Loading training data!")
train_file_path = "./data/mnist_train.csv"
train_data = pd.read_csv(train_file_path)
print("Training Data loaded!")
# Split into features (X) and labels (y)
X_train = train_data.iloc[:, 1:].values  # Pixel values (excluding label column)
y_train = train_data.iloc[:, 0].values   # Labels (digits 0-9)

# Normalize pixel values (0-255 → 0-1)
X_train = X_train / 255.0  

# Convert labels to one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes=10)

# Load test dataset
test_file_path = "./data/mnist_test.csv"
test_data = pd.read_csv(test_file_path)

X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

X_test = X_test / 255.0  # Normalize
y_test = keras.utils.to_categorical(y_test, num_classes=10)

print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
print(f"Test data shape: {X_test.shape}, Labels shape: {y_test.shape}")


# Define the deep MLP model with explicit Input layer
model = keras.Sequential([
    keras.layers.Input(shape=(784,)),  # Explicit input layer
    keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    # keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    # keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")


print("Saving model to 'mlp_model'")
model.save("mlp_digit_model.keras")

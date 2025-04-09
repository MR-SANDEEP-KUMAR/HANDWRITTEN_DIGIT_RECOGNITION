from tensorflow import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the trained CNN model
model = keras.models.load_model("cnn_digit_model.keras")

# Load the image in grayscale
image_path = "digit.png"  # Update with your actual image path
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Image file not found at {image_path}")

# Invert colors if necessary (white background, black digit)
img = 255 - img

# Threshold the image to create a binary mask
_, img_thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

# Find contours of the digit
contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(contours) == 0:
    raise ValueError("No digit found in the image")

# Get bounding box of the largest contour (assumed to be the digit)
x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
digit_cropped = img[y:y+h, x:x+w]

# Resize the cropped digit while maintaining aspect ratio
margin = 4  # Add some padding to prevent edge touching
canvas_size = 28
h, w = digit_cropped.shape
scale = (canvas_size - 2 * margin) / max(h, w)
digit_resized = cv2.resize(digit_cropped, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

# Center the digit on a blank canvas
canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
y_offset = (canvas_size - digit_resized.shape[0]) // 2
x_offset = (canvas_size - digit_resized.shape[1]) // 2
canvas[y_offset:y_offset+digit_resized.shape[0], x_offset:x_offset+digit_resized.shape[1]] = digit_resized

# Normalize pixel values to range [0, 1]
img_normalized = canvas / 255.0

# Reshape to match CNN input shape: (1, 28, 28, 1)
img_input = img_normalized.reshape(1, 28, 28, 1)

# Display preprocessed image
plt.imshow(img_normalized, cmap='gray')
plt.title("Preprocessed Image")
plt.axis("off")
plt.show()

# Predict the digit using the model
prediction = model.predict(img_input)
digit = np.argmax(prediction)
print(f"Predicted Digit: {digit}")

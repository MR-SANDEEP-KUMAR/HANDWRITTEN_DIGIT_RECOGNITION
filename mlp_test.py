import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Load the trained MLP model
model = keras.models.load_model("mlp_digit_model.keras")

# Load and preprocess the image
image_path = "digit.png"  # Update with your actual image path
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Image file not found at {image_path}")

# Apply thresholding to remove noise and improve contrast
_, img_thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

# Find contours to detect the digit's bounding box
contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    # Get bounding box of largest contour (assuming single digit per image)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    img_cropped = img_thresh[y:y+h, x:x+w]  # Crop the digit
else:
    img_cropped = img_thresh  # Use original thresholded image if no contours found

# Resize cropped digit to 20x20 while maintaining aspect ratio
h, w = img_cropped.shape
scale = 20 / max(w, h)
w_new, h_new = int(w * scale), int(h * scale)
img_resized = cv2.resize(img_cropped, (w_new, h_new), interpolation=cv2.INTER_AREA)

# Create a blank 28x28 image and center the resized digit
img_padded = np.zeros((28, 28), dtype=np.uint8)
y_offset = (28 - h_new) // 2
x_offset = (28 - w_new) // 2
img_padded[y_offset:y_offset+h_new, x_offset:x_offset+w_new] = img_resized

# Normalize pixel values to range 0-1 and convert to float32
img_normalized = (img_padded / 255.0).astype('float32')

# Flatten to match MLP input shape (1, 784)
img_flattened = img_normalized.flatten().reshape(1, 784)

print("Preprocessed image shape:", img_flattened.shape)

# Predict the digit
prediction = model.predict(img_flattened)
digit = np.argmax(prediction)
print(f"\nPredicted Digit: {digit}")

# Display the preprocessing steps
plt.figure(figsize=(12, 6))
titles = ["Original", "Thresholded", "Cropped", "Resized & Centered"]
images = [img, img_thresh, img_cropped, img_padded]

for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()

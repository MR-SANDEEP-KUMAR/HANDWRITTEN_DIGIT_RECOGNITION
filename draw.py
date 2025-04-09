import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the trained CNN model
model = keras.models.load_model("cnn_digit_model.keras")

# Constants for canvas size
CANVAS_SIZE = 400
IMG_SIZE = 28

# Global variable to store drawn points
drawing = False
canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)

# Mouse callback function
def draw(event, x, y, flags, param):
    global drawing, canvas
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(canvas, (x, y), 10, 255, -1)  # Draw white circle
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# Function to preprocess and predict the digit
def predict_digit(image):
    # Resize and normalize the image
    img_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    
    # Invert colors if necessary (black background, white digit)
    img_processed = 255 - img_resized

    # Normalize pixel values to [0,1]
    img_normalized = img_processed / 255.0

    # Reshape to match CNN input shape: (1, 28, 28, 1)
    img_input = img_normalized.reshape(1, 28, 28, 1)

    # Predict the digit
    prediction = model.predict(img_input)
    digit = np.argmax(prediction)
    
    return digit

# Create OpenCV window and set mouse callback
cv2.namedWindow("Draw a Digit")
cv2.setMouseCallback("Draw a Digit", draw)

while True:
    # Show the canvas
    # break
    cv2.imshow("Draw a Digit", canvas)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF
    
    if key == 13:  # Enter key: Predict digit
        print(canvas.size)
        digit = predict_digit(canvas)
        print(f"Predicted Digit: {digit}")
        cv2.putText(canvas, f"Predicted: {digit}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

    elif key == ord('r'):  # 'r' key: Reset canvas
        canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)

    elif key == ord('q'):  # 'q' key: Quit
        break

# Close OpenCV windows
cv2.destroyAllWindows()

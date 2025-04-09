import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
# file_path = "./data/mnist_test.csv"  # Update with your file path

file_path = str(input("Enter the file path: "))
data = pd.read_csv(file_path)

num_images = len(data) # Number of rows (images)
current_index = 0  # Index of the current image to display

# Function to display an image
def show_image(index):
    sample = data.iloc[index]
    label = sample[0]  # Assuming the first column is the label
    pixels = sample[1:].values.reshape(28, 28)  # Reshape to 28x28

    plt.clf()  # Clear previous image
    plt.imshow(pixels, cmap="gray")
    plt.title(f"Label: {label} (Image {index+1}/{num_images})")
    plt.axis("off")
    plt.draw()

# Function to handle key press events
def on_key(event):
    global current_index
    if event.key == "right":
        current_index = (current_index + 1) % num_images  # Next image
    elif event.key == "left":
        current_index = (current_index - 1) % num_images  # Previous image
    show_image(current_index)

# Create figure and show first image
fig = plt.figure()
show_image(current_index)

# Connect key press event
fig.canvas.mpl_connect("key_press_event", on_key)

plt.show()

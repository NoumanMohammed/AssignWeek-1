import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('images/drdre.jpg')  # Replace with your actual image file
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format

# Display the original image
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()


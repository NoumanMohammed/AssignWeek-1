import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('images/drdre.jpg')  # Replace with your actual image file
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format

# Display the image
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()


# ========================================
# STEP 2: ROTATE THE IMAGE
# ========================================

# Define rotation parameters
(h, w) = image.shape[:2]  # Get image height and width
center = (w // 2, h // 2)  # Find the center of the image
angle = 45  # Rotate by 45 degrees
scale = 1.0  # Keep the scale the same

# Create the rotation matrix
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

# Apply the rotation
rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

# Display the rotated image
plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
plt.title("Rotated Image (45°)")
plt.axis("off")
plt.show()


# ========================================
# STEP 3: SCALE THE IMAGE
# ========================================

# Scale the image by 1.5x
scaled_image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

# Display the scaled image
plt.imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB))
plt.title("Scaled Image (1.5x)")
plt.axis("off")
plt.show()


# ========================================
# STEP 4: SIMULATE DIFFERENT FOCAL LENGTHS (FIXED VERSION)
# ========================================

# Simulate different zoom levels to represent focal length changes
focal_lengths = [50, 100, 200]
zoom_factors = [0.7, 1.0, 1.5]  # Corresponding zoom levels

plt.figure(figsize=(12, 4))

for i, (f, zoom) in enumerate(zip(focal_lengths, zoom_factors)):
    # Simulate focal length by zooming/cropping
    zoomed_h, zoomed_w = int(h / zoom), int(w / zoom)
    
    # Calculate crop coordinates (centered)
    start_h = (h - zoomed_h) // 2
    start_w = (w - zoomed_w) // 2
    
    # Crop the image
    cropped = image[start_h:start_h+zoomed_h, start_w:start_w+zoomed_w]
    
    # Resize back to original size to show the zoom effect
    focal_sim = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    
    plt.subplot(1, 3, i+1)
    plt.imshow(cv2.cvtColor(focal_sim, cv2.COLOR_BGR2RGB))
    plt.title(f"Focal Length: {f}mm")
    plt.axis("off")

plt.tight_layout()
plt.show()
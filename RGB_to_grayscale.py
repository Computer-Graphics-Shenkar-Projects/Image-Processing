import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def display_image(image):
  """
  Display a 3D (RGB) image.
  """
  plt.imshow(image)
  plt.axis('off')
  plt.show()

def display_gray_image(image, title=""):
    """
    Displays a 2D (grayscale) image with a given title.
    """
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load the image
image = cv2.imread('/content/sample_data/Lenna.png')

# Check if the image was successfully loaded
if image is None:
    print("Error: image is None. Please check the file path/filename.")
else:
    # Convert to RGB for display
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    display_image(image)

# Grayscale.png
def RGB_to_grayscale(image_path):
    """
    Load image at image_path, manually compute a grayscale version without using cmap='gray'.
    """
    # Manually extract B, G, R channels (and convert to float for the math)
    B = image[:, :, 0].astype(np.float32)
    G = image[:, :, 1].astype(np.float32)
    R = image[:, :, 2].astype(np.float32)

    # Apply the standard luminance (grayscale) formula
    gray_float = 0.2989 * R + 0.5870 * G + 0.1140 * B

    # Convert to unsigned 8-bit integers (0-255)
    gray_uint8 = gray_float.astype(np.uint8)

    # Replicate grayscale values across R, G, and B channels
    # so that we can display it as a color image but it looks gray
    gray_3ch = np.stack([gray_uint8, gray_uint8, gray_uint8], axis=-1)

    # Display the resulting 3-channel grayscale image with default colormap
    # display_image(gray_3ch)

    # Save the grayscale image
    cv2.imwrite('Grayscale.png', gray_3ch)

    return gray_uint8

# Run the RGB_to_grayscale function
RGB_to_grayscale(image)

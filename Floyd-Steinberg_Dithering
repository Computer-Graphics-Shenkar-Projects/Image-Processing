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

# FloyedSteinberg.png
def floyd_steinberg_dithering(image_path):
    """
    Apply Floyd-Steinberg dithering to a grayscale image, reducing
    intensity levels from 256 to 16 levels using error diffusion.
    """
    height, width = image_path.shape
    # Define 16 intensity levels
    levels = np.linspace(0, 255, 16).astype(np.uint8)

    # Copy the image to avoid modifying the original
    dithered_image = image_path.astype(np.float32)

    # Floyd-Steinberg error diffusion
    for y in range(height):
        for x in range(width):
            old_pixel = dithered_image[y, x]

            # Find closest grayscale level
            new_pixel = levels[np.argmin(np.abs(levels - old_pixel))]
            dithered_image[y, x] = new_pixel

            # Compute the quantization error
            error = old_pixel - new_pixel

            # Distribute the error to neighboring pixels
            if x + 1 < width:   # Right neighbor
                dithered_image[y, x + 1] += error * (7 / 16)
            if y + 1 < height:
                if x > 0:        # Bottom-left neighbor
                    dithered_image[y + 1, x - 1] += error * (3 / 16)
                # Bottom neighbor
                dithered_image[y + 1, x] += error * (5 / 16)
                if x + 1 < width: # Bottom-right neighbor
                    dithered_image[y + 1, x + 1] += error * (1 / 16)

    # Ensure pixel values remain in the 0-255 range
    dithered_image = np.clip(dithered_image, 0, 255).astype(np.uint8)

    return dithered_image

# Run the floyd_steinberg_dithering function
grayscale_image = RGB_to_grayscale(image)
dithered_image = floyd_steinberg_dithering(grayscale_image)

# Save and display the dithered image
cv2.imwrite('FloyedSteinberg.png', dithered_image)
display_gray_image(dithered_image, "Floyd-Steinberg Dithering")

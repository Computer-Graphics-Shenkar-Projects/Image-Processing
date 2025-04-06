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

def halftone_2x2_quadrants(input_path, output_path='Halftone.png'):
    """
    Reads a grayscale image, converts each pixel to one of 5 levels (0..4),
    and then maps each level to a 2x2 pattern of black (0) or white (255).
    The final layout of black squares in the 2x2 block matches the
    "one circle in bottom-left, two circles bottom-left+top-right," etc.
    from the lecture's diagram.
    """
    # 1) Read the image as single-channel (grayscale)
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")

    H, W = image.shape

    # 2) Create a bigger array (2H x 2W) for the result
    halftone = np.zeros((2*H, 2*W), dtype=np.uint8)

    # 3) Define the 2x2 patterns (using squares, not literal circles).
    #    We will treat 0=black and 255=white, so "circle" = black quadrant.
    #    The layout corresponds to the lecture's figure:
    #      - Level 0 => no black quadrants  (all white)
    #      - Level 1 => black in bottom-left
    #      - Level 2 => black in bottom-left, top-right
    #      - Level 3 => black in bottom-left, top-left, top-right
    #      - Level 4 => black in all four quadrants
    patterns = {
        0: np.array([[255, 255],
                     [255, 255]], dtype=np.uint8),
        1: np.array([[255, 255],
                     [0,   255]], dtype=np.uint8),  # bottom-left black
        2: np.array([[255,   0],
                     [0,   255]], dtype=np.uint8),  # bottom-left, top-right
        3: np.array([[0,     0],
                     [0,   255]], dtype=np.uint8),  # all but bottom-right
        4: np.array([[0,     0],
                     [0,     0]], dtype=np.uint8),  # all black
    }

    # Helper to map grayscale [0..255] -> 0..4 (based on the lecture's scale)
    def get_level(pixel_value):
        intensity = pixel_value / 255.0
        if intensity < 0.2:
            return 0
        elif intensity < 0.4:
            return 1
        elif intensity < 0.6:
            return 2
        elif intensity < 0.8:
            return 3
        else:
            return 4

    # 4) Fill in the halftone result
    for r in range(H):
        for c in range(W):
            level = get_level(image[r, c])
            block = patterns[level]
            # Top-left corner in the output
            rr = 2*r
            cc = 2*c
            halftone[rr:rr+2, cc:cc+2] = block

    # 5) Save and Display
    cv2.imwrite(output_path, halftone)

    return halftone

# Run the halftone_2x2_quadrants function
halftone_2x2_quadrants('/content/Grayscale.png')

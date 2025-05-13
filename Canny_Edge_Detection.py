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

# Canny.png
def create_gaussian_kernel(shape=(5, 5), sigma=1.4):
    """
    After converting the image to Grayscale we will use Gaussian filter to reduction the noise.
    """
    rows, cols = shape

    # Create a range of x-coordinates: from -cols//2 to +cols//2
    x_lin = np.linspace(-(cols // 2), (cols // 2), cols)
    # Create a range of y-coordinates: from -rows//2 to +rows//2
    y_lin = np.linspace(-(rows // 2), (rows // 2), rows)

    # Create a coordinate grid (xx, yy)
    xx, yy = np.meshgrid(x_lin, y_lin)

    # Apply the 2D Gaussian formula: exp(-(x^2 + y^2)/(2*sigma^2))
    # No normalization constant is strictly required, because we will normalize by sum anyway.
    kernel = np.exp(- (xx**2 + yy**2) / (2.0 * sigma**2))

    # Normalize so sum of all kernel elements is 1
    kernel /= np.sum(kernel)
    print("Kernel sum =", np.sum(kernel))

    return kernel

def convolve2d(image, kernel):
    """
    Convolve a 2D image with a 2D kernel using Gaussian kernel (both are NumPy arrays).
    We will do this manually, using zero-padding, no built-in convolution functions.
    """
    H, W = image.shape
    kH, kW = kernel.shape

    # Calculate how many pixels to pad on each side
    pad_h = kH // 2
    pad_w = kW // 2

    # Create an output array of the same size as the input image
    output = np.zeros_like(image, dtype=np.float32)

    # Zero-pad the input image
    padded_img = np.zeros((H + 2 * pad_h, W + 2 * pad_w), dtype=np.float32)
    padded_img[pad_h:pad_h + H, pad_w:pad_w + W] = image

    # Perform the convolution
    for i in range(H):
        for j in range(W):
            # Extract the region of interest (ROI) in the padded image
            roi = padded_img[i : i + kH, j : j + kW]
            # Element-wise multiply and sum
            val = np.sum(roi * kernel)
            output[i, j] = val

    return output

def sobel_filters(image):
    """
    To detect edges, we compute partial derivatives in both the x and y directionss.
    Compute gradient magnitudes and directions using Sobel operator.
    Returns (gradient_magnitude, gradient_direction).
    - gradient_direction is in degrees, range: -180..180
    """
    # Sobel operators for x and y
    # Sx detects vertical edges, Sy detects horizontal edges
    Kx = np.array([[-1, 0,  1],
                   [-2, 0,  2],
                   [-1, 0,  1]], dtype=np.float32)
    Ky = np.array([[ 1,  2,  1],
                   [ 0,  0,  0],
                   [-1, -2, -1]], dtype=np.float32)

    # Convolve with each kernel
    Gx = convolve2d(image, Kx)
    Gy = convolve2d(image, Ky)

    # Gradient magnitude
    magnitude = np.sqrt(Gx**2 + Gy**2)
    # Gradient direction (in degrees)
    direction = np.arctan2(Gy, Gx) * (180.0 / np.pi)

    return (magnitude, direction)

def non_maximum_suppression(magnitude, direction):
    """
    Thins out edges. For each pixel, we look along the gradient direction
    and suppress (set to 0) anything that is not a local maximum.
    Returns a 2D float array (the thinned magnitudes).
    """
    H, W = magnitude.shape
    # Initialize result array
    Z = np.zeros((H, W), dtype=np.float32)

    # Convert direction into [0, 180)
    direction_mod = direction % 180

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            angle = direction_mod[i, j]
            mag = magnitude[i, j]

            # Determine which neighbors to compare
            # 0 degrees
            if (0 <= angle < 22.5) or (157.5 <= angle < 180):
                neighbor1 = magnitude[i, j - 1]
                neighbor2 = magnitude[i, j + 1]
            # 45 degrees
            elif (22.5 <= angle < 67.5):
                neighbor1 = magnitude[i - 1, j + 1]
                neighbor2 = magnitude[i + 1, j - 1]
            # 90 degrees
            elif (67.5 <= angle < 112.5):
                neighbor1 = magnitude[i - 1, j]
                neighbor2 = magnitude[i + 1, j]
            # 135 degrees
            else:
                neighbor1 = magnitude[i - 1, j - 1]
                neighbor2 = magnitude[i + 1, j + 1]

            # Keep the pixel if it's >= both neighbors
            if (mag >= neighbor1) and (mag >= neighbor2):
                Z[i, j] = mag
            else:
                Z[i, j] = 0.0

    return Z

def double_threshold_and_hysteresis(img, low_ratio=0.05, high_ratio=0.15):
    """
    1) Classify pixels based on two thresholds:
       - strong (>= highThresh)
       - weak (between lowThresh and highThresh)
       - non-edge (< lowThresh)
    2) Hysteresis: if a weak pixel is connected to a strong pixel, it's kept.
    Returns a binary (0 or 255) edge map.
    """
    # 1) Determine thresholds
    high_thresh = img.max() * high_ratio
    low_thresh = high_thresh * low_ratio

    # Prepare output
    res = np.zeros_like(img, dtype=np.uint8)

    strong_val = 255
    weak_val = 50  # an intermediate label for weak edges

    # Classify pixels
    strong_i, strong_j = np.where(img >= high_thresh)
    weak_i, weak_j = np.where((img <= high_thresh) & (img >= low_thresh))

    res[strong_i, strong_j] = strong_val
    res[weak_i, weak_j] = weak_val

    # 2) Hysteresis
    H, W = img.shape
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if res[i, j] == weak_val:
                # If any neighbor is strong_val, make this pixel strong
                if ((res[i+1, j-1] == strong_val) or (res[i+1, j] == strong_val) or (res[i+1, j+1] == strong_val)
                    or (res[i, j-1] == strong_val)   or (res[i, j+1] == strong_val)
                    or (res[i-1, j-1] == strong_val) or (res[i-1, j] == strong_val) or (res[i-1, j+1] == strong_val)):
                    res[i, j] = strong_val
                else:
                    res[i, j] = 0

    return res

def canny_edge_detector(image_path):
    """
    Full pipeline for custom Canny Edge Detection,
    displaying each step as an image but avoiding repetitive plotting code.
    """
    # 1) Load + Grayscale
    gray = RGB_to_grayscale(image_path)
    display_gray_image(gray, title="1) Grayscale Image")

    # 2) Gaussian blur
    gauss_kernel = create_gaussian_kernel(shape=(5,5), sigma=1.4)
    smoothed = convolve2d(gray, gauss_kernel)
    display_gray_image(smoothed, title="2) After Gaussian Blur")
    cv2.imwrite('Gaussian_blur.png', smoothed)

    # 3) Sobel gradients
    magnitude, direction = sobel_filters(smoothed)
    display_gray_image(magnitude, title="3) Gradient Magnitude (Sobel)")
    cv2.imwrite('Sobel_filter.png', magnitude)

    # 4) Non-Maximum Suppression
    nms_result = non_maximum_suppression(magnitude, direction)
    display_gray_image(nms_result, title="4) Non-Maximum Suppression")
    cv2.imwrite('Non-Maximum_Suppression.png', nms_result)

    # 5) Double Threshold + Hysteresis
    final_edges = double_threshold_and_hysteresis(nms_result, low_ratio=0.05, high_ratio=0.15)
    display_gray_image(final_edges, title="5) Final Edges (Double Threshold + Hysteresis)")

    return final_edges

# Run the canny_edge_detector function
result = canny_edge_detector(image)
# Ensure the result is in a suitable format, e.g. uint8
saveable_result = result.astype(np.uint8)

# Save the Canny image
cv2.imwrite('Canny.png', saveable_result)

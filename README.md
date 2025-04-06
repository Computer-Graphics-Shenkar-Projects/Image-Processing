# Image Filters Assignment
This project is an implementation of basic image processing filters using Python. The input is a classic RGB .png image of Lenna.

# Objective
Your main task is to process the Lenna image and generate four output images, each showcasing a different filtering technique.

# Output Images
Grayscale.png: 
Converts the original RGB image to grayscale. Each pixel will have a value between 0 (black) and 255 (white).

Canny.png:
Applies Canny Edge Detection to highlight image edges. The output is a binary image where each pixel is 0 (black) or 255 (white).

🔍 Canny Implementation Steps:
1. Noise reduction using Gaussian filter,

2. Compute gradient magnitudes along X and Y axes, then combine,

3. Non-Maximum Suppression to remove spurious edges,

4. Apply Double Thresholding and Edge Tracking by Hysteresis.

Halftone.png:
Applies a Halftone effect using 2×2 square patterns. Each grayscale pixel is expanded into a 2×2 black and white matrix, simulating a 4-level grayscale.

FloydSteinberg.png:
Implements Floyd-Steinberg Dithering, reducing the image to 16 grayscale intensity levels.

Feel free to fork, explore, and enhance!
Happy filtering! 🎨✨

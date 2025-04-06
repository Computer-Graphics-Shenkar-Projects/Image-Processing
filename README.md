🖼️ Image Filters Assignment
This project is an implementation of basic image processing filters using either C++ or Python (your choice). The input is a classic RGB .png image of Lenna, included in the assignment ZIP file.

📌 Objective
Your main task is to process the Lenna image and generate four output images, each showcasing a different filtering technique.

📂 Output Images
Grayscale.png
Converts the original RGB image to grayscale.

Each pixel will have a value between 0 (black) and 255 (white).

Canny.png
Applies Canny Edge Detection to highlight image edges.

The output is a binary image where each pixel is 0 (black) or 255 (white).

🔍 Canny Implementation Steps:
Noise reduction using Gaussian filter

Compute gradient magnitudes along X and Y axes, then combine

(Bonus) Non-Maximum Suppression to remove spurious edges

(Bonus) Apply Double Thresholding and Edge Tracking by Hysteresis

Halftone.png
Applies a Halftone effect using 2×2 square patterns.

Each grayscale pixel is expanded into a 2×2 black and white matrix, simulating a 4-level grayscale.

💡 Tips:
Create a new image with dimensions 512×512

Process two rows at a time

Use predefined 2×2 halftone masks for grayscale values from 0–255 mapped to 0–4 levels

FloydSteinberg.png
Implements Floyd-Steinberg Dithering, reducing the image to 16 grayscale intensity levels.

📐 Error Diffusion:
Use the Floyd–Steinberg formula to spread the quantization error to neighboring pixels.

If you reach an image edge, divide the error among the available neighbors accordingly.

🧰 Tools & Libraries
You may use common libraries for image I/O and basic operations (e.g., OpenCV, PIL, stb_image, etc.) but all filters must be implemented manually.

📝 Notes
Make sure to follow good coding practices and modularize your code.

Add comments where necessary to clarify your implementation logic.

Bonus tasks are optional but encouraged for better results and learning.

Feel free to fork, explore, and enhance!
Happy filtering! 🎨✨

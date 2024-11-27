import cv2
import numpy as np
from PIL import Image, ImageDraw


def detect_and_draw_horizontal_lines(image_path):
    """
    Detect horizontal lines in an image using OpenCV and draw them using PIL.

    Args:
        image_path (str): Path to the input image

    Returns:
        None: Saves the processed image as 'draw_output.png'
    """
    # Read image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image from provided path")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Apply HoughLinesP to detect lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )

    # Convert OpenCV image to PIL Image for drawing
    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # Filter and draw horizontal lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate line angle
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)

            # Consider lines with angle close to 0 or 180 degrees as horizontal
            if angle < 10 or angle > 170:
                # Draw line in red color with width=3
                draw.line([(x1, y1), (x2, y2)], fill='red', width=3)

    # Save the result
    pil_image.save('draw_output.png')

if __name__ == '__main__':
    image_path:str = "/home/ntlpt59/MAIN/experiments/checkbox_optimized/data/Export-Bill_filled_sample0.jpg"
    detect_and_draw_horizontal_lines(image_path)
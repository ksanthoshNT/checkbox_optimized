from PIL import Image, ImageDraw
import numpy as np
import cv2
from typing import List, Tuple

from src.checkbox_fixings.main import ImageProcessor


def detect_horizontal_lines(binary_array: np.ndarray,
                            min_line_length: int = 50,
                            max_line_gap: int = 2,
                            min_points_in_line: int = 40) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Detect horizontal lines from binary array where 1 represents black pixels.
    """
    height, width = binary_array.shape
    horizontal_lines = []

    # Iterate through each row
    for y in range(height):
        line_start = None
        consecutive_ones = 0
        gap_count = 0

        for x in range(width):
            if binary_array[y, x] == 1:
                if line_start is None:
                    line_start = (x, y)
                consecutive_ones += 1
                gap_count = 0
            else:
                if line_start is not None:
                    if gap_count > max_line_gap:
                        if consecutive_ones >= min_points_in_line:
                            horizontal_lines.append((line_start, (x - gap_count - 1, y)))
                        line_start = None
                        consecutive_ones = 0
                    gap_count += 1

        # Check for line ending at image boundary
        if line_start is not None and consecutive_ones >= min_points_in_line:
            horizontal_lines.append((line_start, (width - 1, y)))

    # Filter lines by minimum length
    horizontal_lines = [line for line in horizontal_lines
                        if abs(line[1][0] - line[0][0]) >= min_line_length]

    return horizontal_lines


def draw_lines_on_image(image_path: str,
                        lines: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                        output_path: str = 'draw_output.png',
                        line_color: str = 'red',
                        line_width: int = 3):
    """
    Draw detected lines on the image using PIL

    Args:
        image_path: Path to original image
        lines: List of lines in format [((x1,y1), (x2,y2)), ...]
        output_path: Path to save the output image
        line_color: Color of the lines to draw
        line_width: Width of the lines
    """
    # Read original image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image from provided path")

    # Convert to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # Draw each line
    for (x1, y1), (x2, y2) in lines:
        draw.line([(x1, y1), (x2, y2)], fill=line_color, width=line_width)

    # Save the result
    pil_image.save(output_path)
    return pil_image


# Usage example
if __name__ == '__main__':
    processor = ImageProcessor()
    image_path = "/home/ntlpt59/MAIN/experiments/checkbox_optimized/data/Export-Bill_filled_sample0.jpg"
    binary_arr = processor.get_binary_image(image_path=image_path)

    # Detect horizontal lines
    horizontal_lines = detect_horizontal_lines(binary_arr,
                                               min_line_length=50,
                                               max_line_gap=0,
                                               min_points_in_line=40)

    # Draw lines using PIL
    result_image = draw_lines_on_image(image_path,
                                       horizontal_lines,
                                       output_path='draw_output.png',
                                       line_color='blue',
                                       line_width=1)

    # Print line coordinates
    for i, ((x1, y1), (x2, y2)) in enumerate(horizontal_lines):
        print(f"Line {i + 1}: ({x1}, {y1}) to ({x2}, {y2})")
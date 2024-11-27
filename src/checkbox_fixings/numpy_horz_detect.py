from typing import List, Tuple

import cv2
import numpy as np

from src.checkbox_fixings.main import ImageProcessor


def detect_horizontal_lines(binary_array: np.ndarray,
                            min_line_length: int = 50,
                            max_line_gap: int = 2,
                            min_points_in_line: int = 40) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Detect horizontal lines from binary array where 1 represents black pixels.

    Args:
        binary_array: numpy array where 1 is black (line) and 0 is white (background)
        min_line_length: minimum required length for a line
        max_line_gap: maximum gap between points to be considered same line
        min_points_in_line: minimum number of points required to consider as line

    Returns:
        List of lines, where each line is represented as ((x1,y1), (x2,y2))
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


# Function to draw lines on original image
def draw_detected_lines(image: np.ndarray,
                        lines: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                        color: Tuple[int, int, int] = (0, 0, 255),
                        thickness: int = 2) -> np.ndarray:
    """
    Draw detected lines on the image

    Args:
        image: Original image
        lines: List of lines in format [((x1,y1), (x2,y2)), ...]
        color: BGR color tuple
        thickness: Line thickness

    Returns:
        Image with drawn lines
    """
    result = image.copy()
    for (x1, y1), (x2, y2) in lines:
        cv2.line(result, (x1, y1), (x2, y2), color, thickness)
    return result


# Usage example to add to your main code:
if __name__ == '__main__':
    processor = ImageProcessor()
    binary_arr = processor.get_binary_image(
        image_path="/home/ntlpt59/MAIN/experiments/checkbox_optimized/data/Export-Bill_filled_sample0.jpg")

    # Detect horizontal lines
    horizontal_lines = detect_horizontal_lines(binary_arr,
                                               min_line_length=50,
                                               max_line_gap=2,
                                               min_points_in_line=40)

    # Draw lines on original image
    result_image = draw_detected_lines(processor._image, horizontal_lines)

    # Save result
    cv2.imwrite('horizontal_lines_output.jpg', result_image)

    # Print line coordinates
    for i, ((x1, y1), (x2, y2)) in enumerate(horizontal_lines):
        print(f"Line {i + 1}: ({x1}, {y1}) to ({x2}, {y2})")
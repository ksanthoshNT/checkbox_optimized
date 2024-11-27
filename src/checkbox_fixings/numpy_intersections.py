from PIL import Image, ImageDraw
import numpy as np
import cv2
from typing import List, Tuple, Dict
from dataclasses import dataclass

from src.checkbox_fixings.main import ImageProcessor


@dataclass
class LineDetectionConfig:
    min_line_length: int = 50
    max_line_gap: int = 2
    min_points_in_line: int = 40


@dataclass
class DrawConfig:
    line_color: str = 'red'
    line_width: int = 3
    point_color: str = 'blue'
    point_radius: int = 5


def detect_horizontal_lines(binary_array: np.ndarray,
                            config: LineDetectionConfig) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Detect horizontal lines from binary array."""
    height, width = binary_array.shape
    horizontal_lines = []

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
                    if gap_count > config.max_line_gap:
                        if consecutive_ones >= config.min_points_in_line:
                            horizontal_lines.append((line_start, (x - gap_count - 1, y)))
                        line_start = None
                        consecutive_ones = 0
                    gap_count += 1

        if line_start is not None and consecutive_ones >= config.min_points_in_line:
            horizontal_lines.append((line_start, (width - 1, y)))

    return [line for line in horizontal_lines
            if abs(line[1][0] - line[0][0]) >= config.min_line_length]


def detect_vertical_lines(binary_array: np.ndarray,
                          config: LineDetectionConfig) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Detect vertical lines from binary array."""
    height, width = binary_array.shape
    vertical_lines = []

    for x in range(width):
        line_start = None
        consecutive_ones = 0
        gap_count = 0

        for y in range(height):
            if binary_array[y, x] == 1:
                if line_start is None:
                    line_start = (x, y)
                consecutive_ones += 1
                gap_count = 0
            else:
                if line_start is not None:
                    if gap_count > config.max_line_gap:
                        if consecutive_ones >= config.min_points_in_line:
                            vertical_lines.append((line_start, (x, y - gap_count - 1)))
                        line_start = None
                        consecutive_ones = 0
                    gap_count += 1

        if line_start is not None and consecutive_ones >= config.min_points_in_line:
            vertical_lines.append((line_start, (x, height - 1)))

    return [line for line in vertical_lines
            if abs(line[1][1] - line[0][1]) >= config.min_line_length]


def find_intersections(horizontal_lines: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                       vertical_lines: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                       tolerance: int = 5) -> List[Tuple[int, int]]:
    """Find intersection points between horizontal and vertical lines."""
    intersections = []

    for h_line in horizontal_lines:
        h_y = h_line[0][1]  # y-coordinate is same for horizontal line
        h_x_start, h_x_end = min(h_line[0][0], h_line[1][0]), max(h_line[0][0], h_line[1][0])

        for v_line in vertical_lines:
            v_x = v_line[0][0]  # x-coordinate is same for vertical line
            v_y_start, v_y_end = min(v_line[0][1], v_line[1][1]), max(v_line[0][1], v_line[1][1])

            # Check if lines intersect
            if (h_x_start - tolerance <= v_x <= h_x_end + tolerance and
                v_y_start - tolerance <= h_y <= v_y_end + tolerance):
                intersections.append((v_x, h_y))

    # Remove duplicate points within tolerance
    filtered_intersections = []
    for point in intersections:
        if not any(abs(p[0] - point[0]) < tolerance and abs(p[1] - point[1]) < tolerance
                   for p in filtered_intersections):
            filtered_intersections.append(point)

    return filtered_intersections


def draw_results(image_path: str,
                 horizontal_lines: List[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
                 vertical_lines: List[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
                 intersections: List[Tuple[int, int]] = None,
                 draw_config: DrawConfig = DrawConfig(),
                 output_path: str = 'output.png') -> Image.Image:
    """Draw detected lines and intersections on the image."""
    # Read original image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image from provided path")

    # Convert to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # Draw horizontal lines
    if horizontal_lines:
        for line in horizontal_lines:
            draw.line([line[0], line[1]],
                      fill=draw_config.line_color,
                      width=draw_config.line_width)

    # Draw vertical lines
    if vertical_lines:
        for line in vertical_lines:
            draw.line([line[0], line[1]],
                      fill=draw_config.line_color,
                      width=draw_config.line_width)

    # Draw intersection points
    if intersections:
        for point in intersections:
            x, y = point
            draw.ellipse([x - draw_config.point_radius, y - draw_config.point_radius,
                          x + draw_config.point_radius, y + draw_config.point_radius],
                         fill=draw_config.point_color)

    # Save the result
    pil_image.save(output_path)
    return pil_image


# Usage example
if __name__ == '__main__':
    # Initialize configurations
    line_config = LineDetectionConfig(
        min_line_length=50,
        max_line_gap=2,
        min_points_in_line=40
    )
    draw_config = DrawConfig(
        line_color='red',
        line_width=3,
        point_color='blue',
        point_radius=5
    )

    # Process image
    processor = ImageProcessor()
    image_path = "/home/ntlpt59/MAIN/experiments/checkbox_optimized/data/Export-Bill_filled_sample0.jpg"
    binary_arr = processor.get_binary_image(image_path=image_path)

    # Detect lines
    horizontal_lines = detect_horizontal_lines(binary_arr, line_config)
    vertical_lines = detect_vertical_lines(binary_arr, line_config)

    # Find intersections
    intersection_points = find_intersections(horizontal_lines, vertical_lines)

    # Draw results
    # 1. Draw lines only
    draw_results(image_path,
                 horizontal_lines=horizontal_lines,
                 vertical_lines=vertical_lines,
                 output_path='lines_output.png',
                 draw_config=draw_config)

    # 2. Draw intersections only
    draw_results(image_path,
                 intersections=intersection_points,
                 output_path='intersection_points.png',
                 draw_config=draw_config)

    # 3. Draw both lines and intersections
    draw_results(image_path,
                 horizontal_lines=horizontal_lines,
                 vertical_lines=vertical_lines,
                 intersections=intersection_points,
                 output_path='complete_output.png',
                 draw_config=draw_config)

    # Print statistics
    print(f"Found {len(horizontal_lines)} horizontal lines")
    print(f"Found {len(vertical_lines)} vertical lines")
    print(f"Found {len(intersection_points)} intersection points")
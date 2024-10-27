import argparse
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
import cv2
import numpy as np
import logging
from pathlib import Path

from preprocessor import ImagePreprocessor
from config import Config


@dataclass
class PatternMatchResult:
    """Data class to store pattern matching results."""
    position: Tuple[int, int]
    neighborhood: np.ndarray =None
    confidence: float =None

@dataclass
class PreprocessConfig:
    erosion_kernel: Tuple[int, int] = (2,2)
    dilation_kernel: Tuple[int, int] = (2,2)
    binary_threshold: int = 200


def find_pattern_3x3_fast(arr, pattern=None):
    """
    Efficiently find all occurrences of a 3x3 pattern in a numpy array.

    Parameters:
    arr : numpy.ndarray
        The input array to search in
    pattern : numpy.ndarray, optional
        The 3x3 pattern to search for

    Returns:
    list of tuples
        List of (row, col) coordinates where the pattern starts
    """
    if arr.ndim != 2:
        raise ValueError("Input array must be 2-dimensional")

    if pattern is None:
        # Default pattern - you can modify this
        pattern = np.array([
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 0]
        ])

    if pattern.shape != (3, 3):
        raise ValueError("Pattern must be 3x3")

    # Create a view of all possible 3x3 windows without copying data
    windows = np.lib.stride_tricks.sliding_window_view(arr, (3, 3))

    # Find where the windows match the pattern
    matches = np.all(windows == pattern.reshape(1, 1, 3, 3), axis=(2, 3))

    # Get the coordinates of matches
    match_coords = np.where(matches)

    return list(zip(match_coords[0], match_coords[1]))


class ImagePatternMatcher:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._image: Optional[np.ndarray] = None
        self._binary: Optional[np.ndarray] = None
        self._pattern: Optional[np.ndarray] = None

    @property
    def default_pattern(self) -> np.ndarray:
        """Default pattern to search for if none is provided."""
        return np.array([[0, 0, 0],
                         [0, 1, 1],
                         [0, 1, 0]], dtype=np.uint8)

    def load_image(self, image_source: Union[str, Path, np.ndarray],preprocess_config:PreprocessConfig) -> bool:
        try:
            self._preprocess = ImagePreprocessor(**preprocess_config.__dict__)
            self._image,self._gray_image = self._preprocess.preprocess(image_path=image_source)
            cv2.imwrite("binary_image.png",self._gray_image)
            self._binary= np.where(self._gray_image==255,1,0)
            self._inv_binary= np.where(self._binary==1,0,1)
            self.logger.info(f"Image loaded and preprocessed. Shape: {self._image.shape}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading image: {str(e)}")
            raise ValueError(f"Image loading failed: {str(e)}")

    def _compute_cumsum_arrays(self,arr) -> Tuple[np.ndarray, np.ndarray]:
        """Compute cumulative sum arrays for the binary image."""
        return (np.cumsum(arr, axis=0),
                np.cumsum(arr, axis=1))

    def set_pattern(self, pattern: Optional[np.ndarray] = None) -> None:
        if pattern is None:
            self._pattern = self.default_pattern
        else:
            if not isinstance(pattern, np.ndarray):
                raise ValueError("Pattern must be a numpy array")
            if pattern.dtype != np.uint8:
                pattern = pattern.astype(np.uint8)
            self._pattern = pattern

        self.logger.debug(f"Pattern set with shape: {self._pattern.shape}")

    def _validate_state(self) -> None:
        if self._binary is None:
            raise ValueError("No image loaded. Call load_image first.")
        if self._pattern is None:
            self.set_pattern()

    def _get_neighborhood(self, row: int, col: int) -> np.ndarray:
        """Extract neighborhood around a position."""
        half_height = self._pattern.shape[0] // 2
        half_width = self._pattern.shape[1] // 2

        # Calculate bounds
        row_start = max(0, row - half_height)
        row_end = min(self._binary.shape[0], row + half_height + 1)
        col_start = max(0, col - half_width)
        col_end = min(self._binary.shape[1], col + half_width + 1)

        # Extract neighborhood
        neighborhood = self._binary[row_start:row_end, col_start:col_end]

        # Ensure neighborhood has same shape as pattern
        if neighborhood.shape != self._pattern.shape:
            # Pad if necessary
            pad_height = self._pattern.shape[0] - neighborhood.shape[0]
            pad_width = self._pattern.shape[1] - neighborhood.shape[1]

            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left

            neighborhood = np.pad(
                neighborhood,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant',
                constant_values=0
            )

        return neighborhood

    @staticmethod
    def print_np(mat):
        for x in range(mat.shape[0]):
            for y in range(mat.shape[1]):
                print('1' if mat[x, y] == 1 else '0', end="")
            print()

    def find_matches(self, confidence_threshold: float = 0.8) -> List[PatternMatchResult]:
        """
        Find all pattern matches in the image.

        Args:
            confidence_threshold: Minimum matching confidence (0.8 default for more lenient matching)
        """
        self._validate_state()
        y_cumsum, x_cumsum = self._compute_cumsum_arrays(self._inv_binary)
        self._pad_inv_binary = np.pad(self._inv_binary, (1, ), 'constant', constant_values=(0,))
        matches: List = []
        print(self._pad_inv_binary.shape)
        # self._pad_inv_binary = np.random.randint(10,size=(5,4))
        rows, cols = self._pad_inv_binary.shape
        p_height, p_width = self._pattern.shape
        pattern = np.array([
            [1,1,1],
            [1, 1, 1],
            [1,1,0]
        ])
        matches = find_pattern_3x3_fast(self._inv_binary,pattern)


        print(len(matches))
        with open("matches.json","w") as f:
            json.dump([(int(x[0]),int(x[1])) for x in matches], f,indent=4)
        location_idx = 0
        if matches:
            print("\nVerifying first match:")
            row, col = matches[location_idx]
            print("Pattern found at position:", (row, col))
            print("3x3 window at this position:")
            print(self.print_np(self._inv_binary[row:row + 20, col:col + 20]))

        from PIL import Image, ImageDraw
        point = matches[location_idx]
        img = Image.open("/home/ntlpt59/MAIN/experiments/checkbox_optimized/data/Export-Bill_filled_sample0.jpg")
        draw = ImageDraw.Draw(img)
        Image.fromarray(self._image)
        y,x=point
        with open("matches.json","r") as f:
            points = json.load(f)
        for y,x in points:
            draw.rectangle([(x,y),(x+20,y+20)], outline='blue', width=2)
        img.show()

        exit()

        self.logger.info(f"Found {len(matches)} matches with confidence >= {confidence_threshold}")
        return matches

    def visualize_matches(self, matches: List[PatternMatchResult]) -> np.ndarray:
        """Visualize matches on the original image."""
        if self._image is None:
            raise ValueError("No image loaded")

        # Convert to BGR if grayscale
        vis_image = cv2.cvtColor(self._image, cv2.COLOR_GRAY2BGR)

        # Draw matches
        for match in matches:
            row, col = match.position
            half_height = self._pattern.shape[0] // 2
            half_width = self._pattern.shape[1] // 2

            # Calculate confidence-based color
            confidence_color = int(255 * match.confidence)
            color = (0, confidence_color, 255 - confidence_color)

            # Draw rectangle centered at the match position
            top_left = (col - half_width, row - half_height)
            bottom_right = (col + half_width, row + half_height)

            cv2.rectangle(vis_image, top_left, bottom_right, color, 2)

            # Add confidence text
            cv2.putText(
                vis_image,
                f"{match.confidence:.2f}",
                (col - half_width, row - half_height - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

        return vis_image


def main(image_path: str, threshold: int = 125, confidence: float = 0.8):
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Initialize matcher
    matcher = ImagePatternMatcher()
    preprocess_config = PreprocessConfig()

    try:
        # Load image and find matches
        matcher.load_image(image_path,preprocess_config=preprocess_config)

        # Set custom pattern for checkbox detection
        # This pattern represents a general checkbox shape
        # Find matches
        matches = matcher.find_matches(confidence_threshold=confidence)

        # Print results
        print(f"\nFound {len(matches)} potential checkboxes:")
        for i, match in enumerate(matches):
            print(f"\nCheckbox {i + 1}:")
            print(f"Position: {match.position}")
            print(f"Confidence: {match.confidence:.2f}")
            print("Pattern:")
            print(match.neighborhood)


        # Visualize results
        result_image = matcher.visualize_matches(matches)

        # Resize image if too large
        max_display_height = 800
        if result_image.shape[0] > max_display_height:
            scale = max_display_height / result_image.shape[0]
            new_width = int(result_image.shape[1] * scale)
            result_image = cv2.resize(result_image, (new_width, max_display_height))

        cv2.imshow('Checkbox Matches', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        logging.error(f"Error in pattern matching: {str(e)}")
        raise


if __name__ == "__main__":
    main("/home/ntlpt59/MAIN/experiments/checkbox_optimized/data/Export-Bill_filled_sample0.jpg")
    exit()
    parser = argparse.ArgumentParser(description="Checkbox detection in images")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--threshold", "-t", type=int, default=125,
                        help="Binary threshold value (default: 125)")
    parser.add_argument("--confidence", "-c", type=float, default=0.8,
                        help="Confidence threshold for matching (default: 0.8)")

    args = parser.parse_args()
    main(args.image_path, args.threshold, args.confidence)
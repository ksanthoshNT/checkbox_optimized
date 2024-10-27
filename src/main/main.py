import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
import cv2
import numpy as np
import logging
from pathlib import Path


@dataclass
class PatternMatchResult:
    """Data class to store pattern matching results."""
    position: Tuple[int, int]
    neighborhood: np.ndarray
    confidence: float


class ImagePatternMatcher:
    """
    A class to handle binary pattern matching in images following SOLID principles.

    Single Responsibility: Handles pattern matching in binary images
    Open/Closed: Extensible for different patterns and matching strategies
    Liskov Substitution: Maintains consistent behavior for all image types
    Interface Segregation: Clear, focused methods
    Dependency Inversion: Depends on abstractions (numpy arrays) not concrete implementations
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the pattern matcher.

        Args:
            logger: Optional logger instance for debugging and monitoring
        """
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

    def load_image(self, image_source: Union[str, Path, np.ndarray], threshold: int = 125) -> bool:
        """
        Load and preprocess the image.

        Args:
            image_source: Path to image or numpy array
            threshold: Binary threshold value

        Returns:
            bool: True if loading successful

        Raises:
            ValueError: If image loading fails
        """
        try:
            if isinstance(image_source, (str, Path)):
                self._image = cv2.imread(str(image_source), cv2.IMREAD_GRAYSCALE)
                if self._image is None:
                    raise ValueError(f"Failed to load image from {image_source}")
            elif isinstance(image_source, np.ndarray):
                self._image = image_source.copy()
                if len(self._image.shape) > 2:
                    self._image = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
            else:
                raise ValueError("Unsupported image source type")

            _, self._binary = cv2.threshold(self._image, threshold, 1, cv2.THRESH_BINARY)
            self.logger.info(f"Image loaded and preprocessed. Shape: {self._image.shape}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading image: {str(e)}")
            raise ValueError(f"Image loading failed: {str(e)}")

    def set_pattern(self, pattern: Optional[np.ndarray] = None) -> None:
        """
        Set the pattern to search for.

        Args:
            pattern: Binary numpy array pattern. If None, uses default pattern.

        Raises:
            ValueError: If pattern is invalid
        """
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
        """Validate object state before pattern matching."""
        if self._binary is None:
            raise ValueError("No image loaded. Call load_image first.")
        if self._pattern is None:
            self.set_pattern()

    def _compute_cumsum_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute cumulative sum arrays for the binary image."""
        return (np.cumsum(self._binary, axis=0),
                np.cumsum(self._binary, axis=1))

    def _get_neighborhood(self, row: int, col: int, size: int = 3) -> np.ndarray:
        """Extract neighborhood around a position."""
        half = size // 2
        return self._binary[row - half:row + half + 1, col - half:col + half + 1]

    def find_matches(self, confidence_threshold: float = 1.0) -> List[PatternMatchResult]:
        """
        Find all pattern matches in the image.

        Args:
            confidence_threshold: Minimum matching confidence (1.0 for exact matches)

        Returns:
            List of PatternMatchResult objects

        Raises:
            ValueError: If validation fails
        """
        self._validate_state()

        matches = []
        rows, cols = self._binary.shape
        p_height, p_width = self._pattern.shape

        y_cumsum, x_cumsum = self._compute_cumsum_arrays()

        for i in range(p_height - 1, rows):
            for j in range(p_width - 1, cols):
                window = self._get_neighborhood(i, j)

                # Calculate match confidence
                confidence = np.sum(window == self._pattern) / self._pattern.size

                if confidence >= confidence_threshold:
                    match = PatternMatchResult(
                        position=(i - p_height // 2, j - p_width // 2),
                        neighborhood=window.copy(),
                        confidence=confidence
                    )
                    matches.append(match)

        self.logger.info(f"Found {len(matches)} matches with confidence >= {confidence_threshold}")
        return matches

    def visualize_matches(self, matches: List[PatternMatchResult]) -> np.ndarray:
        """
        Visualize matches on the original image.

        Args:
            matches: List of PatternMatchResult objects

        Returns:
            numpy.ndarray: Image with visualized matches

        Raises:
            ValueError: If no image is loaded
        """
        if self._image is None:
            raise ValueError("No image loaded")

        # Convert to BGR if grayscale
        vis_image = cv2.cvtColor(self._image, cv2.COLOR_GRAY2BGR)

        for match in matches:
            row, col = match.position
            confidence_color = int(255 * match.confidence)
            color = (0, confidence_color, 255 - confidence_color)

            # Draw rectangle around match
            cv2.rectangle(
                vis_image,
                (col, row),
                (col + self._pattern.shape[1], row + self._pattern.shape[0]),
                color,
                1
            )

        return vis_image


# Example usage
def example_usage(image_path):
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Initialize matcher
    matcher = ImagePatternMatcher()

    try:
        # Load image and find matches
        matcher.load_image(image_path, threshold=125)

        # Optional: Set custom pattern
        # custom_pattern = np.array([[0,1,0], [1,1,1], [0,1,0]], dtype=np.uint8)
        # matcher.set_pattern(custom_pattern)

        # Find matches with 90% confidence
        matches = matcher.find_matches(confidence_threshold=0.9)

        # Print results
        for i, match in enumerate(matches):
            print(f"Match {i + 1}:")
            print(f"Position: {match.position}")
            print(f"Confidence: {match.confidence:.2f}")
            print(f"Neighborhood:\n{match.neighborhood}\n")

        # Visualize results
        result_image = matcher.visualize_matches(matches)
        cv2.imshow('Matches', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        logging.error(f"Error in pattern matching: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pattern matching in images")
    parser.add_argument("image_path", default = "Export-Bill_filled_sample0.jpg",help="Path to the input image")
    args = parser.parse_args()
    example_usage(args.image_path)
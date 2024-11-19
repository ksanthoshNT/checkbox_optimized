import argparse
import json
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
import cv2
import numpy as np
import logging
from pathlib import Path

from tqdm import tqdm

from src.main_v2.preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


@dataclass
class PatternMatchResult:
    """Data class to store pattern matching results."""
    position: Tuple[int, int]
    neighborhood: np.ndarray = None
    confidence: float = None


@dataclass
class PreprocessConfig:
    erosion_kernel: Tuple[int, int] = (2, 2)
    dilation_kernel: Tuple[int, int] = (2, 2)
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


    def load_image(self, image_source: Union[str, Path, np.ndarray], preprocess_config: PreprocessConfig) -> bool:
        try:
            self._preprocess = ImagePreprocessor(**preprocess_config.__dict__)
            self._image, self._gray_image = self._preprocess.preprocess(image_path=image_source)
            cv2.imwrite("gray_image.png", self._gray_image)
            self._binary = np.where(self._gray_image == 255, 1, 0)
            self._inv_binary = np.where(self._binary == 1, 0, 1)
            self.logger.info(f"Image loaded and preprocessed. Shape: {self._image.shape}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading image: {str(e)}")
            raise ValueError(f"Image loading failed: {str(e)}")

    def compute_cumsum_arrays(self, arr):
        """Compute cumulative sum arrays for the binary image."""
        self.image_arr_x_cumsum = np.cumsum(arr, axis=1)
        self.image_arr_y_cumsum = np.cumsum(arr, axis=0)


    def is_box(self,x1:int,y1:int,x2:int,y2:int):
        return (self.get_iloc(self.image_arr_x_cumsum,x2,y1)-(self.get_iloc(self.image_arr_x_cumsum,x1-1,y1) if x1-1>=0 else 0) == self.get_iloc(self.image_arr_x_cumsum,x2,y2)-(self.get_iloc(self.image_arr_x_cumsum,x1-1,y2) if x1-1>=0 else 0)) and (
            self.get_iloc(self.image_arr_y_cumsum,x1,y2)-(self.get_iloc(self.image_arr_y_cumsum,x2,y1-1) if y1-1>=0 else 0) == self.get_iloc(self.image_arr_y_cumsum, x2, y2) -(self.get_iloc(self.image_arr_y_cumsum, x2, y1-1) if y1-1>=0 else 0)
        )

    def get_iloc(self,arr,x1,y1):
        return arr[y1][x1]

    @staticmethod
    def print_np(mat):
        for x in range(mat.shape[0]):
            for y in range(mat.shape[1]):
                print('1' if mat[x, y] == 1 else '0', end="")
            print()


def get_indices_of_value(input_array: np.ndarray,val) -> np.ndarray:
    """
    Returns a numpy array with indices of the input array that are equal to 1.

    Parameters:
    input_array (np.ndarray): Input numpy array to find indices of 1.

    Returns:
    np.ndarray: Indices where input_array equals 1.
    """
    # Use np.where to get indices where input_array equals 1
    indices = np.where(input_array == val)
    return indices

def get_rotate_and_imposed_indices(arr:np.ndarray):
    new_rot_array = np.rot90(arr, k=2)
    both_ones = (arr == 1) & (new_rot_array == 1)
    indices = np.where(both_ones)
    return indices


# def list_based_boxes():
#     boxes = []
#     vals = np.sort(np_concatenated).tolist()
#     for idx, x in enumerate(tqdm(vals, desc="X..")):
#         for y in vals[idx:]:
#             if matcher.is_box(x[0], x[1], y[0], y[1]):
#                 boxes.append([x[0], x[1], y[0], y[1]])

def get_merged_indices_from_image(image_arr=None):
    indices = get_rotate_and_imposed_indices(image_arr)

    print(indices[0].shape)  # shape (32494,)
    print(indices[1].shape)  # shape (32494,)

    np_concatenated = np.concatenate((indices[0].reshape(-1, 1), indices[1].reshape(-1, 1)), axis=1)
    return np_concatenated


def main(image_path: str, threshold: int = 125, confidence: float = 0.8):
    start_time = time.perf_counter()

    matcher = ImagePatternMatcher()
    preprocess_config = PreprocessConfig()

    matcher.load_image(image_path, preprocess_config=preprocess_config)

    image_arr = matcher._inv_binary.copy()
    logger.debug(f"Image shape : {image_arr.shape}")

    matcher.compute_cumsum_arrays(image_arr.copy())

    updated_image_indices = get_merged_indices_from_image(image_arr=image_arr)
    #shape of updated_image_indices: (32494,2)

    # boxes = []
    # vals = np.sort(updated_image_indices).tolist()
    # for idx,x in enumerate(tqdm(vals,desc="X..")):
    #     for y in vals[idx:]:
    #         if matcher.is_box(x[0],x[1],y[0],y[1]):
    #             boxes.append([x[0],x[1],y[0],y[1]])
    #
    # print(len(boxes))

    end_time = time.perf_counter()
    print(f"Duration: {end_time - start_time} seconds")




if __name__ == "__main__":
    main("/home/ntlpt59/MAIN/experiments/checkbox_optimized/data/Export-Bill_filled_sample0.jpg")


####tests
# def test_is_box():
#     ar = np.array([[1,1,1,1],
#                    [1,0,0,1],
#                    [1,0,0,1],
#                   [1,1,1,1]])
#     matcher = ImagePatternMatcher()
#     matcher.compute_cumsum_arrays(ar)
#     print(matcher.image_arr_x_cumsum)
#     print('=========')
#     print(matcher.image_arr_y_cumsum)
#     print(matcher.is_box(0,0,3,3))




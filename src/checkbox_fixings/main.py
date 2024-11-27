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


class ImageProcessor:
    def __init__(self, logger: Optional[logging.Logger] = None,
                 preprocess_config:PreprocessConfig=None):
        self.logger = logger or logging.getLogger(__name__)
        self._image: Optional[np.ndarray] = None
        self._binary: Optional[np.ndarray] = None
        self._pattern: Optional[np.ndarray] = None
        self.preprocess_config = preprocess_config or PreprocessConfig()


    def get_binary_image(self, image_path: Union[str, Path, np.ndarray]) -> np.ndarray:
        preprocess = ImagePreprocessor(**self.preprocess_config.__dict__)
        self._image, self._gray_image = preprocess.preprocess(image_path=image_path)
        self._binary = np.where(self._gray_image == 255, 1, 0)
        _inv_binary = np.where(self._binary == 1, 0, 1)

        self.logger.info(f"Image loaded and preprocessed. Shape: {self._image.shape}")
        return _inv_binary.copy()

    @staticmethod
    def print_np(mat):
        for x in range(mat.shape[0]):
            for y in range(mat.shape[1]):
                print('1' if mat[x, y] == 1 else '0', end="")
            print()


class CumulativeSumBoxDetector:
    def __init__(self,input_array: np.ndarray):
        self.image_arr = input_array
        self.image_arr_x_cumsum, self.image_arr_y_cumsum = self.compute_cumsum_arrays(self.image_arr)


    def compute_cumsum_arrays(self, arr):
        """Compute cumulative sum arrays for the binary image."""
        image_arr_x_cumsum = np.cumsum(arr, axis=1)
        image_arr_y_cumsum = np.cumsum(arr, axis=0)
        return image_arr_x_cumsum,image_arr_y_cumsum


    def is_box(self,x1:int,y1:int,x2:int,y2:int):
        return (self.iloc(self.image_arr_x_cumsum,x2,y1)-(self.iloc(self.image_arr_x_cumsum,x1-1,y1) if x1-1>=0 else 0) == self.iloc(self.image_arr_x_cumsum,x2,y2)-(self.iloc(self.image_arr_x_cumsum,x1-1,y2) if x1-1>=0 else 0)) and (
            self.iloc(self.image_arr_y_cumsum,x1,y2)-(self.iloc(self.image_arr_y_cumsum,x2,y1-1) if y1-1>=0 else 0) == self.iloc(self.image_arr_y_cumsum, x2, y2) -(self.iloc(self.image_arr_y_cumsum, x2, y1-1) if y1-1>=0 else 0)
        )

    def iloc(self,arr,x1,y1):
        return arr[y1][x1]


if __name__ == '__main__':
    processor = ImageProcessor()
    input_arr = processor.get_binary_image(image_path="/home/ntlpt59/MAIN/experiments/checkbox_optimized/data/Export-Bill_filled_sample0.jpg")
    box_processor = CumulativeSumBoxDetector(input_arr)
    print(box_processor.iloc(box_processor.image_arr_x_cumsum,
                                 100,100))
    processor.print_np(input_arr[:10,:10])

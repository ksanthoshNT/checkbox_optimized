class Config:
    EROSION_KERNEL = (2, 2)
    DILATION_KERNEL = (2, 2)
    BINARY_THRESHOLD = 200
    BOX_MAX_WIDTH = 50
    BOX_MAX_HEIGHT = 50
    MIN_PERIMETER = 40
    MAX_LENGTH_TO_WIDTH_RATIO = 2
    MAX_WIDTH_TO_LENGTH_RATIO = 1.150
    MAX_DARK_TO_BRIGHT_RATIO = 2
    MIN_DIMENSION = 3
    MIN_NUM_BRIGHT_PIXELS = 1
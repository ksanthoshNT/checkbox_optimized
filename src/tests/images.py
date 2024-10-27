import numpy as np


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


def generate_binary_image_3x3(height=255, width=256, pattern_density=0.05):
    """
    Generate a binary image with 3x3 patterns.

    Parameters:
    -----------
    height : int
        Height of the image
    width : int
        Width of the image
    pattern_density : float
        Probability of creating patterns (0-1)

    Returns:
    --------
    numpy.ndarray
        Binary image array
    """
    # Initialize with random binary values
    img = np.random.randint(0, 2, size=(height, width), dtype=np.uint8)

    # Example 3x3 pattern
    pattern = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])

    # Add patterns at random locations
    num_patterns = int((height * width) * pattern_density)

    print(f"Num of Patterns: {num_patterns}")

    for _ in range(num_patterns):
        # Random position that ensures pattern fits
        h_pos = np.random.randint(0, height - 2)
        w_pos = np.random.randint(0, width - 2)
        img[h_pos:h_pos + 3, w_pos:w_pos + 3] = pattern

    return img


# Generate test image
test_image = generate_binary_image_3x3()

# Example pattern to search for
example_pattern = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])

# Find patterns
matches = find_pattern_3x3_fast(test_image, example_pattern)

# Print information
print(f"Image shape: {test_image.shape}")
print(f"Number of 1s: {np.sum(test_image)}")
print(f"Number of 0s: {test_image.size - np.sum(test_image)}")
print(f"\nNumber of patterns found: {len(matches)}")
print(f"\nSample 12x12 section from the middle:")
middle_h = test_image.shape[0] // 2
middle_w = test_image.shape[1] // 2
print(test_image[middle_h:middle_h + 12, middle_w:middle_w + 12])

# Verify a few matches
if matches:
    print("\nVerifying first match:")
    row, col = matches[0]
    print("Pattern found at position:", (row, col))
    print("3x3 window at this position:")
    print(test_image[row:row + 3, col:col + 3])
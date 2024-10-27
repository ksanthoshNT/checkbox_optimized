import numpy as np


def find_pattern_fast(arr, pattern=np.array([[0, 0], [0, 1]])):
    """
    Efficiently find all occurrences of a 2x2 pattern in a numpy array.

    Parameters:
    arr : numpy.ndarray
        The input array to search in
    pattern : numpy.ndarray, optional
        The 2x2 pattern to search for, defaults to [[0,0],[0,1]]

    Returns:
    list of tuples
        List of (row, col) coordinates where the pattern starts
    """
    if arr.ndim != 2:
        raise ValueError("Input array must be 2-dimensional")

    rows, cols = arr.shape
    pattern_rows, pattern_cols = pattern.shape

    if pattern_rows != 3 or pattern_cols != 3:
        raise ValueError("Pattern must be 2x2")

    # This is the fast method using stride tricks
    # It creates a view of all possible 2x2 windows without copying data
    windows = np.lib.stride_tricks.sliding_window_view(arr, (3, 3))

    # Find where the windows match the pattern
    # np.all(axis=(2,3)) checks if all elements match in each 2x2 window
    print(windows.shape)
    matches = np.all(windows == pattern.reshape(1, 1, 3,3), axis=(2, 3))

    # Get the coordinates of matches
    match_coords = np.where(matches)

    # Convert to list of tuples for easy iteration
    return list(zip(match_coords[0], match_coords[1]))


# Example usage and testing
if __name__ == "__main__":
    # Create a test array
    test_arr = np.array([
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0]
    ])

    # Find the pattern
    pattern = np.array([[0, 0], [0, 1]])
    matches = find_pattern_fast(test_arr)

    print("Test array:")
    print(test_arr)
    print("\nFound pattern at positions:")
    for row, col in matches:
        print(f"Position ({row}, {col}):")
        print(test_arr[row:row + 2, col:col + 2])

    # Performance test
    large_arr = np.random.randint(0, 2, size=(1000, 1000))

    import time

    start_time = time.time()
    matches = find_pattern_fast(large_arr)
    end_time = time.time()

    print(f"\nTime taken for 1000x1000 array: {end_time - start_time:.4f} seconds")
    print(f"Found {len(matches)} matches")
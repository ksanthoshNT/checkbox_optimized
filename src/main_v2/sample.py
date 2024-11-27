from practice import main


import numpy as np


def find_rectangles_memory_efficient(valid_dark_point_x_coords, valid_dark_point_y_coords, chunk_size=1000):
    """
    Memory-efficient version of rectangle finding algorithm

    Args:
        valid_dark_point_x_coords: x coordinates array
        valid_dark_point_y_coords: y coordinates array
        chunk_size: size of chunks to process at once
    """
    total_points = len(valid_dark_point_x_coords)
    print(f"Total points to process: {total_points}")

    # Initialize list to store potential rectangles
    potential_rects = []

    # Process in chunks to avoid memory overflow
    for start_idx in range(0, total_points, chunk_size):
        end_idx = min(start_idx + chunk_size, total_points)

        # Get current chunk of coordinates
        chunk_x = valid_dark_point_x_coords[start_idx:end_idx]
        chunk_y = valid_dark_point_y_coords[start_idx:end_idx]

        print(f"Processing chunk {start_idx // chunk_size + 1}, points {start_idx} to {end_idx}")

        # Create smaller meshgrid for current chunk
        X1, X2 = np.meshgrid(chunk_x, valid_dark_point_x_coords)
        Y1, Y2 = np.meshgrid(chunk_y, valid_dark_point_y_coords)

        # Apply validity mask
        valid_mask = (X2 > X1) & (Y2 > Y1)

        # Extract valid combinations
        if np.any(valid_mask):
            chunk_rects = np.column_stack((
                X1[valid_mask].flatten(),
                Y1[valid_mask].flatten(),
                X2[valid_mask].flatten(),
                Y2[valid_mask].flatten()
            ))

            # Optional: Add initial filtering here if needed
            # For example, filter out rectangles that are too small or too large
            min_size = 5  # Adjust as needed
            max_size = 100  # Adjust as needed
            size_filter = ((chunk_rects[:, 2] - chunk_rects[:, 0] >= min_size) &
                           (chunk_rects[:, 3] - chunk_rects[:, 1] >= min_size) &
                           (chunk_rects[:, 2] - chunk_rects[:, 0] <= max_size) &
                           (chunk_rects[:, 3] - chunk_rects[:, 1] <= max_size))

            chunk_rects = chunk_rects[size_filter]

            potential_rects.append(chunk_rects)

        # Print memory usage for monitoring
        if hasattr(np, 'dtype'):
            memory_usage = (X1.nbytes + X2.nbytes + Y1.nbytes + Y2.nbytes) / (1024 * 1024)
            print(f"Current chunk memory usage: {memory_usage:.2f} MB")

    # Combine all valid rectangles
    if potential_rects:
        all_rects = np.vstack(potential_rects)
        print(f"Total valid rectangles found: {len(all_rects)}")
        return all_rects
    else:
        print("No valid rectangles found")
        return np.array([])

def calculate_test_size(arr, ratio=4):
    unique_x_coords = np.unique(arr)
    total_unique_points = len(unique_x_coords)
    test_size = round(total_unique_points * (1 - 1 / ratio))
    return test_size

# Example usage:
if __name__ == "__main__":
    # Test with a smaller subset first
    from PIL import Image, ImageDraw
    image_path:str = "/home/ntlpt59/MAIN/experiments/checkbox_optimized/data/Export-Bill_filled_sample0.jpg"
    valid_dark_point_x_coords, valid_dark_point_y_coords, img = main(image_path=image_path)

    points = np.concatenate((valid_dark_point_x_coords.reshape(-1, 1), valid_dark_point_y_coords.reshape(-1, 1)), axis=-1)

    print(points.shape)

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    for idx,each_point in enumerate(points):
        x, y = each_point
        draw.rectangle([y, x, y+1, x + 1],
                       outline="blue",
                       width=2)

    image.save("marked_image.jpg")

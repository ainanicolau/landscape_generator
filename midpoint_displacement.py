import random
import numpy as np
from collections import deque


def run_midpoint_displacement(roughness, width, height):
    """
    Generates a 1D height map using the midpoint displacement algorithm.

    Args:
        roughness (float): The initial roughness of the terrain. Smaller values
            produce smoother terrain.
        width (int): The width of the terrain.
        height (int): The height of the terrain.

    Returns:
        A 1D list representing the heights of the terrain.
    """
    # Initialize the heights and points to process queue
    heights, points_to_process = __initialize_data(width, height, roughness)

    # Loop until all points have been processed
    while points_to_process:
        # Get the next segment to process
        previous, next, roughness = points_to_process.popleft()

        # Compute the midpoint of the segment
        midpoint_x = __find_horizontal_midpoint(previous, next)
        midpoint_y = __compute_midpoint_height(
            previous, next, heights, roughness
        )

        # Update the heights list with the new midpoint
        heights[midpoint_x] = midpoint_y

        # Add new segments to process if the current segment can be subdivided
        if next - previous > 2:
            # Decrease the roughness by half
            roughness /= 2
            points_to_process.append((previous, midpoint_x, roughness))
            points_to_process.append((midpoint_x, next, roughness))

    return heights


def __initialize_data(width, height, roughness):
    """
    Initializes the heights and points to process queue for the midpoint
    displacement algorithm.

    Args:
        width (int): The width of the terrain.
        height (int): The height of the terrain.
        roughness (float): The initial roughness of the terrain.

    Returns:
        A tuple containing the initialized heights and points to process queue.
    """
    # Initialize the heights list with the left and right endpoints set to 0
    heights = [0] * (width + 1)
    heights[0] = heights[width] = height // 2

    # Initialize the queue with the left and right endpoints
    points_to_process = deque([(0, width, roughness)])

    return heights, points_to_process


def __find_horizontal_midpoint(previous, next):
    """
    Computes the x-coordinate of the midpoint of a segment.

    Args:
        previous (int): The left endpoint of the segment.
        next (int): The right endpoint of the segment.

    Returns:
        The x-coordinate of the midpoint of the segment.
    """
    return (previous + next) // 2


def __compute_midpoint_height(point1, point2, heights, roughness):
    """
    Compute the midpoint height between two points and apply random
    displacement.

    Args:
        point1 (int): The index of the first point.
        point2 (int): The index of the second point.
        heights (list): The list of heights to compute from.
        roughness (int): The intensity of the roughness.

    Returns:
        int: The computed midpoint height.
    """
    # Compute the mean height of the two points
    mean_height = (heights[point1] + heights[point2]) // 2

    # Apply a random displacement based on the roughness parameter
    roughness_int = int(roughness)
    displacement = random.randint(-roughness_int, roughness_int)
    midpoint_height = mean_height + displacement

    return midpoint_height


def normalize(data, lower_bound, upper_bound):
    """
    Normalize data to the given bounds.

    Args:
        data (list[float]): The data to be normalized.
        lower_bound (float): The lower bound of the new range.
        upper_bound (float): The upper bound of the new range.

    Returns:
        list[float]: The normalized data.
    """
    # Calculate the min and max values of the data
    data_min = np.min(data)
    data_max = np.max(data)

    # Calculate the ranges
    data_range = data_max - data_min
    new_range = upper_bound - lower_bound

    # Normalize the data
    normalized_data = [
        (value - data_min) * new_range / data_range + lower_bound
        for value in data
    ]

    return normalized_data

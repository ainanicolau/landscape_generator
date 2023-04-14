import cv2
import numpy as np
import math

import midpoint_displacement as md


def generate_image(width, height, color):
    """
    Creates an image array with the given size and color.

    Args:
        width (int): The width of the image.
        height (int): The height of the image.
        color (Tuple): The RGBA color tuple to fill the image with.

    Returns:
        np.ndarray: An image of shape (height, width, 4)
    """
    image = np.zeros((height, width, 4), np.uint8)
    image[:, :] = color

    return image


def draw_sun(
    image, radius, center_x, center_y, color, white_contour, sky_element
):
    """
    Adds a sun or moon to the given image with a specific center and radius,
    with an option to draw the contours in white.

    Args:
        image (np.ndarray): The image to add the sun or moon to.
        radius (int): The radius of the sun or moon.
        center_x (int): The x-coordinate of the center of the sun or moon.
        center_y (int): The y-coordinate of the center of the sun or moon.
        color (tuple): The color of the sun or moon, as a tuple of four values
            (B, G, R, A), where B, G, and R are the blue, green, and red
            components of the color, and A is the alpha channel (transparency).
        white_contour (bool): If True, draw the contours of the sun or moon in
            white.
        sky_element (str): Either "Sun" or "Moon", to specify whether to draw
            a sun or moon.
    """
    # Set the color for the contour
    contour_color = (255, 255, 255, 255)

    # Draw the sun or moon
    center = (center_x, center_y)
    if radius > 0:
        if sky_element == "Sun":
            cv2.circle(
                image, center, radius, color, thickness=-1, lineType=8, shift=0
            )
            if white_contour:
                cv2.circle(
                    image,
                    center,
                    radius,
                    contour_color,
                    thickness=12,
                    lineType=8,
                    shift=0,
                )
        elif sky_element == "Moon":
            # Draw a moon instead of a sun
            inner_center = (
                center_x + int(radius / 3),
                center_y - int(radius / 3),
            )
            mask = np.zeros((image.shape[0], image.shape[1], 4), np.uint8)
            filling = mask.copy()
            filling[:, :] = color
            cv2.circle(
                mask, center, radius, (255, 255, 255, 255), thickness=-1
            )
            cv2.circle(
                mask,
                inner_center,
                math.floor(radius / 1.2),
                (0, 0, 0, 255),
                thickness=-1,
            )
            image[mask == 255] = filling[mask == 255]
            if white_contour:
                imgray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(imgray, 127, 255, 0)
                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(image, contours, -1, contour_color, 12)


def generate_mountains(
    image, num_layers, roughness, decrease_roughness, weight, height
):
    """
    Generates layers of mountain heights with a specific roughness using the
    midpoint displacement algorithm.

    Args:
        image (numpy.ndarray): The image on which to generate the mountains.
        num_layers (int): The number of mountain layers to generate.
        roughness (int): The roughness of the mountain terrain. A higher
            roughness value produces more jagged mountains.
        decrease_roughness (bool): If True, decreases the roughness of each
            successive layer by a factor of 1/(layer + 1).
        weight (float): The weight of the mountain terrain. A higher weight
            value produces taller mountains.
        height (int): The height of the image.

    Returns:
        List[numpy.ndarray]: A list of arrays representing the layers of
        mountain heights, ordered from the top layer to the bottom layer.
    """
    mountains = []
    for layer in range(num_layers):
        if not decrease_roughness:
            layer_roughness = roughness
        else:
            layer_roughness = roughness // (layer + 1)
        layer_heights = md.run_midpoint_displacement(
            layer_roughness, weight, height
        )
        mountains.append(layer_heights)

    return mountains


def normalize_mountains(
    mountains, height, lower_padding, upper_padding, mountain_intersection
):
    """
    Given a list of mountains, it normalizes the highest and lower heights and
    applies a padding to the range.

    Args:
        mountains (list): A list of mountain heights.
        height (int): The maximum height of the mountains.
        lower_padding (int): The padding added to the lower end of the range.
        upper_padding (int): The padding added to the upper end of the range.
        mountain_intersection (float): The percentage of intersection between
            the mountains.

    Returns:
        list: A list of normalized mountain heights.
    """
    num_layers = len(mountains)
    for layer in range(num_layers):
        layer_heights = mountains[layer]

        # Calculate the range of heights for the current layer
        layer_range = (
            (height - lower_padding - upper_padding) / num_layers * layer
            + upper_padding,
            height
            - lower_padding
            - (
                (height - lower_padding - upper_padding)
                / num_layers
                * (num_layers - layer - 1)
            )
            * mountain_intersection
            / 100,
        )

        # Normalize the heights for the current layer using the calculated
        # range
        normalized_layer = md.normalize(
            layer_heights, layer_range[0], layer_range[1]
        )

        mountains[layer] = normalized_layer

    return mountains


def draw_mountains(
    image,
    mountains,
    imageWidth,
    imageHeight,
    mountain_color,
    sky_color,
    white_contour,
):
    """
    Draw the mountains on the given image using the provided heights and color.

    Args:
        image (numpy.ndarray): The image on which to draw the mountains.
        mountains (List[List[float]]): A list of lists, where each inner list
            represents the height values for a single mountain layer.
        imageWidth (int): The width of the image in pixels.
        imageHeight (int): The height of the image in pixels.
        mountain_color (tuple): The color or list of colors to use for the
            mountain layers.
        sky_color (Tuple[int]): The color to use for the sky.
        white_contour (bool): Whether or not to draw a white contour around the
            mountains.
    """
    # Initialize the contour color to white
    contour_color = (255, 255, 255, 255)

    # Interpolate colors if a single mountain color is provided
    if len(mountain_color) == 1:
        colors = interpolate_colors(
            mountain_color[0], sky_color, len(mountains) + 1
        )

    # Draw each mountain layer as a filled polygon
    for layer in range(len(mountains)):
        # Convert the heights into the list of points of a polygon
        points = [[i, mountains[layer][i]] for i in range(imageWidth)]

        # Add the lower corners of the image to close the polygon
        points.insert(0, [0, imageHeight])
        points.append([imageWidth - 1, imageHeight])
        points = np.array(points, np.int32)
        points = points.reshape((-1, 1, 2))

        # Determine the layer color
        if len(mountain_color) > 1:
            layer_color = mountain_color[layer % len(mountain_color)]
        else:
            layer_color = colors[len(mountains) - layer - 1]

        # Draw the filled polygon
        cv2.fillPoly(image, [points], layer_color)

        # Draw the white contour if requested
        if white_contour:
            cv2.polylines(image, [points], True, contour_color, 12)


def interpolate_colors(start_color, end_color, num_divisions):
    """
    Given two colors, creates a list of interpolated colors.

    Args:
        start_color (tuple): An RGB tuple representing the starting color.
        end_color (tuple): An RGB tuple representing the ending color.
        num_divisions (int): The number of colors to create between the start
            and end colors.

    Returns:
        list of tuples: A list of `num_divisions` colors, interpolated between
            `start_color` and `end_color`.
    """
    color_channels = 3
    color_ranges = []
    for channel in range(color_channels):
        range_values = np.linspace(
            start_color[channel], end_color[channel], num_divisions
        )
        color_ranges.append([int(x) for x in range_values])
    color_ranges.append([255] * num_divisions)

    interpolated_colors = list(map(tuple, zip(*color_ranges)))

    return interpolated_colors


def smooth_mountains(mountains, smoothing_range):
    """
    Smoothes the mountain heights given a specific neighborhood range.

    Args:
        mountains (list[list[float]]): A list of lists containing the height
            values for each layer of the mountains.
        smoothing_range (int): The size of the neighborhood range to consider
            when smoothing the mountain heights.

    Returns:
        list[list[float]]: A list of lists containing the smoothed height
            values for each layer of the mountains.
    """
    smoothed_mountains = []

    for layer in mountains:
        smoothed_layer = []

        # Iterate through each height value in the layer
        for i in range(len(layer)):
            lower_bound = max(i - smoothing_range, 0)
            upper_bound = min(i + smoothing_range + 1, len(layer))
            neighborhood = layer[lower_bound:upper_bound]
            new_height = sum(neighborhood) / len(neighborhood)
            smoothed_layer.append(new_height)

        smoothed_mountains.append(smoothed_layer)

    return smoothed_mountains


def apply_texture(image, texture_path, alpha):
    """
    Given an image and a texture, it merges both using the texture as a mask.

    Args:
        image (np.array): The input image as a numpy array.
        texture_path (str): Path to the texture file.
        alpha (float): Alpha value for blending the image and texture.

    Returns:
        np.ndarray: The blended image.
    """
    out = np.ones(image.shape, np.uint8) * 255
    texture = cv2.imread(texture_path)
    texture = cv2.resize(texture, image.shape[1::-1])
    mask = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)
    mask = mask / 255

    for channel in range(3):
        mat = image[:, :, channel] * (1 - mask) + out[:, :, channel] * mask
        out[:, :, channel] = mat.astype(int)

    return out


def draw_margin(image, margin_type, width, height):
    """
    Draws a circular or rectangular white margin to the given image.

    Args:
        image (np.array): The input image as a numpy array.
        margin_type (str): The type of margin to draw - "Circle" or "Window".
        width (int): The width of the image.
        height (int): The height of the image.

    Returns:
        np.array: The image with the white margin added.
    """
    spacing_circle = 200
    spacing_window = 300

    # Create a blank mask and an output image of white pixels.
    mask = np.zeros((height, width, 4), np.uint8)
    out = np.ones((height, width, 4), np.uint8) * 255
    mask[:, :, 3] = 255

    # Draw the margin based on the margin_type.
    if margin_type == "Circle":
        # Calculate the center and radius of the circle.
        radius = math.floor(min(width, height) / 2) - spacing_circle
        center = (math.floor(width / 2), math.floor(height / 2))
        # Draw a filled circle on the mask.
        cv2.circle(mask, center, radius, (255, 255, 255, 255), thickness=-1)

    if margin_type == "Window":
        # Calculate the center and radius of the inner circle.
        radius = math.floor(min(width, height) / 2) - spacing_window
        center = (
            math.floor(width / 2),
            math.floor(height / 2) - spacing_window,
        )
        # Draw a filled circle on the mask.
        cv2.circle(mask, center, radius, (255, 255, 255, 255), thickness=-1)
        # Draw a filled rectangle on the mask.
        top_left = (center[0] - radius, center[1])
        bottom_right = (
            center[0] + radius,
            center[1] + math.floor(radius * 1.5),
        )
        cv2.rectangle(mask, top_left, bottom_right, (255, 255, 255, 255), -1)

    # Copy the input image onto the output image where the mask is white.
    out[mask == 255] = image[mask == 255]

    return out

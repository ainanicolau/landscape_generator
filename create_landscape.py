#!/usr/bin/env python3

import cv2
import numpy as np
import math

import midpoint_displacement as md


def generate_image(width, height, color):
    """
    Creates an image array with the given size and color.
    """
    image = np.zeros((height, width, 4), np.uint8)
    image[:, :] = color

    return image


def draw_sun(
    image, radius, center_x, center_y, color, white_contour, sky_element
):
    """
    Adds a sun to the given image with specific center and radius, with an
    option to draw the contours in white. The sun can be changes into a moon
    if specified.
    """
    contour_color = (255, 255, 255, 255)
    # contour_color = (0, 0, 0, 255)
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
        if sky_element == "Moon":
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
    Generated layers of mountain heights with a specific roughness using the
    midpoint dispacement algorithm.
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
    """
    num_layers = len(mountains)
    for layer in range(num_layers):
        layer_heights = mountains[layer]
        normalized_layer = md.normalize(
            layer_heights,
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

        mountains[layer] = normalized_layer


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
    Draw the different mountain layers by creating polygons with the heights
    and the bottom corners of the image. Has the possibility to add the white
    contours.
    """
    contour_color = (255, 255, 255, 255)
    # contour_color = (0, 0, 0, 255)
    if len(mountain_color) == 1:
        colors = interpolate_colors(
            mountain_color[0], sky_color, len(mountains) + 1
        )

    for layer in range(len(mountains)):
        # Convert the heights into the list of points of a polygon
        points = [[i, mountains[layer][i]] for i in range(0, imageWidth)]

        # Add the lower corners of the image to close the polygon
        points.insert(0, [0, imageHeight])
        points.append([imageWidth - 1, imageHeight])
        points = np.array(points, np.int32)
        points = points.reshape((-1, 1, 2))

        if len(mountain_color) > 1:
            layer_color = mountain_color[layer % len(mountain_color)]
        else:
            layer_color = colors[len(mountains) - layer - 1]
        cv2.fillPoly(image, [points], layer_color)

        if white_contour:
            cv2.polylines(image, [points], True, contour_color, 12)


def interpolate_colors(color1, color2, divisions):
    """
    Given two colors it creates a list of interpolated colors.
    """
    colors = []
    for channel in range(3):
        values = np.linspace(color1[channel], color2[channel], divisions)
        colors.append([int(x) for x in values])
    colors.append([255] * divisions)

    colors = list(map(tuple, zip(*colors)))

    return colors


def smooth_mountains(mountains, smooth_value):
    """
    Smoothes the mountain heighs given a specific neighbourhood range.
    """
    smoothed_mountains = []
    for layer in mountains:
        smoothed_layer = []
        for x in range(len(layer)):
            lower_bound = max(x - smooth_value, 0)
            upper_bound = min(x + smooth_value + 1, len(layer))
            neighbourhood = layer[lower_bound:upper_bound]
            new_height = sum(neighbourhood) / len(neighbourhood)
            smoothed_layer.append(new_height)
        smoothed_mountains.append(smoothed_layer)

    return smoothed_mountains


def apply_texture(image, texture, alpha):
    """
    Given an image and a texture, it merges both using the texture as a mask.
    """
    out = np.ones(image.shape, np.uint8) * 255
    texture = cv2.imread(texture)
    texture = cv2.resize(texture, image.shape[1::-1])
    mask = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)
    mask = mask / 255

    for channel in range(3):
        mat = image[:, :, channel] * (1 - mask) + out[:, :, channel] * mask
        out[:, :, channel] = mat.astype(int)

    return out


def draw_margin(image, margin, width, height):
    """
    Draws a circular or window shape white margin to the given image.
    """
    spacing_circle = 200
    spacing_window = 300
    mask = np.zeros((height, width, 4), np.uint8)
    out = np.ones((height, width, 4), np.uint8) * 255
    mask[:, :, 3] = 255
    if margin == "Circle":
        radius = math.floor(min(width, height) / 2) - spacing_circle
        center = (math.floor(width / 2), math.floor(height / 2))
        cv2.circle(mask, center, radius, (255, 255, 255, 255), thickness=-1)

    if margin == "Window":
        radius = math.floor(min(width, height) / 2) - spacing_window
        center = (
            math.floor(width / 2),
            math.floor(height / 2) - spacing_window,
        )
        cv2.circle(mask, center, radius, (255, 255, 255, 255), thickness=-1)
        top_left = (center[0] - radius, center[1])
        bottom_right = (
            center[0] + radius,
            center[1] + math.floor(radius * 1.5),
        )
        cv2.rectangle(mask, top_left, bottom_right, (255, 255, 255, 255), -1)

    out[mask == 255] = image[mask == 255]

    return out

#!/usr/bin/env python3

import argparse
import random
import numpy as np
from collections import deque
from PIL import Image, ImageDraw


# Image Resolution
WIDTH = 513
HEIGHT = 720
IMAGE_PADDING = 50


def parseArguments():
    """
    Parses command line options.
    """
    parser = argparse.ArgumentParser(description="Implementation of midpoint "
                                                 "displacement algorithm")
    parser.add_argument('--layers', '-l', type=int, default=3,
                        help="number of layers (default: 3)")
    parser.add_argument('--roughness', '-r', type=int, default=200,
                        help="roughness factor (default: 100)")
    args = parser.parse_args()

    # Left and right points
    return args


def run_midpoint_displacement(roughness, width, height):
    """
    Runs the midpoint displacement algorithm to compute the layer heights.
    """
    heights, points_to_process = initilize_data(width, height, roughness)

    while len(points_to_process) > 0:
        previous, next, roughness = points_to_process.popleft()
        midpoint_x = find_horizontal_midpoint(previous, next)
        midpoint_y = compute_midpoint_height(previous, next, heights,
                                             roughness)

        # Update heights with the new point
        heights[midpoint_x] = midpoint_y

        # Add new points to process if the segment can be subdivided
        if next - previous > 2:
            # Descrease roughness intensity by half
            roughness = roughness // 2
            points_to_process.append((previous, midpoint_x, roughness))
            points_to_process.append((midpoint_x, next, roughness))

    return heights


def initilize_data(width, height, roughness):
    """
    Defines first and last column heights and create a deque with the necessary
    information to start the midpoint displacement algorithm.
    """
    # Create list of heights with values to 0
    heights = [0] * width

    # Initialize first and last column heights
    # if not args.
    heights[0] = random.randint(0, height)
    heights[-1] = random.randint(0, height)

    points_to_process = deque()
    points_to_process.append((0, width - 1, roughness))

    return heights, points_to_process


def find_horizontal_midpoint(point_left, point_right):
    """
    Finds the horitzontal midpoint between two points.
    """
    midpoint = (point_left + point_right + 1) // 2

    return midpoint


def compute_midpoint_height(point1, point2, heights, roughness):
    """
    Computes mean height between two points and applies random displacement.
    """
    mean_height = (heights[point1] + heights[point2]) // 2
    displacement = random.randint(-roughness, roughness)
    midpoint_height = mean_height + displacement

    return midpoint_height


def normalize(data, lowerBound, upperBound):
    """
    Maps values into the given bounds.
    """
    min_ = np.min(data)
    max_ = np.max(data)
    previousRange = max_ - min_
    newRange = upperBound - lowerBound

    return [(a - min_) * newRange / previousRange + lowerBound for a in data]


def renderImage(image, heights, layer, imageWidth, imageHeight):
    """
    Generates an image with the result of the midpoint displacement algorithm
    and plots it.
    """
    colormap = ["#FF5733", "#C70039", "#900C3F"]
    # Convert the heights into the list of points of a polygon
    points = [(i, heights[i]) for i in range(0, imageWidth)]

    # Add the lower corners of the image to close the polygon
    points.insert(0, (0, imageHeight))
    points.append((imageWidth - 1, imageHeight))

    draw = ImageDraw.Draw(image)
    draw.polygon(points, fill=colormap[layer])

    return image


def main():
    args = parseArguments()

    image = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))

    layers = []
    for layer in range(args.layers):
        layer_roughness = args.roughness // (layer + 1)
        layer_heights = run_midpoint_displacement(layer_roughness, WIDTH, HEIGHT)

        layer_heights = normalize(layer_heights, HEIGHT - IMAGE_PADDING,
                                  IMAGE_PADDING + layer * 200)

        image = renderImage(image, layer_heights, layer, WIDTH, HEIGHT)

    image.show()
    image.save("landscape.png")


if __name__ == '__main__':
    main()

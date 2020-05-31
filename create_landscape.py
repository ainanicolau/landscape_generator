#!/usr/bin/env python3
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np

import midpoint_displacement as md

# Image Resolution
WIDTH = 513
HEIGHT = 720
IMAGE_PADDING = 50



# Colour Palettes
COLOUR_PALETTES = {"test":[(0,0,255,255),(0,0,150,255), (0,0,50,255)]}
# COLOUR_PALETTES = {"terracotta":{"sun":(162, 102, 81, 4), "sky":(216, 195, 165, 4),
                   # "land":[(171, 122, 106, 4), (44, 34, 25, 4)]}}

# Texture files
TEX = "paper.png"


def generate_image(width, height):

    image = np.zeros((height, width, 4), np.uint8)
    image[:, :] = (255, 255, 255, 255)

    return image


def draw_sun(image, radius, center_x, center_y):
    center = (center_x,center_y)
    colour = (0,0,0, 255)

    if radius > 0:
        cv2.circle(image, center, radius, colour, thickness=-1, lineType=8, shift=0)

    return image


def generate_mountains(image, num_layers, roughness, width, height):
    mountains = []
    for layer in range(num_layers):
        layer_roughness = roughness // (layer + 1)
        layer_heights = md.run_midpoint_displacement(layer_roughness)

        layer_heights = md.normalize(layer_heights, HEIGHT - IMAGE_PADDING,
                                  IMAGE_PADDING + layer * 200)
        mountains.append(layer_heights)

    return mountains


def draw_mountains(image, mountains, imageWidth, imageHeight,    
                  colour):

    for layer in range(len(mountains)):
        # Convert the heights into the list of points of a polygon
        points = [[i, mountains[layer][i]] for i in range(0, imageWidth)]

        # Add the lower corners of the image to close the polygon
        points.insert(0, [0, imageHeight])
        points.append([imageWidth - 1, imageHeight])

        points = np.array(points, np.int32)
        points = points.reshape((-1,1,2))

        square = np.array([[10,10], [10,100], [100,100], [100, 10], [10, 10]], np.int32)
        square = square.reshape((-1,1,2))

        cv2.fillPoly(image,[points],colour[layer])

    return image

def smooth_mountains(mountains, smooth_value):
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


class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # widget = QtWidgets.QLabel("holi")
        # self.setCentralWidget(widget)

        # Attributes to build the image
        self.__sun_radius = 5
        self.__center_x = 100
        self.__center_y = 100
        self.__mountain_layers = 3
        self.__roughness = 100
        self.__mountains = []
        self.__smoothed_mountains = []
        self.__smooth = 0

        # Sun Radius Slider
        self.__sun_radius_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.__sun_radius_slider.setMinimum(0)
        self.__sun_radius_slider.setMaximum(WIDTH)
        self.__sun_radius_slider.setValue(int(self.__sun_radius))
        self.__sun_radius_slider.valueChanged[int].connect(self.on_sun_radius_changed)

        # Center X Slider
        center_x_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        center_x_slider.setMinimum(0)
        center_x_slider.setMaximum(420)
        center_x_slider.setValue(int(self.__center_x))
        center_x_slider.valueChanged[int].connect(self.on_center_x_changed)

        # Center Y Slider
        center_y_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        center_y_slider.setMinimum(0)
        center_y_slider.setMaximum(595)
        center_y_slider.setValue(int(self.__center_y))
        center_y_slider.valueChanged[int].connect(self.on_center_y_changed)

        # Generate Mountains
        self.__mountain_layers_edit = QtWidgets.QLineEdit(str(self.__mountain_layers))
        self.__mountain_layers_edit.setFixedWidth(69)
        self.__roughness_edit = QtWidgets.QLineEdit(str(self.__roughness))
        self.__roughness_edit.setFixedWidth(69)
        self.__generate_mountains_button = QtWidgets.QPushButton('Generate Mountains');
        self.__generate_mountains_button.clicked.connect(
            self.on_generate_mountains_button_clicked)
        generate_mountains_layout = QtWidgets.QHBoxLayout()
        generate_mountains_layout.addWidget(self.__mountain_layers_edit)
        generate_mountains_layout.addWidget(self.__roughness_edit)
        generate_mountains_layout.addWidget(self.__generate_mountains_button)

        # Smooth mountains
        self.__smooth_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.__smooth_slider.setMinimum(0)
        self.__smooth_slider.setMaximum(100)
        self.__smooth_slider.setValue(int(self.__smooth))
        self.__smooth_slider.valueChanged[int].connect(self.on_smooth_changed)


        # Parameters Layout
        parameters_layout = QtWidgets.QVBoxLayout()
        parameters_layout.addWidget(QtWidgets.QLabel('Sun Radius'))
        parameters_layout.addWidget(self.__sun_radius_slider)
        parameters_layout.addWidget(QtWidgets.QLabel('Center X'))
        parameters_layout.addWidget(center_x_slider)
        parameters_layout.addWidget(QtWidgets.QLabel('Center Y'))
        parameters_layout.addWidget(center_y_slider)
        parameters_layout.addLayout(generate_mountains_layout)
        parameters_layout.addWidget(QtWidgets.QLabel('Smooth'))
        parameters_layout.addWidget(self.__smooth_slider)

        # Image
        self.__image_frame = QtWidgets.QLabel()
        self.__image = np.zeros((595, 420, 4), np.uint8)
        self.__image[:, :] = (255, 255, 255, 255)
        qImage = QtGui.QImage(
            self.__image.data, self.__image.shape[1], self.__image.shape[0],
            QtGui.QImage.Format_ARGB32)
        self.__image_frame.setPixmap(QtGui.QPixmap.fromImage(qImage))


        # Main Layout
        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(parameters_layout)
        layout.addWidget(self.__image_frame)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)


        self.setWindowTitle('Landscape')


    def on_sun_radius_changed(self, value):
        # normalized_value = value / 100
        # if normalized_value > self.__sun_radius:
        #     value = int(self.__sun_radius)# * 100)
        #     self.__sun_radius_slider.setValue(value)
        #     return
        self.__sun_radius = value


        #normalized_value
        self.__update()

    def on_center_x_changed(self, value):
        self.__center_x = value
        self.__update()

    def on_center_y_changed(self, value):
        self.__center_y = value
        self.__update()


    def on_generate_mountains_button_clicked(self):
        self.__mountains = generate_mountains(self.__image, 
                                          int(self.__mountain_layers_edit.text()), 
                                          int(self.__roughness_edit.text()), 
                                          WIDTH, HEIGHT)
        self.__smooth = 0
        self.__smooth_slider.setValue(0)
        self.__update()

    def on_smooth_changed(self, value):
        self.__smooth = value
        self.__smoothed_mountains = smooth_mountains(self.__mountains, self.__smooth)
        self.__update()


    def __update(self):

        # Generate Image
        self.__image = generate_image(WIDTH, HEIGHT)
        self.__image = draw_sun(self.__image, self.__sun_radius, self. __center_x,
                                      self.__center_y)
        mountains = self.__smoothed_mountains if self.__smooth else self.__mountains
        self.__image = draw_mountains(self.__image, mountains, WIDTH, HEIGHT,    
                                      COLOUR_PALETTES["test"])

        qImage = QtGui.QImage(
            self.__image.data, self.__image.shape[1], self.__image.shape[0],
            QtGui.QImage.Format_ARGB32)
        self.__image_frame.setPixmap(QtGui.QPixmap.fromImage(qImage))
        self.__image_frame.repaint()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
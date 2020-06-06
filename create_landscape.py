#!/usr/bin/env python3
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np

import midpoint_displacement as md

# Image Resolution
WIDTH = 420
HEIGHT = 595
IMAGE_PADDING = 50

# Color Palettes
COLOR_PALETTES = { "Terracotta":{"sun":(60, 83, 147, 255), "sky":(163, 196, 220, 255),
                   "land":[(106, 122, 171, 255), (25, 34, 44, 255)]},
                   "Desert":{"sun":(125, 187, 227, 255), "sky":(175, 206, 229, 255),
                   "land":[(44, 67, 129, 255)]},
                   "Retro":{"sun":(201, 222, 237, 255), "sky":(210, 182, 88, 255),
                   "land":[(50, 59, 222, 255), (26, 138, 232, 255)]}}
# Texture files
TEX = "paper.png"


def generate_image(width, height, color):

    image = np.zeros((height, width, 4), np.uint8)
    image[:, :] = color

    return image


def draw_sun(image, radius, center_x, center_y, color):
    center = (center_x,center_y)

    if radius > 0:
        cv2.circle(image, center, radius, color, thickness=-1, lineType=8, shift=0)

    return image


def generate_mountains(image, num_layers, roughness):
    mountains = []
    for layer in range(num_layers):
        layer_roughness = roughness // (layer + 1)
        layer_heights = md.run_midpoint_displacement(layer_roughness)

        # layer_heights = md.normalize(layer_heights, HEIGHT - IMAGE_PADDING,
        #                           IMAGE_PADDING + layer * 200)
        # layer_heights = md.normalize(layer_heights, HEIGHT - IMAGE_PADDING,
        #                           IMAGE_PADDING + HEIGHT/num_layers * layer)
        # layer_heights = md.normalize(layer_heights, HEIGHT - 100,
        #                              (HEIGHT -100-100)/num_layers * layer + 100)

        # Upper padding
        # Lower padding
        # mountain intersection

        # layer_heights = md.normalize(layer_heights, HEIGHT - 100 - ((HEIGHT -100-100)/num_layers * (num_layers-layer-1)),
        #                             (HEIGHT -100-100)/num_layers * layer + 100) 
        # layer_heights = md.normalize(layer_heights, HEIGHT - 100 - ((HEIGHT -100-upper_padding)/num_layers * (num_layers-layer-1)),
        #                             (HEIGHT -100-upper_padding)/num_layers * layer + upper_padding) 
        mountains.append(layer_heights)

    return mountains


def normalize_mountains(mountains, height, upper_padding):
    num_layers = len(mountains)
    for layer in range(num_layers):
        layer_heights = mountains[layer]
        # normalized_layer = md.normalize(layer_heights, upper_padding, height - upper_padding)
        # normalized_layer = md.normalize(layer_heights, height - 100 - ((height -100-upper_padding)/num_layers * (num_layers-layer-1)),
        #                     (height -100-upper_padding)/num_layers * layer + upper_padding)
        normalized_layer = md.normalize(layer_heights, 
                                       (height -100-upper_padding)/num_layers * layer + upper_padding,
                                        height - 100 - ((height -100-upper_padding)/num_layers * (num_layers-layer-1))) 

        mountains[layer] = normalized_layer




def draw_mountains(image, mountains, imageWidth, imageHeight, color):

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
        # layer_color = len(color)>1: color[layer%2] % 
        if len(color) > 1:
            layer_color = color[layer%2]
        else:
            initial_color = list(color[0])
            initial_color[3] = initial_color[3] / (len(mountains)) * (layer+1)
            layer_color = tuple(initial_color)
        cv2.fillPoly(image,[points],layer_color)

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
        self.__sun_radius = 0
        self.__center_x = 100
        self.__center_y = 100
        self.__mountain_layers = 3
        self.__roughness = 100
        self.__mountains = []
        self.__upper_padding = 100
        self.__smoothed_mountains = []
        self.__smooth = 0
        self.__color_palette = "Terracotta"

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

        # Upper padding
        self.__upper_padding_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.__upper_padding_slider.setMinimum(0)
        self.__upper_padding_slider.setMaximum(HEIGHT/2)
        self.__upper_padding_slider.setValue(int(self.__upper_padding))
        self.__upper_padding_slider.valueChanged[int].connect(self.on_upper_padding_changed)

        # Smooth mountains
        self.__smooth_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.__smooth_slider.setMinimum(0)
        self.__smooth_slider.setMaximum(100)
        self.__smooth_slider.setValue(int(self.__smooth))
        self.__smooth_slider.valueChanged[int].connect(self.on_smooth_changed)

        # Color Palette
        # color_palette_combobox = self.__create_combobox(COLOR_PALETTES.keys())
        self.__color_palette_combobox = QtWidgets.QComboBox()
        self.__color_palette_combobox.addItems(COLOR_PALETTES.keys())
        self.__currentIndex = list(COLOR_PALETTES.keys()).index(self.__color_palette)
        self.__color_palette_combobox.setCurrentIndex(self.__currentIndex)
        self.__color_palette_combobox.currentIndexChanged[int].connect(
            self.on_color_palette_changed)
        self.__color_palette_layout = QtWidgets.QHBoxLayout()
        self.__color_palette_layout.addWidget(self.__color_palette_combobox)


        # Parameters Layout
        parameters_layout = QtWidgets.QVBoxLayout()
        parameters_layout.addWidget(QtWidgets.QLabel('Sun Radius'))
        parameters_layout.addWidget(self.__sun_radius_slider)
        parameters_layout.addWidget(QtWidgets.QLabel('Center X'))
        parameters_layout.addWidget(center_x_slider)
        parameters_layout.addWidget(QtWidgets.QLabel('Center Y'))
        parameters_layout.addWidget(center_y_slider)
        parameters_layout.addLayout(generate_mountains_layout)
        parameters_layout.addWidget(QtWidgets.QLabel('Upper Padding'))
        parameters_layout.addWidget(self.__upper_padding_slider)
        parameters_layout.addWidget(QtWidgets.QLabel('Smooth'))
        parameters_layout.addWidget(self.__smooth_slider)
        parameters_layout.addWidget(QtWidgets.QLabel("Color Palette"))
        parameters_layout.addLayout(self.__color_palette_layout)


        # Image
        self.__image_frame = QtWidgets.QLabel()
        self.__image = np.zeros((HEIGHT, WIDTH, 4), np.uint8)
        self.__image[:, :] = COLOR_PALETTES[self.__color_palette]["sky"]
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
                                          int(self.__roughness_edit.text()))
        self.__smooth = 0
        self.__smooth_slider.setValue(0)
        self.__update()


    def on_upper_padding_changed(self, value):
        self.__upper_padding = value
        self.__update()


    def on_smooth_changed(self, value):
        self.__smooth = value
        self.__smoothed_mountains = smooth_mountains(self.__mountains, self.__smooth)
        self.__update()


    def on_color_palette_changed(self, value):
        self.__currentIndex = value
        self.__color_palette = list(COLOR_PALETTES.keys())[self.__currentIndex]
        self.__update()


    def __update(self):

        # Generate Image
        self.__image = generate_image(WIDTH, HEIGHT, COLOR_PALETTES[self.__color_palette]["sky"])
        self.__image = draw_sun(self.__image, self.__sun_radius, self. __center_x,
                                      self.__center_y, COLOR_PALETTES[self.__color_palette]["sun"])
        mountains = self.__smoothed_mountains if self.__smooth else self.__mountains

        normalize_mountains(mountains, HEIGHT, self.__upper_padding)

        self.__image = draw_mountains(self.__image, mountains, WIDTH, HEIGHT,    
                                      COLOR_PALETTES[self.__color_palette]["land"])

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
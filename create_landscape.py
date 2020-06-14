#!/usr/bin/env python3
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np
import math
from colour import Color

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
                   "land":[(50, 59, 222, 255), (26, 138, 232, 255)]},
                   "Custom": None}

MARGIN_OPTIONS = ["None", "Circle", "Window"]

MOUNTAIN_COLOR_TYPES = ["Gradient", "Set"]

SKY_ELEMENT_OPTIONS = ["Sun", "Moon"]
# Texture files
TEX = "paper.png"


def generate_image(width, height, color):

    image = np.zeros((height, width, 4), np.uint8)
    image[:, :] = color

    return image


def draw_sun(image, radius, center_x, center_y, color, white_contour, sky_element):
    center = (center_x,center_y)
    if radius > 0:
        if sky_element == "Sun":
            cv2.circle(image, center, radius, color, thickness=-1, lineType=8, shift=0)
            if white_contour:
                cv2.circle(image, center, radius, (255,255,255,255), thickness=2, lineType=8, shift=0)
        if sky_element == "Moon":
            inner_center = (center_x + int(radius/3) ,center_y - int(radius/3))
            mask = np.zeros((image.shape[0], image.shape[1], 4), np.uint8)
            filling = mask.copy()
            filling[:,:] = color
            cv2.circle(mask, center, radius, (255, 255, 255, 255), thickness=-1)
            cv2.circle(mask, inner_center, math.floor(radius/1.2), (0, 0, 0, 255), thickness=-1)
            image[mask == 255] = filling[mask == 255]
            if white_contour:
                imgray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(imgray, 127, 255, 0)
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, (255,255,255,255), 2)


    return image


def generate_mountains(image, num_layers, roughness, decrease_roughness):
    mountains = []
    for layer in range(num_layers):
        if not decrease_roughness:
            layer_roughness = roughness
        else:
            layer_roughness = roughness // (layer + 1)
        layer_heights = md.run_midpoint_displacement(layer_roughness)
        mountains.append(layer_heights)

    return mountains


def normalize_mountains(mountains, height, lower_padding, upper_padding, mountain_intersection):
    num_layers = len(mountains)
    for layer in range(num_layers):
        layer_heights = mountains[layer]
        normalized_layer = md.normalize(layer_heights, 
                                       (height - lower_padding - upper_padding)/num_layers * layer + upper_padding,
                                        height - lower_padding - ((height - lower_padding -upper_padding)/num_layers * (num_layers-layer-1)) * mountain_intersection/100) 

        mountains[layer] = normalized_layer


def draw_mountains(image, mountains, imageWidth, imageHeight, mountain_color, sky_color, white_contour):

    # if len(mountain_color) == 1:
    colors = interpolate_colors(mountain_color[0], sky_color,len(mountains)+1)

    for layer in range(len(mountains)):
        # Convert the heights into the list of points of a polygon
        points = [[i, mountains[layer][i]] for i in range(0, imageWidth)]

        # Add the lower corners of the image to close the polygon
        points.insert(0, [0, imageHeight])
        points.append([imageWidth - 1, imageHeight])
        points = np.array(points, np.int32)
        points = points.reshape((-1,1,2))

        if len(mountain_color) > 1:
            layer_color = mountain_color[layer%2]
        else:
            layer_color = colors[len(mountains) - layer - 1]
        cv2.fillPoly(image,[points],layer_color)

        if white_contour:
            cv2.polylines(image,[points],True,(255,255,255,255),2)

    return image


def interpolate_colors(color1, color2, divisions):
    colors = [] 
    for channel in range(3):
        values = np.linspace(color1[channel], color2[channel], divisions)
        colors.append([ int(x) for x in values ])
    colors.append([255]*divisions)

    colors=list(map(tuple, zip(*colors)))
    return colors


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


def draw_margin(image, margin, width, height):
    mask = np.zeros((height, width, 4), np.uint8)
    out = np.ones((height, width, 4), np.uint8)*255
    mask[:,:,3] = 255
    if margin == "Circle":
        radius = math.floor(min(width, height)/2) - 30     
        center = (math.floor(width/2), math.floor(height/2))
        cv2.circle(mask, center, radius, (255, 255, 255, 255), thickness=-1)

    if margin == "Window":
        radius = math.floor(min(width, height)/2) - 50
        center = (math.floor(width/2), math.floor(height/2) - 50)
        cv2.circle(mask, center, radius, (255, 255, 255, 255), thickness=-1)
        top_left = (center[0] - radius, center[1])
        bottom_right = (center[0] + radius, center[1] + math.floor(radius * 1.5))
        cv2.rectangle(mask,top_left,bottom_right,(255, 255, 255, 255),-1)

    out[mask == 255] = image[mask == 255]

    return out


class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # widget = QtWidgets.QLabel("holi")
        # self.setCentralWidget(widget)

        # Attributes to build the image
        self.__sky_element = "Sun"
        self.__sun_radius = 0
        self.__center_x = 100
        self.__center_y = 100
        self.__mountain_layers = 3
        self.__roughness = 100
        self.__decrease_roughness = 2
        self.__mountains = []
        self.__upper_padding = 100
        self.__lower_padding = 100
        self.__mountain_intersection = 0
        self.__smoothed_mountains = []
        self.__smooth = 0
        self.__color_palette = "Terracotta"
        self.__sky_color = COLOR_PALETTES[self.__color_palette]["sky"]
        self.__sun_color = COLOR_PALETTES[self.__color_palette]["sun"]
        if len(COLOR_PALETTES[self.__color_palette]["land"]) > 1:
            self.__mountain_color_type = "Set"
        else:
            self.__mountain_color_type = "Gradient"
        self.__gradient_color = COLOR_PALETTES[self.__color_palette]["land"][0]
        self.__white_contour = 0
        self.__margin = "None"

        # Sky Element
        self.__sky_element_combobox = QtWidgets.QComboBox()
        self.__sky_element_combobox.addItems(SKY_ELEMENT_OPTIONS)
        self.__currentSkyElementIndex = 0
        self.__sky_element_combobox.setCurrentIndex(self.__currentSkyElementIndex)
        self.__sky_element_combobox.currentIndexChanged[int].connect(
            self.on_sky_element_changed)
        self.__sky_element_layout = QtWidgets.QHBoxLayout()
        self.__sky_element_layout.addWidget(QtWidgets.QLabel("Sky Element"))
        self.__sky_element_layout.addWidget(self.__sky_element_combobox)

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
        # Mountain Layers
        self.__mountain_layers_edit = QtWidgets.QLineEdit(str(self.__mountain_layers))
        self.__mountain_layers_edit.setFixedWidth(69)
        # Roughness
        self.__roughness_edit = QtWidgets.QLineEdit(str(self.__roughness))
        self.__roughness_edit.setFixedWidth(69)
        # Decrease roughness
        decrease_roughness_checkbox = QtWidgets.QCheckBox('Decrease roughness')
        decrease_roughness_checkbox.setCheckState(self.__decrease_roughness)
        decrease_roughness_checkbox.stateChanged[int].connect(
            self.on_decrease_roughness_changed)
        # Generate Mountains Button
        self.__generate_mountains_button = QtWidgets.QPushButton('Generate Mountains');
        self.__generate_mountains_button.clicked.connect(
            self.on_generate_mountains_button_clicked)
        # Generate Mountains Layout
        generate_mountains_layout = QtWidgets.QHBoxLayout()
        generate_mountains_layout.addWidget(QtWidgets.QLabel('Mountain Layers'))
        generate_mountains_layout.addWidget(self.__mountain_layers_edit)
        generate_mountains_layout.addWidget(QtWidgets.QLabel('Roughness'))
        generate_mountains_layout.addWidget(self.__roughness_edit)
        generate_mountains_layout.addWidget(decrease_roughness_checkbox)
        generate_mountains_layout.addWidget(self.__generate_mountains_button)

        # Upper padding
        self.__upper_padding_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.__upper_padding_slider.setMinimum(0)
        self.__upper_padding_slider.setMaximum(HEIGHT)
        self.__upper_padding_slider.setValue(int(self.__upper_padding))
        self.__upper_padding_slider.valueChanged[int].connect(self.on_upper_padding_changed)

        # Lower padding
        self.__lower_padding_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.__lower_padding_slider.setMinimum(0)
        self.__lower_padding_slider.setMaximum(HEIGHT)
        self.__lower_padding_slider.setValue(int(self.__lower_padding))
        self.__lower_padding_slider.valueChanged[int].connect(self.on_lower_padding_changed)

        # Mountain Intersecion
        self.__mountain_intersection_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.__mountain_intersection_slider.setMinimum(0)
        self.__mountain_intersection_slider.setMaximum(100)
        self.__mountain_intersection_slider.setValue(int(self.__mountain_intersection))
        self.__mountain_intersection_slider.valueChanged[int].connect(self.on_mountain_intersection_changed)

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
        self.__currentPaletteIndex = list(COLOR_PALETTES.keys()).index(self.__color_palette)
        self.__color_palette_combobox.setCurrentIndex(self.__currentPaletteIndex)
        self.__color_palette_combobox.currentIndexChanged[int].connect(
            self.on_color_palette_changed)
        self.__color_palette_layout = QtWidgets.QHBoxLayout()
        self.__color_palette_layout.addWidget(self.__color_palette_combobox)

        # Custom Colors
        self.__custom_colors_layout = QtWidgets.QHBoxLayout()
        # Sky Color
        self.__sky_color_button = QtWidgets.QPushButton()
        background = (self.__sky_color[2], self.__sky_color[1], self.__sky_color[0])
        self.__sky_color_button.setStyleSheet("background-color:rgb{}; border-style: outset; border: none; border-radius: 4px;max-width: 5em;".format(background))
        self.__sky_color_button.clicked.connect(
            self.on_sky_color_button_clicked)
        self.__custom_colors_layout.addWidget(QtWidgets.QLabel("Sky Color"))
        self.__custom_colors_layout.addWidget(self.__sky_color_button)

        # Sun Color
        self.__sun_color_button = QtWidgets.QPushButton()
        background = (self.__sun_color[2], self.__sun_color[1], self.__sun_color[0])
        self.__sun_color_button.setStyleSheet("background-color:rgb{}; border-style: outset; border: none; border-radius: 4px;max-width: 5em;".format(background))
        self.__sun_color_button.clicked.connect(
            self.on_sun_color_button_clicked)
        self.__custom_colors_layout.addWidget(QtWidgets.QLabel("Sun Color"))
        self.__custom_colors_layout.addWidget(self.__sun_color_button)

        # Mountain Color
        self.__mountain_color_type_combobox = QtWidgets.QComboBox()
        self.__mountain_color_type_combobox.addItems(MOUNTAIN_COLOR_TYPES)
        self.__currentMountainColorTypeIndex = MOUNTAIN_COLOR_TYPES.index(self.__mountain_color_type)
        self.__mountain_color_type_combobox.setCurrentIndex(self.__currentMountainColorTypeIndex)
        self.__mountain_color_type_combobox.currentIndexChanged[int].connect(
            self.on_mountain_color_type_changed)
        self.__custom_colors_layout.addWidget(QtWidgets.QLabel("Mountain Color Type"))
        self.__custom_colors_layout.addWidget(self.__mountain_color_type_combobox)

        # Mountain Gradient Color
        self.__gradient_color_button = QtWidgets.QPushButton()
        background = (self.__gradient_color[2], self.__gradient_color[1], self.__gradient_color[0])
        self.__gradient_color_button.setStyleSheet("background-color:rgb{}; border-style: outset; border: none; border-radius: 4px;max-width: 5em;".format(background))
        self.__gradient_color_button.clicked.connect(
            self.on_gradient_color_button_clicked)
        self.__custom_colors_layout.addWidget(QtWidgets.QLabel("Gradient Color"))
        self.__custom_colors_layout.addWidget(self.__gradient_color_button)


        # White Contour
        white_contour_checkbox = QtWidgets.QCheckBox('White Contour')
        white_contour_checkbox.setCheckState(self.__white_contour)
        white_contour_checkbox.stateChanged[int].connect(
            self.on_white_contour_changed)

        # Margin
        self.__margin_combobox = QtWidgets.QComboBox()
        self.__margin_combobox.addItems(MARGIN_OPTIONS)
        self.__currentMarginIndex = 0
        self.__margin_combobox.setCurrentIndex(self.__currentMarginIndex)
        self.__margin_combobox.currentIndexChanged[int].connect(
            self.on_margin_changed)
        self.__margin_layout = QtWidgets.QHBoxLayout()
        self.__margin_layout.addWidget(QtWidgets.QLabel("Margin"))
        self.__margin_layout.addWidget(self.__margin_combobox)

        # Parameters Layout
        parameters_layout = QtWidgets.QVBoxLayout()
        parameters_layout.addLayout(self.__sky_element_layout)
        parameters_layout.addWidget(QtWidgets.QLabel('Sun Radius'))
        parameters_layout.addWidget(self.__sun_radius_slider)
        parameters_layout.addWidget(QtWidgets.QLabel('Center X'))
        parameters_layout.addWidget(center_x_slider)
        parameters_layout.addWidget(QtWidgets.QLabel('Center Y'))
        parameters_layout.addWidget(center_y_slider)
        parameters_layout.addLayout(generate_mountains_layout)
        parameters_layout.addWidget(QtWidgets.QLabel('Upper Padding'))
        parameters_layout.addWidget(self.__upper_padding_slider)
        parameters_layout.addWidget(QtWidgets.QLabel('Lower Padding'))
        parameters_layout.addWidget(self.__lower_padding_slider)
        parameters_layout.addWidget(QtWidgets.QLabel('Mountain Intersecion'))
        parameters_layout.addWidget(self.__mountain_intersection_slider)
        parameters_layout.addWidget(QtWidgets.QLabel('Smooth'))
        parameters_layout.addWidget(self.__smooth_slider)
        parameters_layout.addWidget(QtWidgets.QLabel("Color Palette"))
        parameters_layout.addLayout(self.__color_palette_layout)
        parameters_layout.addLayout(self.__custom_colors_layout)
        parameters_layout.addWidget(white_contour_checkbox)
        parameters_layout.addLayout(self.__margin_layout)


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


    def on_sky_element_changed(self, value):
        self.__currentSkyElementIndex = value
        self.__sky_element = SKY_ELEMENT_OPTIONS[self.__currentSkyElementIndex]
        self.__update()


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


    def on_decrease_roughness_changed(self, value):
        self.__decrease_roughness = value
        self.__update()


    def on_generate_mountains_button_clicked(self):
        self.__mountains = generate_mountains(self.__image, 
                                          int(self.__mountain_layers_edit.text()), 
                                          int(self.__roughness_edit.text()),
                                          self.__decrease_roughness)
        self.__smooth = 0
        self.__smooth_slider.setValue(0)
        self.__update()


    def on_upper_padding_changed(self, value):
        if value < HEIGHT - self.__lower_padding:
            self.__upper_padding = value
        else:
            self.__upper_padding = HEIGHT - self.__lower_padding - 1

        self.__update()


    def on_lower_padding_changed(self, value):
        if value < HEIGHT - self.__upper_padding:
            self.__lower_padding = value
        else:
            self.__lower_padding = HEIGHT - self.__upper_padding - 1
        self.__update()


    def on_mountain_intersection_changed(self, value):
        self.__mountain_intersection = value
        self.__update() 


    def on_smooth_changed(self, value):
        self.__smooth = value
        self.__smoothed_mountains = smooth_mountains(self.__mountains, self.__smooth)
        self.__update()


    def on_color_palette_changed(self, value):
        self.__currentPaletteIndex = value
        self.__color_palette = list(COLOR_PALETTES.keys())[self.__currentPaletteIndex]
        self.__sky_color = COLOR_PALETTES[self.__color_palette]["sky"]
        background = (self.__sky_color[2], self.__sky_color[1], self.__sky_color[0])
        self.__sky_color_button.setStyleSheet("background-color:rgb{}; border-style: outset; border: none; border-radius: 4px;max-width: 5em;".format(background))
        self.__sun_color = COLOR_PALETTES[self.__color_palette]["sun"]
        background = (self.__sun_color[2], self.__sun_color[1], self.__sun_color[0])
        self.__sun_color_button.setStyleSheet("background-color:rgb{}; border-style: outset; border: none; border-radius: 4px;max-width: 5em;".format(background))
        self.__update()


    def on_sky_color_button_clicked(self, value):
        selected_color = QtWidgets.QColorDialog().getColor().getRgb()
        self.__sky_color_button.setStyleSheet("background-color:rgb{}; border-style: outset; border: none; border-radius: 4px;max-width: 5em;".format(selected_color))
        self.__sky_color = (selected_color[2], selected_color[1], selected_color[0], 255)
        # self.__color_palette = "Custom"
        # self.__currentPaletteIndex = list(COLOR_PALETTES.keys()).index(self.__color_palette)
        self.__update()


    def on_sun_color_button_clicked(self, value):
        selected_color = QtWidgets.QColorDialog().getColor().getRgb()
        self.__sun_color_button.setStyleSheet("background-color:rgb{}; border-style: outset; border: none; border-radius: 4px;max-width: 5em;".format(selected_color))
        self.__sun_color = (selected_color[2], selected_color[1], selected_color[0], 255)
        # self.__color_palette = "Custom"
        # self.__currentPaletteIndex = list(COLOR_PALETTES.keys()).index(self.__color_palette)
        self.__update()


    def on_mountain_color_type_changed(self, value):
        pass


    def on_gradient_color_button_clicked(self, value):
        selected_color = QtWidgets.QColorDialog().getColor().getRgb()
        self.__gradient_color_button.setStyleSheet("background-color:rgb{}; border-style: outset; border: none; border-radius: 4px;max-width: 5em;".format(selected_color))
        self.__gradient_color = (selected_color[2], selected_color[1], selected_color[0], 255)
        # self.__color_palette = "Custom"
        # self.__currentPaletteIndex = list(COLOR_PALETTES.keys()).index(self.__color_palette)
        self.__update()


    def on_white_contour_changed(self, value):
        self.__white_contour = value
        self.__update()


    def on_margin_changed(self, value):
        self.__currentMarginIndex = value
        self.__margin = MARGIN_OPTIONS[self.__currentMarginIndex]
        self.__update()


    def __update(self):

        # Generate Image
        self.__image = generate_image(WIDTH, HEIGHT, self.__sky_color)
        self.__image = draw_sun(self.__image, self.__sun_radius, self. __center_x,
                                      self.__center_y, self.__sun_color,
                                      self.__white_contour, self.__sky_element)
        mountains = self.__smoothed_mountains if self.__smooth else self.__mountains

        normalize_mountains(mountains, HEIGHT, self.__lower_padding, self.__upper_padding, self.__mountain_intersection)

        # self.__image = draw_mountains(self.__image, mountains, WIDTH, HEIGHT,    
        #                               COLOR_PALETTES[self.__color_palette], self.__white_contour)
        self.__image = draw_mountains(self.__image, mountains, WIDTH, HEIGHT,    
                                      [self.__gradient_color], self.__sky_color, self.__white_contour)

        if not self.__margin == "None":
            self.__image = draw_margin(self.__image, self.__margin, WIDTH, HEIGHT)

        # self.__image = cv2.blur(self.__image,(2,2))

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

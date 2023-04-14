import cv2
import math
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from create_landscape import (
    apply_texture,
    generate_image,
    generate_mountains,
    smooth_mountains,
    draw_sun,
    normalize_mountains,
    draw_mountains,
    draw_margin,
)

# Image Resolution
WIDTH = 2480
HEIGHT = 3508

# Color Palettes
COLOR_PALETTES = {
    "Terracotta": {
        "sun": (60, 83, 147, 255),
        "sky": (163, 196, 220, 255),
        "land": [
            (106, 122, 171, 255),
            (100, 100, 100, 255),
            (25, 34, 44, 255),
        ],
    },
    "Desert": {
        "sun": (125, 187, 227, 255),
        "sky": (175, 206, 229, 255),
        "land": [(44, 67, 129, 255)],
    },
    "Retro": {
        "sun": (201, 222, 237, 255),
        "sky": (210, 182, 88, 255),
        "land": [
            (50, 59, 222, 255),
            (38, 87, 228, 255),
            (26, 138, 232, 255),
            (60, 166, 237, 255),
        ],
    },
    "Candy": {
        "sun": (194, 176, 187, 255),
        "sky": (169, 143, 209, 255),
        "land": [
            (55, 96, 168, 255),
            (102, 138, 215, 255),
            (144, 170, 206, 255),
            (93, 104, 214, 255),
            (84, 82, 189, 255),
        ],
    },
    "Gold": {
        "sun": (58, 148, 201, 255),
        "sky": (179, 201, 206, 255),
        "land": [(66, 59, 116, 255), (45, 90, 163, 255), (101, 124, 180, 255)],
    },
    "Night": {
        "sun": (239, 249, 237, 255),
        "sky": (30, 25, 27, 255),
        "land": [
            (149, 140, 142, 255),
            (207, 220, 246, 255),
            (74, 76, 86, 255),
            (138, 150, 206, 255),
            (240, 245, 248, 255),
            (207, 215, 186, 255),
        ],
    },
    "Forest": {
        "sun": (181, 263, 245, 255),
        "sky": (148, 230, 201, 255),
        "land": [(37, 30, 15, 255)],
    },
    "Vintage": {
        "sun": (171, 189, 220, 255),
        "sky": (71, 63, 63, 255),
        "land": [
            (60, 170, 242, 255),
            (171, 189, 220, 255),
            (74, 88, 82, 255),
            (103, 141, 173, 255),
            (57, 114, 196, 255),
        ],
    },
    "Peach": {
        "sun": (106, 141, 210, 255),
        "sky": (226, 235, 244, 255),
        "land": [
            (147, 183, 217, 255),
            (118, 136, 190, 255),
            (106, 141, 210, 255),
            (110, 125, 169, 255),
        ],
    },
    "Summer": {
        "sun": (95, 139, 234, 255),
        "sky": (245, 240, 255, 255),
        "land": [(147, 166, 87, 255)],
    },
    "Tropical": {
        "sun": (205, 186, 245, 255),
        "sky": (226, 231, 235, 255),
        "land": [(150, 196, 77, 255)],
    },
    "Mono": {
        "sun": (101, 134, 197, 255),
        "sky": (101, 134, 197, 255),
        "land": [(101, 134, 197, 255), (101, 134, 197, 255)],
    },
}

# Combobox options
MARGIN_OPTIONS = ["None", "Circle", "Window"]
SKY_ELEMENT_OPTIONS = ["Sun", "Moon"]

# Textures
TEX = "texture.jpg"
TEX_LOW = "texture_low.jpg"

# Buttons style
STYLE = (
    "background-color:rgb{};"
    "border-style: outset;"
    "border: none; "
    "border-radius: 4px;"
)


class CreateLandscapeGUI(QtWidgets.QMainWindow):
    """
    A graphical user interface for generating landscape images with
    customizable settings.

    The user interface provides controls for adjusting the appearance of the
    sky, sun, mountains, and other elements of the landscape, as well as
    options for saving the images. The main window displays a preview of the
    landscape image with the current settings.
    """

    def __init__(self):
        """
        Attributes:
            __image (np.ndarray): The current landscape image as a numpy array.
            __sky_color (Tuple[): The RGB color of the sky background.
            __sun_color (Tuple): The RGB color of the sun.
            __sun_radius (int): The radius of the sun in pixels.
            __white_contour (bool): Whether to draw a white contour around the
                sun.
            __sky_element (bool): Whether to draw an element in the sky.
            __mountains (List[float]): The initial generated mountain heights.
            __smoothed_mountains (List[float]): The smoothed mountain heights.
            __land_color (List): The colors of the land gradient.
            __gradient_color (Tuple): The current selected color for the land.
            __color_palette (str): The current selected color palette.
            __lower_padding (int): The lower padding of the mountain in pixels.
            __upper_padding (int): The upper padding of the mountain in pixels.
            __mountain_intersection (float): The intersection point of the
                mountains in the range [0, 1].
            __smooth (bool): Whether to use the smoothed mountains or the
                initial mountains for rendering.
            __margin (str): The type of margin to apply to the final image.
            __currentMarginIndex (int): The index of the current margin option
                in the menu.
            __center_x (int): The x-coordinate of the center of the image.
            __center_y (int): The y-coordinate of the center of the image.
            __image_name_edit (QtWidgets.QLineEdit): The line edit for entering
                the image name.
            __image_frame (QtWidgets.QLabel): The label for displaying the
                landscape image preview.
        """
        super().__init__()

        self.__initialize_defaults()

        # Sky Element Group
        sky_element_group = QtWidgets.QGroupBox()
        sky_element_group.setTitle("Sky Element")
        sky_element_layout = QtWidgets.QHBoxLayout()
        sky_element_center_layout = QtWidgets.QHBoxLayout()

        # Sky Element
        sky_element_combobox = QtWidgets.QComboBox()
        sky_element_combobox.addItems(SKY_ELEMENT_OPTIONS)
        currentSkyElementIndex = 0
        sky_element_combobox.setCurrentIndex(currentSkyElementIndex)
        sky_element_combobox.currentIndexChanged[int].connect(
            self.on_sky_element_changed
        )
        sky_element_layout.addWidget(QtWidgets.QLabel("Shape"))
        sky_element_layout.addWidget(sky_element_combobox)

        # Sun Radius Slider
        sun_radius_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sun_radius_slider.setMinimum(0)
        sun_radius_slider.setMaximum(WIDTH)
        sun_radius_slider.setValue(int(self.__sun_radius))
        sun_radius_slider.valueChanged[int].connect(self.on_sun_radius_changed)
        sky_element_layout.addWidget(QtWidgets.QLabel("Radius"))
        sky_element_layout.addWidget(sun_radius_slider)

        # Center X Slider
        center_x_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        center_x_slider.setMinimum(0)
        center_x_slider.setMaximum(100)
        center_x_slider.setValue(int(self.__center_x))
        center_x_slider.valueChanged[int].connect(self.on_center_x_changed)
        sky_element_center_layout.addWidget(QtWidgets.QLabel("Center: "))
        sky_element_center_layout.addWidget(QtWidgets.QLabel("X"))
        sky_element_center_layout.addWidget(center_x_slider)

        # Center Y Slider
        center_y_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        center_y_slider.setMinimum(0)
        center_y_slider.setMaximum(100)
        center_y_slider.setValue(int(self.__center_y))
        center_y_slider.valueChanged[int].connect(self.on_center_y_changed)
        sky_element_center_layout.addWidget(QtWidgets.QLabel("Y"))
        sky_element_center_layout.addWidget(center_y_slider)

        sky_element_v_layout = QtWidgets.QVBoxLayout()
        sky_element_v_layout.addLayout(sky_element_layout)
        sky_element_v_layout.addLayout(sky_element_center_layout)

        sky_element_group.setLayout(sky_element_v_layout)

        # Generate Mountains
        # Mountains Group
        mountain_group = QtWidgets.QGroupBox()
        mountain_group.setTitle("Mountains")
        mountains_layout = QtWidgets.QVBoxLayout()
        # Mountain Layers
        self.__mountain_layers_edit = QtWidgets.QLineEdit(
            str(self.__mountain_layers)
        )
        self.__mountain_layers_edit.setFixedWidth(69)
        # Roughness
        self.__roughness_edit = QtWidgets.QLineEdit(str(self.__roughness))
        self.__roughness_edit.setFixedWidth(69)
        # Decrease roughness
        decrease_roughness_checkbox = QtWidgets.QCheckBox("Decrease roughness")
        decrease_roughness_checkbox.setCheckState(self.__decrease_roughness)
        decrease_roughness_checkbox.stateChanged[int].connect(
            self.on_decrease_roughness_changed
        )
        # Generate Mountains Button
        generate_mountains_button = QtWidgets.QPushButton("Generate Mountains")
        generate_mountains_button.clicked.connect(
            self.on_generate_mountains_button_clicked
        )
        # Generate Mountains Layout
        generate_mountains_layout = QtWidgets.QHBoxLayout()
        generate_mountains_layout.addWidget(QtWidgets.QLabel("Layers"))
        generate_mountains_layout.addWidget(self.__mountain_layers_edit)
        generate_mountains_layout.addWidget(QtWidgets.QLabel("Roughness"))
        generate_mountains_layout.addWidget(self.__roughness_edit)
        generate_mountains_layout.addWidget(decrease_roughness_checkbox)
        generate_mountains_layout.addWidget(generate_mountains_button)
        mountains_layout.addLayout(generate_mountains_layout)

        padding_layout = QtWidgets.QHBoxLayout()
        padding_layout.addWidget(QtWidgets.QLabel("Padding: "))
        # Upper padding
        upper_padding_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        upper_padding_slider.setMinimum(0)
        upper_padding_slider.setMaximum(HEIGHT)
        upper_padding_slider.setValue(int(self.__upper_padding))
        upper_padding_slider.valueChanged[int].connect(
            self.on_upper_padding_changed
        )
        padding_layout.addWidget(QtWidgets.QLabel("Upper"))
        padding_layout.addWidget(upper_padding_slider)

        # Lower padding
        lower_padding_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        lower_padding_slider.setMinimum(0)
        lower_padding_slider.setMaximum(HEIGHT)
        lower_padding_slider.setValue(int(self.__lower_padding))
        lower_padding_slider.valueChanged[int].connect(
            self.on_lower_padding_changed
        )
        padding_layout.addWidget(QtWidgets.QLabel("Lower"))
        padding_layout.addWidget(lower_padding_slider)
        mountains_layout.addLayout(padding_layout)

        mountain_modifier_layout = QtWidgets.QHBoxLayout()
        # Mountain Intersecion
        mountain_intersection_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        mountain_intersection_slider.setMinimum(0)
        mountain_intersection_slider.setMaximum(100)
        mountain_intersection_slider.setValue(
            int(self.__mountain_intersection)
        )
        mountain_intersection_slider.valueChanged[int].connect(
            self.on_mountain_intersection_changed
        )
        mountain_modifier_layout.addWidget(QtWidgets.QLabel("Intersecions"))
        mountain_modifier_layout.addWidget(mountain_intersection_slider)

        # Smooth mountains
        self.__smooth_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.__smooth_slider.setMinimum(0)
        self.__smooth_slider.setMaximum(100)
        self.__smooth_slider.setValue(int(self.__smooth))
        self.__smooth_slider.valueChanged[int].connect(self.on_smooth_changed)
        mountain_modifier_layout.addWidget(QtWidgets.QLabel("Smooth"))
        mountain_modifier_layout.addWidget(self.__smooth_slider)
        mountains_layout.addLayout(mountain_modifier_layout)
        mountain_group.setLayout(mountains_layout)

        # Colors Group
        colors_group = QtWidgets.QGroupBox()
        colors_group.setTitle("Colors")
        colors_layout = QtWidgets.QVBoxLayout()
        # Color Palette
        # color_palette_combobox = self.__create_combobox(COLOR_PALETTES.keys()
        color_palette_combobox = QtWidgets.QComboBox()
        color_palette_combobox.addItems(COLOR_PALETTES.keys())
        currentPaletteIndex = list(COLOR_PALETTES.keys()).index(
            self.__color_palette
        )
        color_palette_combobox.setCurrentIndex(currentPaletteIndex)
        color_palette_combobox.currentIndexChanged[int].connect(
            self.on_color_palette_changed
        )
        color_palette_layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("Palette")
        label.setFixedWidth(60)
        color_palette_layout.addWidget(label)
        color_palette_layout.addWidget(color_palette_combobox)

        # Reset palette
        reset_palette_button = QtWidgets.QPushButton("Reset Palette")
        reset_palette_button.clicked.connect(
            self.on_reset_palette_button_clicked
        )
        color_palette_layout.addWidget(reset_palette_button)
        colors_layout.addLayout(color_palette_layout)

        # Custom Colors
        custom_colors_layout = QtWidgets.QHBoxLayout()
        # Sky Color
        self.__sky_color_button = QtWidgets.QPushButton()
        background = (
            self.__sky_color[2],
            self.__sky_color[1],
            self.__sky_color[0],
        )
        self.__sky_color_button.setStyleSheet(STYLE.format(background))
        self.__sky_color_button.clicked.connect(
            self.on_sky_color_button_clicked
        )
        label = QtWidgets.QLabel("Sky")
        label.setFixedWidth(30)
        custom_colors_layout.addWidget(label)
        custom_colors_layout.addWidget(self.__sky_color_button)

        # Sun Color
        self.__sun_color_button = QtWidgets.QPushButton()
        background = (
            self.__sun_color[2],
            self.__sun_color[1],
            self.__sun_color[0],
        )
        self.__sun_color_button.setStyleSheet(STYLE.format(background))
        self.__sun_color_button.clicked.connect(
            self.on_sun_color_button_clicked
        )
        label = QtWidgets.QLabel("Sun")
        label.setFixedWidth(30)
        custom_colors_layout.addWidget(label)
        custom_colors_layout.addWidget(self.__sun_color_button)

        # Land Color
        self.__gradient_color_button = QtWidgets.QPushButton()
        background = (
            self.__gradient_color[2],
            self.__gradient_color[1],
            self.__gradient_color[0],
        )
        self.__gradient_color_button.setStyleSheet(STYLE.format(background))
        self.__gradient_color_button.clicked.connect(
            self.on_gradient_color_button_clicked
        )
        label = QtWidgets.QLabel("Land")
        label.setFixedWidth(30)
        custom_colors_layout.addWidget(label)
        custom_colors_layout.addWidget(self.__gradient_color_button)

        colors_layout.addLayout(custom_colors_layout)
        colors_group.setLayout(colors_layout)

        # Details Group
        details_group = QtWidgets.QGroupBox()
        details_group.setTitle("Details")
        details_layout = QtWidgets.QHBoxLayout()

        # White Contour
        white_contour_checkbox = QtWidgets.QCheckBox("White Contour")
        white_contour_checkbox.setCheckState(self.__white_contour)
        white_contour_checkbox.stateChanged[int].connect(
            self.on_white_contour_changed
        )
        details_layout.addWidget(white_contour_checkbox)

        # Margin
        margin_combobox = QtWidgets.QComboBox()
        margin_combobox.addItems(MARGIN_OPTIONS)
        currentMarginIndex = 0
        margin_combobox.setCurrentIndex(currentMarginIndex)
        margin_combobox.currentIndexChanged[int].connect(
            self.on_margin_changed
        )
        margin_layout = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("Margin")
        label.setFixedWidth(60)
        margin_layout.addWidget(label)
        margin_layout.addWidget(margin_combobox)
        details_layout.addLayout(margin_layout)
        details_group.setLayout(details_layout)

        # Save Image
        save_image_layout = QtWidgets.QHBoxLayout()
        self.__image_name_edit = QtWidgets.QLineEdit(self.__image_name)
        save_image_layout.addWidget(self.__image_name_edit)
        save_image_button = QtWidgets.QPushButton("Save")
        save_image_button.clicked.connect(self.on_save_image_button_clicked)
        save_image_layout.addWidget(save_image_button)

        # Parameters Layout
        parameters_layout = QtWidgets.QVBoxLayout()
        parameters_layout.addWidget(sky_element_group)
        parameters_layout.addWidget(mountain_group)
        parameters_layout.addWidget(colors_group)
        parameters_layout.addWidget(details_group)
        parameters_layout.addLayout(save_image_layout)

        # Image
        self.__image_frame = QtWidgets.QLabel()
        self.__image = np.zeros((HEIGHT, WIDTH, 4), np.uint8)
        self.__image[:, :] = COLOR_PALETTES[self.__color_palette]["sky"]
        resized = cv2.resize(
            self.__image, (496, 702), interpolation=cv2.INTER_LINEAR
        )
        resized = apply_texture(resized, TEX_LOW, 0.5)
        qImage = QtGui.QImage(
            resized.data,
            resized.shape[1],
            resized.shape[0],
            QtGui.QImage.Format_ARGB32,
        )
        self.__image_frame.setPixmap(QtGui.QPixmap.fromImage(qImage))

        # Main Layout
        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(parameters_layout)
        layout.addWidget(self.__image_frame)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.setWindowTitle("Boho Minimalist Landscape Generator")

    def __initialize_defaults(self):
        """
        Initializes the default values for the various parameters used in
        generating the landscape image.
        """
        self.__sky_element = "Sun"
        self.__sun_radius = 0
        self.__center_x = 0
        self.__center_y = 0
        self.__mountain_layers = 3
        self.__roughness = 300
        self.__decrease_roughness = 2
        self.__mountains = []
        self.__upper_padding = 100
        self.__lower_padding = 100
        self.__mountain_intersection = 0
        self.__smoothed_mountains = []
        self.__smooth = 0
        self.__color_palette = "Desert"
        self.__sky_color = COLOR_PALETTES[self.__color_palette]["sky"]
        self.__sun_color = COLOR_PALETTES[self.__color_palette]["sun"]
        self.__gradient_color = COLOR_PALETTES[self.__color_palette]["land"][0]
        self.__land_color = COLOR_PALETTES[self.__color_palette]["land"]
        self.__white_contour = 0
        self.__margin = "None"
        self.__image_name = "myLandscape.png"

    def on_sky_element_changed(self, value):
        """
        Updates the sky element and triggers an update of the display.

        Args:
            value (int): Index of the new sky element to be displayed.
        """
        self.__currentSkyElementIndex = value
        self.__sky_element = SKY_ELEMENT_OPTIONS[self.__currentSkyElementIndex]
        self.__update_display()

    def on_sun_radius_changed(self, value):
        """
        Updates the radius of the sun and triggers an update of the display.

        Args:
            value (int): The new radius of the sun.
        """
        self.__sun_radius = value
        self.__update_display()

    def on_center_x_changed(self, value):
        """
        Updates the x-coordinate of the center of the sun and triggers an
        update of the display.

        Args:
            value (int): The new x-coordinate of the center of the sun as a
            percentage of the width of the display.
        """
        self.__center_x = math.floor((value / 100) * WIDTH)
        self.__update_display()

    def on_center_y_changed(self, value):
        """
        Updates the y-coordinate of the center of the sun and triggers an
        update of the display.

        Args:
            value (int): The new y-coordinate of the center of the sun as a
            percentage of the height of the display.
        """
        self.__center_y = math.floor((value / 100) * HEIGHT)
        self.__update_display()

    def on_decrease_roughness_changed(self, value):
        """
        Update the decrease roughness value based on the slider value and
        update the display.

        Args:
            value: a float representing the new decrease roughness value
        """
        self.__decrease_roughness = value
        self.__update_display()

    def on_generate_mountains_button_clicked(self):
        """
        Generate new mountains based on the current parameters and update the
        display.
        """
        self.__mountains = generate_mountains(
            self.__image,
            int(self.__mountain_layers_edit.text()),
            int(self.__roughness_edit.text()),
            self.__decrease_roughness,
            WIDTH,
            HEIGHT,
        )
        self.__smooth = 0
        self.__smooth_slider.setValue(0)
        self.__update_display()

    def on_upper_padding_changed(self, value):
        """
        Update the upper padding value based on the slider value and update the
        display. The upper padding cannot exceed the height of the landscape
        minus the lower padding.

        Args:
            value: an integer representing the new upper padding value
        """
        if value < HEIGHT - self.__lower_padding:
            self.__upper_padding = value
        else:
            self.__upper_padding = HEIGHT - self.__lower_padding - 1

        self.__update_display()

    def on_lower_padding_changed(self, value):
        """
        Update the lower padding value based on the slider value and update the
        display. The lower padding cannot exceed the height of the landscape
        minus the upper padding.

        Args:
            value: an integer representing the new lower padding value
        """
        if value < HEIGHT - self.__upper_padding:
            self.__lower_padding = value
        else:
            self.__lower_padding = HEIGHT - self.__upper_padding - 1

        self.__update_display()

    def on_mountain_intersection_changed(self, value):
        """
        Updates the mountain intersection value and updates the display

        Args:
            value (float): The new value of the mountain intersection.
        """
        self.__mountain_intersection = value
        self.__update_display()

    def on_smooth_changed(self, value):
        """
        Updates the smooth value, smooths the mountains with the new smooth
        value, and updates the display.

        Args:
            value (float): The new value of the smoothness slider.
        """
        self.__smooth = value
        self.__smoothed_mountains = smooth_mountains(
            self.__mountains, self.__smooth
        )
        self.__update_display()

    def on_color_palette_changed(self, value):
        """
        Updates the color palette used for generating the landscape based on
        the selected option from the dropdown menu.

        Args:
            value (int): The index of the selected color palette.
        """
        self.__currentPaletteIndex = value
        self.__color_palette = list(COLOR_PALETTES.keys())[
            self.__currentPaletteIndex
        ]
        self.__sky_color = COLOR_PALETTES[self.__color_palette]["sky"]
        background = (
            self.__sky_color[2],
            self.__sky_color[1],
            self.__sky_color[0],
        )
        self.__sky_color_button.setStyleSheet(STYLE.format(background))
        self.__sun_color = COLOR_PALETTES[self.__color_palette]["sun"]
        background = (
            self.__sun_color[2],
            self.__sun_color[1],
            self.__sun_color[0],
        )
        self.__sun_color_button.setStyleSheet(STYLE.format(background))
        self.__gradient_color = COLOR_PALETTES[self.__color_palette]["land"][0]
        background = (
            self.__gradient_color[2],
            self.__gradient_color[1],
            self.__gradient_color[0],
        )
        self.__gradient_color_button.setStyleSheet(STYLE.format(background))
        self.__land_color = COLOR_PALETTES[self.__color_palette]["land"]
        self.__update_display()

    def on_sky_color_button_clicked(self):
        """
        Opens a color dialog to let the user select a new color for the sky.
        Updates the sky color attribute with the selected color and updates the
        display.
        """
        selected_color = QtWidgets.QColorDialog().getColor().getRgb()
        self.__sky_color_button.setStyleSheet(STYLE.format(selected_color))
        self.__sky_color = (
            selected_color[2],
            selected_color[1],
            selected_color[0],
            255,
        )
        self.__update_display()

    def on_sun_color_button_clicked(self):
        """
        Opens a color dialog to let the user select a new color for the sun.
        Updates the sun color attribute with the selected color and updates the
        display.
        """
        selected_color = QtWidgets.QColorDialog().getColor().getRgb()
        self.__sun_color_button.setStyleSheet(STYLE.format(selected_color))
        self.__sun_color = (
            selected_color[2],
            selected_color[1],
            selected_color[0],
            255,
        )
        self.__update_display()

    def on_gradient_color_button_clicked(self):
        """
        Opens a color picker dialog to allow the user to select a new gradient
        color.
        Updates the gradient color and land color with the new color and
        pdates the display.
        """
        selected_color = QtWidgets.QColorDialog().getColor().getRgb()
        self.__gradient_color_button.setStyleSheet(
            STYLE.format(selected_color)
        )
        self.__gradient_color = (
            selected_color[2],
            selected_color[1],
            selected_color[0],
            255,
        )
        self.__land_color = [self.__gradient_color]
        self.__update_display()

    def on_reset_palette_button_clicked(self):
        """
        Resets the color palette to its default settings and updates the
        display.
        """
        self.__sky_color = COLOR_PALETTES[self.__color_palette]["sky"]
        background = (
            self.__sky_color[2],
            self.__sky_color[1],
            self.__sky_color[0],
        )
        self.__sky_color_button.setStyleSheet(STYLE.format(background))
        self.__sun_color = COLOR_PALETTES[self.__color_palette]["sun"]
        background = (
            self.__sun_color[2],
            self.__sun_color[1],
            self.__sun_color[0],
        )
        self.__sun_color_button.setStyleSheet(STYLE.format(background))
        self.__gradient_color = COLOR_PALETTES[self.__color_palette]["land"][0]
        background = (
            self.__gradient_color[2],
            self.__gradient_color[1],
            self.__gradient_color[0],
        )
        self.__gradient_color_button.setStyleSheet(STYLE.format(background))
        self.__land_color = COLOR_PALETTES[self.__color_palette]["land"]
        self.__update_display()

    def on_white_contour_changed(self, value):
        """
        Updates the presence of a white contour around the mountains and
        triggers an update of the display.

        Args:
            value (bool): Whether or not to include a white contour around the
            mountains.
        """
        self.__white_contour = value
        self.__update_display()

    def on_margin_changed(self, value):
        """
        Updates the margin size around the mountains and triggers an update of
        the display.

        Args:
            value (int): The index of the selected margin size option.
        """
        self.__currentMarginIndex = value
        self.__margin = MARGIN_OPTIONS[self.__currentMarginIndex]
        self.__update_display()

    def on_save_image_button_clicked(self, value):
        """
        Saves the generated landscape image with the chosen file name and
        format.

        Args:
            value (str): The chosen file name and format for the saved image.
        """
        image = self.__image
        resized = cv2.resize(
            image, (4960, 7016), interpolation=cv2.INTER_LINEAR
        )
        resized = apply_texture(resized, TEX, 0.5)
        cv2.imwrite(self.__image_name_edit.text(), resized)

    def __update_display(self):
        """
        Updates the display with the latest configuration.
        This function generates an image using the current configuration
        parameters and displays it in the GUI.
        """

        # Generate Image
        self.__image = generate_image(WIDTH, HEIGHT, self.__sky_color)

        # Draw sun
        draw_sun(
            self.__image,
            self.__sun_radius,
            self.__center_x,
            self.__center_y,
            self.__sun_color,
            self.__white_contour,
            self.__sky_element,
        )

        # Get mountains based on smooth flag
        mountains = (
            self.__smoothed_mountains if self.__smooth else self.__mountains
        )

        # Normalize mountains based on padding and intersection
        normalize_mountains(
            mountains,
            HEIGHT,
            self.__lower_padding,
            self.__upper_padding,
            self.__mountain_intersection,
        )

        # Draw mountains
        draw_mountains(
            self.__image,
            mountains,
            WIDTH,
            HEIGHT,
            self.__land_color,
            self.__sky_color,
            self.__white_contour,
        )

        # Draw margin if specified
        if not self.__margin == "None":
            self.__image = draw_margin(
                self.__image, self.__margin, WIDTH, HEIGHT
            )

        # Resize and apply texture to image
        resized = cv2.resize(
            self.__image, (496, 702), interpolation=cv2.INTER_LINEAR
        )
        resized = apply_texture(resized, TEX_LOW, 0.5)

        # Convert image to QImage and set it as pixmap for display
        qImage = QtGui.QImage(
            resized.data,
            resized.shape[1],
            resized.shape[0],
            QtGui.QImage.Format_ARGB32,
        )
        self.__image_frame.setPixmap(QtGui.QPixmap.fromImage(qImage))
        self.__image_frame.repaint()

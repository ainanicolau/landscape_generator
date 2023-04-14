# Minimalist Landscape Generator Tool

The Minimalist Landscape Generator Tool a graphical user interface for generating landscape images with customizable settings. The tool provides controls for adjusting the appearance of the sky, sun, mountains, and other elements of the landscape, as well as saving the generated images. The main window displays a preview of the landscape image with the current settings.

The landscape is generated using the Midpoint Displacement algorithm, which produces natural-looking terrain features with a range of parameters. The tool also allows for smoothing and normalization of the mountains, as well as the application of textures and margins to the final image.

With this tool, users can easily create custom landscapes for a variety of purposes, such as backgrounds for websites, desktop wallpapers, or artistic projects. The tool is written in Python and uses the Qt framework for the user interface.

## Getting Started

To use this tool, you will need to have Python 3 installed on your system. You can download the latest version of Python from the official website: https://www.python.org/downloads/

Once you have Python installed, you can clone this repository to your local machine:

```bash
git clone https://github.com/ainanicolau/landscape_generator.git
```
Next, navigate to the project directory and install the required dependencies using pip:

```bash
cd landscape_generator
pip3 install -r requirements.txt
```
You can now run the GUI by running the `run.py` script:

```bash
python3 run.py
```
This will launch the GUI and allow you to start generating landscape images with the provided settings.

## Usage

The GUI provides a range of customizable parameters for generating landscape images. The parameters are grouped into the following sections:

#### Sky Element
- Sky Shape: the shape of the sky element. Select from options of "Sun" or "Moon".
- Sun Radius: The radius of the sun in the image.
- Center X: The x-coordinate of the center of the sky element.
- Center Y: The y-coordinate of the center of the sky element.

#### Mountains
- Layers: the number of layers of mountains.
- Roughness: the roughness of the mountain terrain.
- Decrease Roughness: a checkbox to decrease roughness with each additional layer.
- Padding: the padding between the mountains and the top and bottom of the image.
- Intersections: the amount of intersection between the mountain layers.
- Smoothness: the smoothness of the mountains.

#### Colors
- Palette: A dropdown of preselected colors for the different elements of the landscape.
- Reset palette: Changes back the colors to the selected palette originals.
- Sky: Color of the sky or background.
- Sun: Color of the sky element.
- Land: Color of the mountains.

#### Details
- White Contour: Toggles a white contour around the mountains.
- Margin: Adds a frame to the image. It can be a regular window or a circle.

#### Save
- Image Name: The name to use when saving the image. 
- Save: Button to save the generated landscape image.

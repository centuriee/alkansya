ALKANSYA - a simple Philippine Peso coin counter
CMSC 191 Final Project

This OpenCV program detects Philippine coins and calculates the total denomination of the coins present on frame through size and color analysis.

The application allows both image upload counting and live feed counting.

This program applies principles of:
- Python + OpenCV image handling
- image manipulation and processing
- image segmentation and contour detection
- object detection
- object tracking and motion analysis

Resources used include the following:
- Lab 4 (for reviewing contours)
- Lab 5 (for image processing)
- Lab 6 (for printing text on frame)
- Lab 8 Ball Tracking (for object tracking)
- This YouTube Video -> https://youtu.be/-iN7NDbDz3Q
Their code applies only for static camera view.
My implementation involves classifying the coin's
denomination by identifying its size based on a
reference (Coca-Cola bottle cap, red color)

"""Open an image from the testImages folder in Napari.

Run it from the top folder of the workshop repository with:

    uv run Session2_BitsAndBytes/show_tif.py Cell_cropped_2.tif

The name after the script is the image file to open from the testImages
folder.  Close the Napari window to end the script.
"""

import sys
from pathlib import Path

import napari
from skimage import io

# Where the images live.  Path(__file__) is this script, ".parent.parent" walks
# up two folders to the top of the workshop repository.
REPO = Path(__file__).resolve().parent.parent
IMAGE_DIR = REPO / "testImages"

# sys.argv is the list of words typed on the command line: sys.argv[1] is the
# image name you typed after the script.
if len(sys.argv) > 1:
    IMAGE = IMAGE_DIR / sys.argv[1]
else:
    IMAGE = IMAGE_DIR / "Cell_cropped_2.tif"

pixels = io.imread(IMAGE)
print(f"Loaded {IMAGE.name}: shape {pixels.shape}, number type {pixels.dtype}")

# A color image has 3 numbers (red, green, blue) per pixel, so its last
# dimension is 3.  In that case tell Napari to combine them into one color
# picture ("rgb=True"); otherwise show it as a gray image.
is_rgb = pixels.ndim == 3 and pixels.shape[-1] == 3

viewer = napari.Viewer()
viewer.add_image(pixels, name=IMAGE.stem, rgb=is_rgb)

# Start the Napari window and wait until you close it.
napari.run()

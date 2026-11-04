"""Look inside an image: the numbers behind the picture.

Run it from the top folder of the workshop repository with:

    uv run Session2_BitsAndBytes/inspect_nucleus.py

The script prints a few facts about the image in the terminal, then opens it
in Napari.  Close the Napari window to end the script.
"""

from pathlib import Path

import napari
import numpy as np
from skimage import io

# Where the image lives.  Path(__file__) is this script, ".parent.parent" walks
# up two folders to the top of the workshop repository.
REPO = Path(__file__).resolve().parent.parent
IMAGE = REPO / "testImages" / "nucleus_cropped_1.tif"

# Read the image.  To Python, an image is simply a grid (a "matrix") of
# numbers, one number per pixel.  This grid is stored as a NumPy array.
pixels = io.imread(IMAGE)

# Show the top-left corner of the grid: the first 10 rows and 10 columns.
# "[:10, :10]" means "rows 0 up to 10, columns 0 up to 10".
# Each number is the brightness of one pixel.
np.set_printoptions(linewidth=200)
print(f"Loaded {IMAGE.name}\n")
print("First 10 rows and 10 columns (pixel brightness values):")
print(pixels[:10, :10])

# "shape" gives the size of the grid as (number of rows, number of columns).
rows, columns = pixels.shape
print(f"\nNumber of rows   : {rows}")
print(f"Number of columns: {columns}")

# "dtype" (data type) says what kind of number is stored in each pixel.
# For example, uint16 means "unsigned 16-bit integer": whole numbers that
# cannot be negative, ranging from 0 to 2**16 - 1 = 65535.
print(f"Number type      : {pixels.dtype}")
if np.issubdtype(pixels.dtype, np.integer):
    info = np.iinfo(pixels.dtype)
    print(f"  (this type can hold whole numbers from {info.min} to {info.max})")

# Show the same grid of numbers as a picture in Napari.
viewer = napari.Viewer()
viewer.add_image(pixels, name=IMAGE.stem)

# Start the Napari window and wait until you close it.
napari.run()

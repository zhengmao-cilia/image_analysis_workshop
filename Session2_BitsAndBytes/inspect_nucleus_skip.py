"""Look inside an image, sampling only some of its rows and columns.

Run it from the top folder of the workshop repository with:

    uv run Session2_BitsAndBytes/inspect_nucleus_skip.py nucleus_cropped_2.tif

The name after the script is the image file to open from the testImages
folder.  Leave it out to open nucleus_cropped_1.tif.

The script prints part of the image's number grid in the terminal, then opens
the image in Napari.  Close the Napari window to end the script.
"""

import sys
from pathlib import Path

import napari
import numpy as np
from skimage import io

# Where the images live.  Path(__file__) is this script, ".parent.parent" walks
# up two folders to the top of the workshop repository.
REPO = Path(__file__).resolve().parent.parent
IMAGE_DIR = REPO / "testImages"

# sys.argv is the list of words typed on the command line: sys.argv[0] is the
# script itself, sys.argv[1] (if present) is the image name you typed after it.
if len(sys.argv) > 1:
    IMAGE = IMAGE_DIR / sys.argv[1]
else:
    IMAGE = IMAGE_DIR / "nucleus_cropped_1.tif"

# How many rows and columns to show, and how far to jump each time.
# STEP = 3 means: keep one row, skip the next 2, keep one, skip 2, and so on.
# (Change STEP to 2 to skip only one row and column at a time.)
COUNT = 10
STEP = 3

# Read the image.  To Python, an image is a grid (a "matrix") of numbers, one
# number per pixel, stored as a NumPy array.
pixels = io.imread(IMAGE)

# "[start:stop:step]" picks rows (before the comma) and columns (after it).
# "::STEP" means "from the beginning, every STEP-th one", so rows 0, 3, 6, ...
# Then "[:COUNT, :COUNT]" keeps the first 10 of those rows and columns.
sampled = pixels[::STEP, ::STEP][:COUNT, :COUNT]

# Which original row and column numbers we picked, so you can find them in
# Napari.  Python counts from 0, so the top-left pixel is row 0, column 0.
picked_rows = np.arange(0, pixels.shape[0], STEP)[:COUNT]
picked_columns = np.arange(0, pixels.shape[1], STEP)[:COUNT]

np.set_printoptions(linewidth=200)
print(f"Loaded {IMAGE.name}  ({pixels.shape[0]} rows x {pixels.shape[1]} columns)\n")
print(f"{COUNT} rows and {COUNT} columns, skipping {STEP - 1} each time:")
print(f"  rows used   : {picked_rows.tolist()}")
print(f"  columns used: {picked_columns.tolist()}\n")
print(sampled)

# "dtype" (data type) says what kind of number is stored in each pixel.
# uint16 means "unsigned 16-bit integer": whole numbers from 0 to 65535.
print(f"\nNumber type: {pixels.dtype}")
if np.issubdtype(pixels.dtype, np.integer):
    info = np.iinfo(pixels.dtype)
    print(f"  (this type can hold whole numbers from {info.min} to {info.max})")

# The darkest and brightest pixel actually present in the whole image.
print(f"Values in this image: {pixels.min()} to {pixels.max()}")

# Show the full image as a picture in Napari.
viewer = napari.Viewer()
viewer.add_image(pixels, name=IMAGE.stem)

# Start the Napari window and wait until you close it.
napari.run()

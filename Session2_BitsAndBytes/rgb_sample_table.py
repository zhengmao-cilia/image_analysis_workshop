"""Print the red, green and blue values of a regular sample of pixels.

Run it from the top folder of the workshop repository with:

    uv run Session2_BitsAndBytes/rgb_sample_table.py

The script prints a table in the terminal, then opens the image in Napari.
Close the Napari window to end the script.
"""

from pathlib import Path

import napari
from skimage import io

# Where the image lives.  Path(__file__) is this script, ".parent.parent" walks
# up two folders to the top of the workshop repository.
REPO = Path(__file__).resolve().parent.parent
IMAGE = REPO / "testImages" / "Cell_cropped_2.tif"

# What fraction of the pixels to show: 0.03 means 3%.
FRACTION = 0.03

# Read the image.  A color (RGB) image is a stack of three grids of numbers,
# one each for red, green and blue.  Its shape is (rows, columns, 3).
pixels = io.imread(IMAGE)
rows, columns, colors = pixels.shape
total = rows * columns

# To show 3% of the pixels, keep one pixel out of every 1 / 0.03 = 33.
# We walk through the pixels the way you read a book: left to right along a
# row, then on to the next row, and keep every 33rd pixel we pass.
step = round(1 / FRACTION)
positions = range(0, total, step)

print(f"Loaded {IMAGE.name}: {rows} rows x {columns} columns, "
      f"{colors} color channels, number type {pixels.dtype}")
print(f"Showing 1 pixel in every {step}: {len(positions)} of {total} pixels "
      f"({100 * len(positions) / total:.1f}%)\n")

# The table header.  The ":>6" parts make each column 6 characters wide and
# right-aligned, so the numbers line up.
print(f"{'row':>6} {'column':>6} {'R':>6} {'G':>6} {'B':>6}")
print("-" * 34)

for position in positions:
    # Turn "the Nth pixel" back into a row and column number.  divmod divides
    # and gives both the whole result (the row) and the remainder (the column).
    row, column = divmod(position, columns)
    red, green, blue = pixels[row, column]
    print(f"{row:>6} {column:>6} {red:>6} {green:>6} {blue:>6}")

# Show the image in Napari.  "rgb=True" tells Napari that the last dimension
# holds red, green and blue, so it shows one color picture rather than three
# separate gray images.
viewer = napari.Viewer()
viewer.add_image(pixels, name=IMAGE.stem, rgb=True)

# Start the Napari window and wait until you close it.
napari.run()

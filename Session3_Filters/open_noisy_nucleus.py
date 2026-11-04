"""Open the noisy nucleus image in Napari, without any filtering.

Run this script from the top folder of the workshop repository with:

    uv run Session3_Filters/open_noisy_nucleus.py

Close the Napari window to end the script.
"""

from pathlib import Path

import napari
from skimage import io

# Where the image lives.  Path(__file__) is this script; ".parent.parent"
# walks up two folders to the top of the workshop repository.
REPO = Path(__file__).resolve().parent.parent
IMAGE = REPO / "testImages" / "Nucleus_cropped_noise.tif"

# Read the pixels into memory.  The result is a grid (array) of numbers,
# one number per pixel.  Nothing is changed - this is the raw image.
image = io.imread(IMAGE)

# Print a few facts so you know what you are looking at.
print(f"Shape (size in pixels): {image.shape}")
print(f"Number type: {image.dtype}")
print(f"Darkest pixel: {image.min()}, brightest pixel: {image.max()}")

# Show the image in Napari as one layer called "noisy (original)".
viewer = napari.Viewer()
viewer.add_image(image, name="noisy (original)")

# Start the Napari window and wait until you close it.
napari.run()

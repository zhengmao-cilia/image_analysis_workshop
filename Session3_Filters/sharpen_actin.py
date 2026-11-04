"""Apply the ImageJ "Sharpen" filter to the actin image and compare in Napari.

Run this script from the top folder of the workshop repository with:

    uv run Session3_Filters/sharpen_actin.py

The sharpened image is only kept in memory - nothing is written to disk,
so the original file is never changed.  Close the Napari window to end.
"""

from pathlib import Path

import napari
import numpy as np
from scipy import ndimage
from skimage import io

REPO = Path(__file__).resolve().parent.parent
IMAGE = REPO / "testImages" / "Actin_cropped.tif"

original = io.imread(IMAGE)
print(f"Shape: {original.shape}, number type: {original.dtype}")

# The ImageJ Sharpen kernel.  Each new pixel = 12 x (the pixel itself)
# minus each of its 8 neighbours.  Where a pixel is brighter than its
# surroundings (an edge or fibre), that difference gets amplified.
kernel = np.array([[-1, -1, -1],
                   [-1, 12, -1],
                   [-1, -1, -1]], dtype=float)

# The numbers add up to 12 - 8 = 4.  Like ImageJ, we divide by that sum so
# a flat (uniform) area keeps its original brightness.
kernel = kernel / kernel.sum()

# Convolution: slide the kernel over every pixel and compute the weighted
# sum.  We work with decimal numbers (float) so negative values in the
# middle of the calculation are not lost.  mode="nearest" repeats the edge
# pixels at the image border, which is what ImageJ does.
sharpened = ndimage.convolve(original.astype(float), kernel, mode="nearest")

# Sharpening can push values below 0 or above the maximum that a 16-bit
# image can store (65535).  Like ImageJ, clip them back into that range and
# convert to the same number type as the original.
limits = np.iinfo(original.dtype)
sharpened = np.clip(np.round(sharpened), limits.min, limits.max).astype(original.dtype)

print(f"Original  min/max: {original.min()} / {original.max()}")
print(f"Sharpened min/max: {sharpened.min()} / {sharpened.max()}")

# Show both images.  Use the same contrast limits for both so the
# comparison is fair (otherwise Napari auto-scales each one differently).
viewer = napari.Viewer()
contrast = (float(original.min()), float(original.max()))
viewer.add_image(original, name="original", contrast_limits=contrast)
viewer.add_image(sharpened, name="sharpened (ImageJ kernel)", contrast_limits=contrast)

# Show the two layers side by side instead of on top of each other.
viewer.grid.enabled = True

napari.run()

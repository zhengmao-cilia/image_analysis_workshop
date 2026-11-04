"""Sobel edge detection on the actin image, compared in Napari.

Run this script from the top folder of the workshop repository with:

    uv run Session3_Filters/sobel_actin.py

Sobel x, Sobel y and the gradient magnitude are saved as NEW files in
Session3_Filters/results/.  The original image is only read, never written.
Close the Napari window to end.
"""

from pathlib import Path

import napari
import numpy as np
from scipy import ndimage
from skimage import io

REPO = Path(__file__).resolve().parent.parent
IMAGE = REPO / "testImages" / "Actin_cropped.tif"
OUT_DIR = REPO / "Session3_Filters" / "results"
OUT_DIR.mkdir(exist_ok=True)

original = io.imread(IMAGE)
image = original.astype(float)  # decimals, because results can be negative

# Sobel x: right column minus left column.  Responds where brightness
# changes from left to right, i.e. to VERTICAL edges.  The middle row
# counts double, which also smooths a little along the edge.
sobel_x_kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=float)

# Sobel y: the same kernel turned 90 degrees (bottom row minus top row).
# Responds where brightness changes from top to bottom: HORIZONTAL edges.
sobel_y_kernel = sobel_x_kernel.T

# Slide each kernel over the image ("correlate" uses the kernel exactly as
# written above).  mode="nearest" repeats edge pixels at the border.
sobel_x = ndimage.correlate(image, sobel_x_kernel, mode="nearest")
sobel_y = ndimage.correlate(image, sobel_y_kernel, mode="nearest")

# Gradient magnitude: how steep the brightness change is in ANY direction.
# Pythagoras: sqrt(x^2 + y^2).  It is always >= 0 and is bright on all
# edges, whatever their direction.
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

# Save as 32-bit decimal TIFFs.  Sobel x and y contain negative numbers
# (the sign tells you dark->bright or bright->dark), and all three can be
# larger than 65535, so a 16-bit file would lose information.
results = {"sobel_x": sobel_x, "sobel_y": sobel_y, "gradient_magnitude": magnitude}
for name, data in results.items():
    out_file = OUT_DIR / f"Actin_cropped_{name}.tif"
    io.imsave(out_file, data.astype(np.float32), check_contrast=False)
    print(f"Saved {out_file.relative_to(REPO)}  "
          f"(min {data.min():.0f}, max {data.max():.0f})")

viewer = napari.Viewer()
viewer.add_image(original, name="original")

# Sobel x / y: 0 = no edge, so show them with a symmetric contrast range and
# a colour map where negative and positive look different (grey middle).
for name, data in [("Sobel x", sobel_x), ("Sobel y", sobel_y)]:
    limit = float(np.abs(data).max())
    viewer.add_image(data, name=name, contrast_limits=(-limit, limit))

viewer.add_image(magnitude, name="gradient magnitude")

viewer.grid.enabled = True

napari.run()

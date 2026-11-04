"""Compare smoothing (averaging) kernels with median filters in Napari.

Run this script from the top folder of the workshop repository with:

    uv run Session3_Filters/smoothing_vs_median_compare.py

All results are kept in memory only - nothing is written to disk, so the
original file is never changed.  Close the Napari window to end.
"""

from pathlib import Path

import napari
import numpy as np
from scipy import ndimage
from skimage import io

REPO = Path(__file__).resolve().parent.parent
IMAGE = REPO / "testImages" / "Nucleus_cropped_noise.tif"

original = io.imread(IMAGE)

# Work with decimal numbers (float).  Averaging gives values like 1234.56;
# in the original whole-number (uint16) format the decimals would be cut off.
original_f = original.astype(np.float32)

# ---------------------------------------------------------------------------
# 1. Smoothing kernels.  Each number is a weight: how much that neighbour
#    counts.  Dividing by the sum ("normalizing") makes the weights add up
#    to 1, so the result is a weighted average and the overall brightness of
#    the image stays the same.
# ---------------------------------------------------------------------------
box_3x3 = np.ones((3, 3))

center_weighted_3x3 = np.ones((3, 3))
center_weighted_3x3[1, 1] = 4          # the centre pixel counts 4 times

box_7x7 = np.ones((7, 7))

kernels = {
    "mean 3x3 (all 1)": box_3x3,
    "mean 3x3 (centre 4)": center_weighted_3x3,
    "mean 7x7 (all 1)": box_7x7,
}

smoothed = {}
for name, kernel in kernels.items():
    kernel = kernel / kernel.sum()      # normalize: weights now sum to 1
    print(f"{name}: kernel sum before normalizing = {kernels[name].sum():.0f}")
    # convolve slides the kernel over the image; each output pixel is the
    # weighted sum of the pixels under the kernel.
    smoothed[name] = ndimage.convolve(original_f, kernel, mode="reflect")

# ---------------------------------------------------------------------------
# 2. Median filters (same as median_filters_compare.py), for comparison.
# ---------------------------------------------------------------------------
cross_3x3 = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]], dtype=bool)
footprints = {
    "median 3x3 cross (5 px)": cross_3x3,
    "median 3x3 full (9 px)": np.ones((3, 3), dtype=bool),
    "median 5x5 full (25 px)": np.ones((5, 5), dtype=bool),
}
medians = {
    name: ndimage.median_filter(original_f, footprint=fp, mode="reflect")
    for name, fp in footprints.items()
}

# ---------------------------------------------------------------------------
# 3. Show everything side by side with identical contrast settings, so the
#    differences you see come from the filters, not from the display.
# ---------------------------------------------------------------------------
limits = (float(original.min()), float(original.max()))

viewer = napari.Viewer()
viewer.add_image(original_f, name="original (noisy)", contrast_limits=limits)
for name, image in {**smoothed, **medians}.items():
    viewer.add_image(image, name=name, contrast_limits=limits)

viewer.grid.enabled = True   # tile all layers instead of stacking them

napari.run()

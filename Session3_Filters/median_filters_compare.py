"""Compare three median filters on the noisy nucleus image in Napari.

Run this script from the top folder of the workshop repository with:

    uv run Session3_Filters/median_filters_compare.py

The filtered images are only kept in memory - nothing is written to disk,
so the original file is never changed.  Close the Napari window to end.
"""

from pathlib import Path

import napari
import numpy as np
from scipy import ndimage
from skimage import io

REPO = Path(__file__).resolve().parent.parent
IMAGE = REPO / "testImages" / "Nucleus_cropped_noise.tif"

original = io.imread(IMAGE)

# A "footprint" (kernel) says which neighbours take part in the median.
# 1 = this pixel is included, 0 = it is ignored.  The centre is the pixel
# being replaced.
cross_3x3 = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]], dtype=bool)   # 3x3 without the 4 corners
full_3x3 = np.ones((3, 3), dtype=bool)          # all 9 pixels
full_5x5 = np.ones((5, 5), dtype=bool)          # all 25 pixels

# The median filter replaces every pixel by the middle value (median) of the
# pixels under the footprint.  A single very bright or dark noise pixel is
# an outlier, so it rarely ends up as the median - that is why the median
# removes "salt and pepper" style noise while keeping edges fairly sharp.
# mode="reflect" decides how to handle pixels at the image border (it
# mirrors the image outwards).
filtered = {
    "median 3x3 cross (5 px)": ndimage.median_filter(original, footprint=cross_3x3, mode="reflect"),
    "median 3x3 full (9 px)": ndimage.median_filter(original, footprint=full_3x3, mode="reflect"),
    "median 5x5 full (25 px)": ndimage.median_filter(original, footprint=full_5x5, mode="reflect"),
}

# Use the same brightness/contrast setting for every layer, so differences
# you see come from the filters and not from different display settings.
limits = (float(original.min()), float(original.max()))

viewer = napari.Viewer()
viewer.add_image(original, name="original (noisy)", contrast_limits=limits)
for name, image in filtered.items():
    viewer.add_image(image, name=name, contrast_limits=limits)
    print(f"{name}: min {image.min()}, max {image.max()}")

# Grid mode shows all layers side by side instead of stacked on top of
# each other.  Zooming or panning one view moves all of them together.
viewer.grid.enabled = True

napari.run()

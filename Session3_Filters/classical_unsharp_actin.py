"""Classical unsharp masking of the actin image, compared in Napari.

Run this script from the top folder of the workshop repository with:

    uv run Session3_Filters/classical_unsharp_actin.py

    sharpened = original + amount * (original - blurred)

The sharpened images are only kept in memory - nothing is written to disk,
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
image = original.astype(float)  # decimal numbers so negatives are not lost

# Same Gaussian kernels as gaussian_unsharp_actin.py:
# kernel size = 2 * radius + 1, and sigma = radius / 2.
kernels = {
    "5x5": {"radius": 2, "sigma": 1.0},
    "21x21": {"radius": 10, "sigma": 5.0},
}

# How much of the detail ("mask") to add back.  0 = no change,
# larger = stronger sharpening (and stronger noise).
amounts = [0.5, 1.0, 2.0]

limits = np.iinfo(original.dtype)  # 0 .. 65535 for a 16-bit image

viewer = napari.Viewer()
contrast = (float(original.min()), float(original.max()))

for size, k in kernels.items():
    blurred = ndimage.gaussian_filter(
        image, sigma=k["sigma"], radius=k["radius"], mode="nearest"
    )

    # The mask: the fine detail that the blur removed.
    mask = image - blurred

    for amount in amounts:
        # Add a fraction of the detail back onto the original.
        sharpened = image + amount * mask

        # Keep values inside what a 16-bit image can store, then convert
        # back to the original number type (like ImageJ does).
        sharpened = np.clip(np.round(sharpened), limits.min, limits.max)
        sharpened = sharpened.astype(original.dtype)

        print(f"{size}, amount {amount}: min/max "
              f"{sharpened.min()} / {sharpened.max()}")

        # Same contrast as the original so the comparison is fair.
        viewer.add_image(sharpened, name=f"unsharp {size}, amount {amount}",
                         contrast_limits=contrast)

# Add the original last so it appears first (top-left) in the grid.
viewer.add_image(original, name="original", contrast_limits=contrast)
print(f"Original min/max: {original.min()} / {original.max()}")

# Show all layers side by side.  Grid order follows the layer list from
# the top, so the original comes first, then 21x21, then 5x5.
viewer.grid.enabled = True

napari.run()

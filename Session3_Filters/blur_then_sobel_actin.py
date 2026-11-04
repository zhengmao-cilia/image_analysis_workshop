"""Sobel gradient magnitude with and without a 5x5 Gaussian blur first.

Run this script from the top folder of the workshop repository with:

    uv run Session3_Filters/blur_then_sobel_actin.py

Nothing is written to disk, so the original file is never changed.
Close the Napari window to end.
"""

from pathlib import Path

import napari
import numpy as np
from scipy import ndimage
from skimage import io

REPO = Path(__file__).resolve().parent.parent
IMAGE = REPO / "testImages" / "Actin_cropped.tif"

original = io.imread(IMAGE)
image = original.astype(float)

# Same Sobel kernels as sobel_actin.py.
sobel_x_kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=float)
sobel_y_kernel = sobel_x_kernel.T


def gradient_magnitude(img):
    """Apply Sobel x and Sobel y, then combine: sqrt(x^2 + y^2)."""
    gx = ndimage.correlate(img, sobel_x_kernel, mode="nearest")
    gy = ndimage.correlate(img, sobel_y_kernel, mode="nearest")
    return np.sqrt(gx**2 + gy**2)


# Same 5x5 Gaussian as gaussian_unsharp_actin.py (radius 2, sigma 1).
blurred = ndimage.gaussian_filter(image, sigma=1.0, radius=2, mode="nearest")

# Sobel is a "difference" filter, so it amplifies pixel-to-pixel noise.
# Blurring first averages that noise away, leaving mainly the real edges.
mag_raw = gradient_magnitude(image)
mag_blurred = gradient_magnitude(blurred)

for name, data in [("without blur", mag_raw), ("with 5x5 blur", mag_blurred)]:
    print(f"Gradient magnitude {name}: min {data.min():.0f}, "
          f"max {data.max():.0f}, mean {data.mean():.0f}")

viewer = napari.Viewer()
viewer.add_image(original, name="original")
viewer.add_image(mag_raw, name="gradient magnitude (no blur)")
viewer.add_image(mag_blurred, name="gradient magnitude (5x5 blur first)")

viewer.grid.enabled = True

napari.run()

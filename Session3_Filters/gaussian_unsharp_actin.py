"""Gaussian blur (5x5 and 21x21) and unsharp masks of the actin image.

Run this script from the top folder of the workshop repository with:

    uv run Session3_Filters/gaussian_unsharp_actin.py

The two blurred images are saved as NEW files in Session3_Filters/results/.
The original image is only read, never written, so it cannot be changed.
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
print(f"Shape: {original.shape}, number type: {original.dtype}")

# A Gaussian kernel is a small grid of weights shaped like a bell curve:
# the centre pixel counts most, pixels further away count less.  Blurring
# replaces every pixel with this weighted average of its neighbours.
#
# Two numbers describe the kernel:
#   radius - how many pixels the kernel reaches out from the centre.
#            Kernel size = 2 * radius + 1, so radius 2 -> 5x5, 10 -> 21x21.
#   sigma  - how wide the bell curve is.  Here sigma = radius / 2, so the
#            kernel reaches 2 sigma out on each side and both kernels have
#            the same bell shape, just at different scales.
kernels = {
    "5x5": {"radius": 2, "sigma": 1.0},
    "21x21": {"radius": 10, "sigma": 5.0},
}

viewer = napari.Viewer()
contrast = (float(original.min()), float(original.max()))
viewer.add_image(original, name="original", contrast_limits=contrast)

for size, k in kernels.items():
    # Blur.  We work with decimal numbers (float) for accuracy.
    # mode="nearest" repeats the edge pixels at the image border.
    blurred = ndimage.gaussian_filter(
        original.astype(float), sigma=k["sigma"], radius=k["radius"], mode="nearest"
    )

    # Unsharp mask: original minus blurred.  The blur keeps only the slow,
    # large-scale brightness changes, so subtracting it leaves the fine
    # detail (edges, thin fibres) - and also the noise.  The result can be
    # negative, so it stays a decimal (float) image.
    unsharp = original.astype(float) - blurred

    # Save the blurred image.  A blur only averages pixel values, so it never
    # goes outside the original range - we can safely store it with the same
    # 16-bit number type as the original.
    blurred_16bit = np.round(blurred).astype(original.dtype)
    out_file = OUT_DIR / f"Actin_cropped_gaussian_{size}.tif"
    io.imsave(out_file, blurred_16bit, check_contrast=False)
    print(f"Saved {out_file.relative_to(REPO)}")

    print(f"{size}: blurred min/max {blurred.min():.0f} / {blurred.max():.0f}, "
          f"unsharp mask min/max {unsharp.min():.0f} / {unsharp.max():.0f}")

    # Blurred images use the same contrast as the original for a fair look.
    # Unsharp masks are centred on 0, so let Napari pick their contrast.
    viewer.add_image(blurred_16bit, name=f"blurred {size}", contrast_limits=contrast)
    viewer.add_image(unsharp, name=f"unsharp mask {size}")

# Show all five layers side by side.
viewer.grid.enabled = True

napari.run()

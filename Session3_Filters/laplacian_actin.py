"""Laplacian filter on the actin image, compared with the sharpening results.

Run this script from the top folder of the workshop repository with:

    uv run Session3_Filters/laplacian_actin.py

The Laplacian result is saved as a NEW file:
    Session3_Filters/results/Actin_cropped_laplacian.tif
The original image is only read, never written.  The other images (ImageJ
Sharpen, unsharp masks) are recalculated here exactly as in the earlier
scripts and only shown, not saved.  Close the Napari window to end.
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
image = original.astype(float)
limits = np.iinfo(original.dtype)  # 0 .. 65535 for a 16-bit image


def to_16bit(data):
    """Round, clip to the 16-bit range and convert, like ImageJ does."""
    return np.clip(np.round(data), limits.min, limits.max).astype(original.dtype)


# --- 1. Laplacian -----------------------------------------------------------
# 8 x (the pixel) minus its 8 neighbours.  The numbers add up to 0, so flat
# areas become 0 and only places where brightness CURVES (edges, thin
# fibres, noise) give a signal.  Bright, thin fibres give a positive peak
# in the centre with negative values on either side.
laplacian_kernel = np.array([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]], dtype=float)
laplacian = ndimage.correlate(image, laplacian_kernel, mode="nearest")

# Negative values matter, so save as a 32-bit decimal TIFF.
out_file = OUT_DIR / "Actin_cropped_laplacian.tif"
io.imsave(out_file, laplacian.astype(np.float32), check_contrast=False)
print(f"Saved {out_file.relative_to(REPO)}  "
      f"(min {laplacian.min():.0f}, max {laplacian.max():.0f})")

# --- 2. ImageJ Sharpen (as in sharpen_actin.py) ------------------------------
sharpen_kernel = np.array([[-1, -1, -1],
                           [-1, 12, -1],
                           [-1, -1, -1]], dtype=float) / 4
sharpened = to_16bit(ndimage.convolve(image, sharpen_kernel, mode="nearest"))

# Note: the Sharpen kernel = (identity x 4 + Laplacian) / 4, so
#   ImageJ Sharpen = original + Laplacian / 4.
# Check that this is really true for our image:
same = np.array_equal(sharpened, to_16bit(image + laplacian / 4))
print(f"ImageJ Sharpen == original + Laplacian/4 ?  {same}")

# --- 3. Unsharp masks (as in gaussian_unsharp_actin.py and
#        classical_unsharp_actin.py) --------------------------------------------
kernels = {"5x5": {"radius": 2, "sigma": 1.0},
           "21x21": {"radius": 10, "sigma": 5.0}}
amounts = [0.5, 1.0, 2.0]

masks = {}
unsharp = {}
for size, k in kernels.items():
    blurred = ndimage.gaussian_filter(image, sigma=k["sigma"],
                                      radius=k["radius"], mode="nearest")
    masks[size] = image - blurred                      # the detail
    for amount in amounts:
        unsharp[(size, amount)] = to_16bit(image + amount * masks[size])

# --- Show everything ----------------------------------------------------------
viewer = napari.Viewer()
contrast = (float(original.min()), float(original.max()))


def symmetric(data):
    """Contrast range centred on 0, so 0 shows as mid-grey."""
    m = float(np.abs(data).max())
    return (-m, m)


viewer.add_image(original, name="original", contrast_limits=contrast)
viewer.add_image(laplacian, name="Laplacian", contrast_limits=symmetric(laplacian))
viewer.add_image(sharpened, name="ImageJ Sharpen", contrast_limits=contrast)
for size, mask in masks.items():
    viewer.add_image(mask, name=f"mask (original - blur {size})",
                     contrast_limits=symmetric(mask))
for (size, amount), data in unsharp.items():
    viewer.add_image(data, name=f"unsharp {size}, amount {amount}",
                     contrast_limits=contrast)

viewer.grid.enabled = True

napari.run()

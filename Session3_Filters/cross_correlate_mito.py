"""Cross-correlate the mitochondria image with the template (NO normalization).

Cross-correlation slides the small template over every position of the big
image.  At each position it multiplies every template pixel with the image
pixel underneath it and adds up all those products.  The sum is one number,
the "correlation value" for that position.  A high value is supposed to
mean "the image looks like the template here".

Run this script from the top folder of the workshop repository with:

    uv run Session3_Filters/cross_correlate_mito.py

Close the Napari window to end the script.
"""

from pathlib import Path

import napari
import numpy as np
from scipy import signal
from skimage import io

REPO = Path(__file__).resolve().parent.parent
MITO_FILE = REPO / "testImages" / "Mito_Cropped.tif"
TEMPLATE_FILE = REPO / "testImages" / "Mito_template.tif"
RESULT_FILE = Path(__file__).resolve().parent / "results" / "mito_xcorr_raw.tif"

# Read both images.  We turn the pixel numbers into decimal numbers
# ("float64") so that the large sums of products cannot overflow.
mito = io.imread(MITO_FILE).astype(np.float64)
template = io.imread(TEMPLATE_FILE).astype(np.float64)
th, tw = template.shape

# --- 1. Raw cross-correlation ---------------------------------------------
# mode="valid" only uses positions where the template fits completely inside
# the image.  The result is then (284 - 19 + 1) x (288 - 19 + 1) pixels, and
# the value at (row, col) belongs to the template placed with its TOP-LEFT
# corner at (row, col) of the mitochondria image.
xcorr = signal.correlate(mito, template, mode="valid", method="fft")

# Where is the highest value?  argmax finds it in the flattened array;
# unravel_index turns that back into (row, column).
peak_row, peak_col = np.unravel_index(np.argmax(xcorr), xcorr.shape)
print(f"Correlation result size: {xcorr.shape}")
print(f"Highest correlation value: {xcorr.max():.4g} "
      f"at top-left (row {peak_row}, col {peak_col})")

# --- 2. Where was the template really cut from? ---------------------------
# The template was cut out of this image, so somewhere there is a 19 x 19
# patch that is IDENTICAL to it.  We find it by checking, for every
# position, the summed squared difference between patch and template.
# A perfect copy gives exactly 0.
patches = np.lib.stride_tricks.sliding_window_view(mito, (th, tw))
ssd = ((patches - template) ** 2).sum(axis=(2, 3))
true_row, true_col = np.unravel_index(np.argmin(ssd), ssd.shape)
print(f"Template was cut from top-left (row {true_row}, col {true_col}) "
      f"(difference there: {ssd.min():.0f}; 0 means a perfect copy)")
print(f"Correlation value at the true location: "
      f"{xcorr[true_row, true_col]:.4g}")

if (peak_row, peak_col) == (true_row, true_col):
    print("-> The highest correlation IS at the true template location.")
else:
    distance = np.hypot(peak_row - true_row, peak_col - true_col)
    print(f"-> The highest correlation is NOT at the true location "
          f"({distance:.1f} pixels away).")

# --- 3. Save the result ----------------------------------------------------
RESULT_FILE.parent.mkdir(exist_ok=True)
io.imsave(RESULT_FILE, xcorr.astype(np.float32), check_contrast=False)
print(f"Saved correlation image to {RESULT_FILE}")

# --- 4. Show everything in Napari -----------------------------------------
viewer = napari.Viewer()
viewer.add_image(mito, name="mitochondria", colormap="gray")
# Shift the correlation image by half a template so each value sits over
# the CENTER of the template position it describes - then it lines up with
# the mitochondria image underneath.
viewer.add_image(xcorr, name="cross-correlation (raw)", colormap="magma",
                 translate=(th // 2, tw // 2), opacity=0.7)
# Mark the two locations (centers of the template positions).
viewer.add_points([[peak_row + th // 2, peak_col + tw // 2]],
                  name="highest correlation", face_color="red",
                  symbol="cross", size=8)
viewer.add_points([[true_row + th // 2, true_col + tw // 2]],
                  name="true template location", face_color="cyan",
                  symbol="disc", size=8)
viewer.add_shapes([[[true_row, true_col], [true_row + th, true_col + tw]]],
                  shape_type="rectangle", edge_color="cyan",
                  face_color="transparent", edge_width=1,
                  name="true template box")
napari.run()

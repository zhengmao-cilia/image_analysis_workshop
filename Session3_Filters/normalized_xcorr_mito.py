"""Normalized cross-correlation of the mitochondria image with the template.

Plain cross-correlation is fooled by bright areas (see cross_correlate_mito.py).
NORMALIZED cross-correlation fixes that.  At every position it first takes
away the average brightness of the patch and of the template, and divides by
how much each one varies.  What is left only measures how similar the
PATTERN is.  The result is always between -1 and 1:
     1  = patch looks exactly like the template (brighter or dimmer is fine)
     0  = no similarity
    -1  = patch looks like the "negative" of the template

Run this script from the top folder of the workshop repository with:

    uv run Session3_Filters/normalized_xcorr_mito.py

Close the Napari window to end the script.
"""

from pathlib import Path

import napari
import numpy as np
from skimage import io
from skimage.feature import match_template, peak_local_max

REPO = Path(__file__).resolve().parent.parent
MITO_FILE = REPO / "testImages" / "Mito_Cropped.tif"
TEMPLATE_FILE = REPO / "testImages" / "Mito_template.tif"
RESULT_FILE = Path(__file__).resolve().parent / "results" / "mito_xcorr_normalized.tif"

# Settings for finding matches - try changing these!
THRESHOLD = 0.8      # only keep peaks with a score above this
MIN_DISTANCE = 4     # peaks must be at least this many pixels apart

mito = io.imread(MITO_FILE).astype(np.float64)
template = io.imread(TEMPLATE_FILE).astype(np.float64)
th, tw = template.shape

# --- 1. Normalized cross-correlation ---------------------------------------
# match_template does the normalized cross-correlation for us.  Like
# mode="valid" before, the value at (row, col) belongs to the template placed
# with its TOP-LEFT corner at (row, col), so the result is 266 x 270 pixels.
ncc = match_template(mito, template)
print(f"Result size: {ncc.shape}, scores from {ncc.min():.3f} to {ncc.max():.3f}")

# --- 2. Find the peaks ------------------------------------------------------
# peak_local_max finds the "hilltops": pixels higher than all their
# neighbours.  threshold_abs throws away hilltops below 0.8, and min_distance
# makes sure two peaks are at least 4 pixels apart (otherwise one bump could
# be counted twice).  It returns one (row, col) pair per peak.
peaks = peak_local_max(ncc, min_distance=MIN_DISTANCE, threshold_abs=THRESHOLD)
print(f"\nFound {len(peaks)} matches with score > {THRESHOLD} "
      f"and at least {MIN_DISTANCE} pixels apart:")

# Add half the template size to go from the top-left corner to the CENTER
# of each match, which is where we want to draw the circle.
centers = peaks + np.array([th // 2, tw // 2])
for (r, c), (cr, cc) in zip(peaks, centers):
    print(f"  center at (row {cr}, col {cc})   score {ncc[r, c]:.3f}")

# --- 3. Save the result -----------------------------------------------------
RESULT_FILE.parent.mkdir(exist_ok=True)
io.imsave(RESULT_FILE, ncc.astype(np.float32), check_contrast=False)
print(f"\nSaved normalized correlation image to {RESULT_FILE}")

# --- 4. Show in Napari ------------------------------------------------------
viewer = napari.Viewer()
viewer.add_image(mito, name="mitochondria", colormap="gray")
# The score map, shifted so each score sits over the center of its match.
# It starts hidden - click its eye icon to see it.
viewer.add_image(ncc, name="normalized cross-correlation", colormap="magma",
                 translate=(th // 2, tw // 2), contrast_limits=(-1, 1),
                 visible=False)
# One ring per match, about as big as the template.
viewer.add_points(centers, name=f"matches (score > {THRESHOLD})",
                  symbol="ring", size=tw, face_color="yellow",
                  border_width=0)
napari.run()

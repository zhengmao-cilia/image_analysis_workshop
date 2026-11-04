"""Compare normalized cross-correlation peak detection at several thresholds.

The threshold decides how similar a spot must be to the template to count
as a "match".  Here we try 0.6, 0.8 and 0.9 (minimum distance between peaks
stays at 4 pixels) and show the matches for each threshold as its own layer.

Run this script from the top folder of the workshop repository with:

    uv run Session3_Filters/compare_ncc_thresholds.py

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

# Thresholds to compare, each with its own circle color and size.  The
# circles get smaller for higher thresholds so that rings at the same spot
# do not hide each other.
THRESHOLDS = [(0.6, "magenta", 27), (0.8, "yellow", 21), (0.9, "cyan", 15)]
MIN_DISTANCE = 4

mito = io.imread(MITO_FILE).astype(np.float64)
template = io.imread(TEMPLATE_FILE).astype(np.float64)
th, tw = template.shape

# Normalized cross-correlation - done once, then reused for every threshold.
ncc = match_template(mito, template)

viewer = napari.Viewer()
viewer.add_image(mito, name="mitochondria", colormap="gray")
viewer.add_image(ncc, name="normalized cross-correlation", colormap="magma",
                 translate=(th // 2, tw // 2), contrast_limits=(-1, 1),
                 visible=False)

print(f"Minimum distance between peaks: {MIN_DISTANCE} pixels\n")
for threshold, color, size in THRESHOLDS:
    # Same peak finding as before, only the threshold changes.
    peaks = peak_local_max(ncc, min_distance=MIN_DISTANCE,
                           threshold_abs=threshold)
    centers = peaks + np.array([th // 2, tw // 2])  # top-left -> center

    print(f"Threshold {threshold}: {len(peaks)} match(es)")
    for (r, c), (cr, cc) in zip(peaks, centers):
        print(f"    center at (row {cr}, col {cc})   score {ncc[r, c]:.3f}")

    viewer.add_points(centers, name=f"threshold {threshold} ({len(peaks)})",
                      symbol="ring", size=size, face_color=color,
                      border_width=0)

napari.run()

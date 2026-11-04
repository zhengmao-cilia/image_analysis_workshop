"""Find how far an image was moved, using cross-correlation.

Step 1: make a shifted copy of Mito_Cropped.tif (7 pixels down, 4 right).
Step 2: cross-correlate the original with the shifted copy WITHOUT telling
        the computer the shift, and read the shift from the correlation peak.

Run this script from the top folder of the workshop repository with:

    uv run Session3_Filters/detect_shift_xcorr.py
"""

from pathlib import Path

import numpy as np
from scipy import ndimage, signal
from skimage import io
from skimage.registration import phase_cross_correlation

REPO = Path(__file__).resolve().parent.parent
MITO_FILE = REPO / "testImages" / "Mito_Cropped.tif"
RESULT_DIR = Path(__file__).resolve().parent / "results"

original = io.imread(MITO_FILE).astype(np.float64)

# --- 1. Make the shifted copy ----------------------------------------------
# ndimage.shift moves the image by (rows, columns).  Positive rows = down,
# positive columns = right.  order=0 moves whole pixels without blurring;
# the empty strip that appears at the top and left is filled with 0 (black).
SHIFT_DOWN, SHIFT_RIGHT = 7, 4
shifted = ndimage.shift(original, (SHIFT_DOWN, SHIFT_RIGHT), order=0, cval=0)
RESULT_DIR.mkdir(exist_ok=True)
io.imsave(RESULT_DIR / "Mito_shifted_7down_4right.tif",
          shifted.astype(np.uint16), check_contrast=False)

# Everything below only uses `original` and `shifted` - the numbers 7 and 4
# are NOT given to the shift detection.

# --- 2. Cross-correlation ---------------------------------------------------
# Subtract each image's average brightness first.  Otherwise the score is
# mostly "how much bright area overlaps", which always favours zero shift.
a = original - original.mean()
b = shifted - shifted.mean()

# mode="full" tries EVERY possible offset between the two images, including
# ones where they only overlap a little.  The result has (2*H-1) x (2*W-1)
# values; the center element, at (H-1, W-1), is "no shift at all".
xcorr = signal.correlate(b, a, mode="full", method="fft")
peak = np.unravel_index(np.argmax(xcorr), xcorr.shape)
H, W = original.shape
dy = peak[0] - (H - 1)   # rows    (y): how far the peak is from the center
dx = peak[1] - (W - 1)   # columns (x)

print("Cross-correlation result")
print(f"  detected y shift (rows):    {dy:+d}")
print(f"  detected x shift (columns): {dx:+d}")
print("  Sign convention: this is how far the SHIFTED image moved relative to")
print("  the ORIGINAL.  +y = down (image row numbers grow downward),")
print("  +x = right.  Negative values would mean up / left.")

# --- 3. Check with scikit-image's built-in tool -----------------------------
# phase_cross_correlation(reference, moving) answers a DIFFERENT question:
# "how far must I move `moving` so it lines up with `reference`?"
# That is the opposite direction, so the signs are flipped.
shift_back, error, _ = phase_cross_correlation(original, shifted)
print("\nscikit-image phase_cross_correlation(original, shifted)")
print(f"  returned (y, x) = ({shift_back[0]:+.0f}, {shift_back[1]:+.0f})")
print("  Sign convention: the shift you must APPLY to the shifted image to")
print("  move it back onto the original - so the opposite signs.")

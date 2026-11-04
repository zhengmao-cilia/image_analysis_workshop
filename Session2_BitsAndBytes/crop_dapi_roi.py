"""Cut a small rectangle (an ROI, "region of interest") out of the DAPI channel
and save it as its own small TIFF file.

Run it from the top folder of the workshop repository with:

    uv run Session2_BitsAndBytes/crop_dapi_roi.py

The cropped image is saved next to this script as DAPI_roi.tif.
"""

from pathlib import Path

import numpy as np
from bioio import BioImage
from skimage import io

# Where the images live.  Path(__file__) is this script, ".parent.parent" walks
# up two folders to the top of the workshop repository.
REPO = Path(__file__).resolve().parent.parent
IMAGE_DIR = REPO / "testImages" / "3Channel_FF_Corrected_1"
IMAGE = IMAGE_DIR / "3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif"
OUTPUT = Path(__file__).resolve().parent / "DAPI_roi.tif"

# The ROI, in pixels.  x counts columns from the left edge, y counts rows from
# the top edge, both starting at 0.  The top-left corner of the ROI is (x, y).
X = 215
Y = 292
WIDTH = 32
HEIGHT = 30

# Open the image and find which channel is DAPI by its name, which the
# microscope stored in the file.  This is safer than assuming DAPI is
# "channel 2": it keeps working even if the channels come in another order.
img = BioImage(IMAGE)
channel_names = [str(name) for name in img.channel_names]
dapi_index = channel_names.index("DAPI")

# Get only the DAPI channel, as a grid of rows ("Y") and columns ("X").
dapi = img.get_image_data("YX", C=dapi_index)
print(f"Loaded {IMAGE.name}")
print(f"  channels: {channel_names}  -> DAPI is channel {dapi_index}")
print(f"  DAPI image: {dapi.shape[0]} rows x {dapi.shape[1]} columns, "
      f"number type {dapi.dtype}")

# Cut out the ROI.  Rows come first, columns second: [rows, columns].
# "Y:Y + HEIGHT" means "from row Y up to (not including) row Y + HEIGHT".
roi = dapi[Y:Y + HEIGHT, X:X + WIDTH]
print(f"\nROI: x={X}, y={Y}, width={WIDTH}, height={HEIGHT}")
print(f"  cropped shape: {roi.shape[0]} rows x {roi.shape[1]} columns, "
      f"number type {roi.dtype}")

# Compare the brightness inside the ROI with the whole DAPI image, so you can
# check that the ROI really covers a nucleus and not empty background.
print(f"  values in ROI   : {roi.min()} to {roi.max()} (average {roi.mean():.0f})")
print(f"  whole DAPI image: {dapi.min()} to {dapi.max()} "
      f"(average {dapi.mean():.0f}, 99% of pixels below "
      f"{np.percentile(dapi, 99):.0f})")

# Save the ROI.  The pixel values and number type (uint16) are kept exactly
# as they are; nothing is rescaled.  "check_contrast=False" stops a warning
# that the image looks dark, which is expected for raw microscope data.
io.imsave(OUTPUT, roi, check_contrast=False)
print(f"\nSaved {OUTPUT}")

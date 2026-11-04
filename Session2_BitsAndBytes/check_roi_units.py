"""Check why the DAPI ROI came out as background noise.

Run it from the top folder of the workshop repository with:

    uv run Session2_BitsAndBytes/check_roi_units.py

It does not change or save anything.  It prints:
  1. the image's dimensions, axis order, channels and pixel size,
  2. what is inside the ROI under different readings of the numbers
     x=215, y=292, width=32, height=30.
"""

from pathlib import Path

import numpy as np
from bioio import BioImage
from skimage import filters, measure

REPO = Path(__file__).resolve().parent.parent
IMAGE = (REPO / "testImages" / "3Channel_FF_Corrected_1"
         / "3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif")

# The ROI exactly as given (not changed).
X, Y, WIDTH, HEIGHT = 215, 292, 32, 30

# --- 1. What is in the file? ---------------------------------------------
img = BioImage(IMAGE)
channel_names = [str(name) for name in img.channel_names]
pixel_size = img.physical_pixel_sizes.X  # micrometers per pixel

print(f"File: {IMAGE.name}")
print(f"  axis order and sizes: {img.dims}")
print(f"  channels            : {channel_names}")
print(f"  pixel size          : {pixel_size} micrometers")

dapi_index = channel_names.index("DAPI")
dapi = img.get_image_data("YX", C=dapi_index)
print(f"  DAPI = channel {dapi_index}, {dapi.shape[0]} rows x {dapi.shape[1]} columns")
print(f"  DAPI average {dapi.mean():.0f}, 99% of pixels below "
      f"{np.percentile(dapi, 99):.0f}")

# --- 2. Find the nuclei, so we can see whether an ROI contains one --------
# Otsu's method picks a brightness cut-off that separates bright nuclei from
# dark background.  "label" then numbers each separate bright blob.
cutoff = filters.threshold_otsu(dapi)
nuclei = [n for n in measure.regionprops(measure.label(dapi > cutoff))
          if n.area > 100]  # ignore tiny specks
sizes = [n.bbox[2] - n.bbox[0] for n in nuclei]
print(f"\nFound {len(nuclei)} nuclei; a typical one is about "
      f"{np.median(sizes):.0f} pixels "
      f"({np.median(sizes) * pixel_size:.0f} micrometers) across.")


def describe(label, row, col, height, width):
    """Print the brightness inside an ROI and how many whole nuclei it holds."""
    roi = dapi[row:row + height, col:col + width]
    whole = sum(
        1 for n in nuclei
        if n.bbox[0] >= row and n.bbox[1] >= col
        and n.bbox[2] <= row + height and n.bbox[3] <= col + width
    )
    print(f"\n{label}")
    print(f"  rows {row}-{row + height}, columns {col}-{col + width} "
          f"({height} x {width} pixels)")
    print(f"  average brightness {roi.mean():.0f}, brightest {roi.max()}, "
          f"whole nuclei inside: {whole}")


# --- 3. Try different readings of the same four numbers --------------------
describe("Reading A: numbers are PIXELS (what crop_dapi_roi.py did)",
         Y, X, HEIGHT, WIDTH)
describe("Reading B: pixels, but x and y swapped",
         X, Y, HEIGHT, WIDTH)
describe("Reading C: numbers are MICROMETERS (converted to pixels)",
         round(Y / pixel_size), round(X / pixel_size),
         round(HEIGHT / pixel_size), round(WIDTH / pixel_size))

"""Open the mitochondria image and the mitochondria template together in Napari.

No analysis is done here - this is only for looking at the two images.

Run this script from the top folder of the workshop repository with:

    uv run Session3_Filters/open_mito_and_template.py

Close the Napari window to end the script.
"""

from pathlib import Path

import napari
from skimage import io

# Where the images live.  Path(__file__) is this script; ".parent.parent"
# walks up two folders to the top of the workshop repository.
REPO = Path(__file__).resolve().parent.parent
MITO_FILE = REPO / "testImages" / "Mito_Cropped.tif"
TEMPLATE_FILE = REPO / "testImages" / "Mito_template.tif"

# Read the pixels of both images into memory (each is a grid of numbers).
mito = io.imread(MITO_FILE)
template = io.imread(TEMPLATE_FILE)

# Print a few facts about each image so you know what you are looking at.
for name, image in [("Mitochondria", mito), ("Template", template)]:
    print(f"{name}: size {image.shape[0]} x {image.shape[1]} pixels, "
          f"number type {image.dtype}, "
          f"darkest {image.min()}, brightest {image.max()}")

viewer = napari.Viewer()

# The full mitochondria image, placed with its top-left corner at (0, 0).
viewer.add_image(mito, name="mitochondria (full)", colormap="gray")

# The template is tiny compared with the full image.  If we added it without
# moving it, Napari would draw it on top of the top-left corner of the
# mitochondria image.  "translate" shifts the layer: (rows down, columns
# right).  Here we put it just to the right of the full image, with a
# 10-pixel gap.  Both layers keep the same pixel size, so you see how big
# the template really is compared with the mitochondria.
gap = 10
viewer.add_image(
    template,
    name="template",
    colormap="gray",
    translate=(0, mito.shape[1] + gap),
)

# Zoom so that both layers fit in the window.
viewer.reset_view()

# Start the Napari window and wait until you close it.
napari.run()

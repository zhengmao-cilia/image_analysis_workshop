"""Open Cell_cropped.tif in Napari using the "napari-tiff" reader.

This needs the napari-tiff plugin.  Install it once, from the top folder of
the workshop repository, with:

    uv add napari-tiff

Then run this script from the same folder with:

    uv run Session2_BitsAndBytes/open_with_napari_tiff.py

Close the Napari window to end the script.
"""

from pathlib import Path

import napari

# Where the image lives.  Path(__file__) is this script, ".parent.parent" walks
# up two folders to the top of the workshop repository.
REPO = Path(__file__).resolve().parent.parent
IMAGE = REPO / "testImages" / "Cell_cropped.tif"

# Napari can read a TIFF file with more than one "reader" (a plugin that
# turns the file into pixels and metadata).  When you drag a file into the
# window, Napari asks which reader to use.  Here we answer that question in
# the code: plugin="napari-tiff" picks the napari-tiff reader, so no question
# pops up.
viewer = napari.Viewer()
layers = viewer.open(IMAGE, plugin="napari-tiff")

# Print what the reader created, so you can see how it interpreted the file.
for layer in layers:
    print(f"Layer '{layer.name}': shape {layer.data.shape}, "
          f"number type {layer.data.dtype}, color image: {layer.rgb}")

# Start the Napari window and wait until you close it.
napari.run()

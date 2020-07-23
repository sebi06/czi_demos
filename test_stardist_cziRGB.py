from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

from glob import glob
from tifffile import imread
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible

from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
from stardist.models import StarDist2D

from aicspylibczi import CziFile

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imgfileutils as imf
#from czitools import imgfileutils as imf
from czitools import segmentation_tools as sgt
from aicsimageio import AICSImage, imread
from skimage import measure, segmentation
from skimage.measure import regionprops
from skimage.color import label2rgb
from skimage.color import rgb2gray
import progressbar
from IPython.display import display, HTML
from MightyMosaic import MightyMosaic

# specify the filename of the CZI file
filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\Atomic\Nuclei\nuclei_RGB\H&E\Tumor_H&E_small2.czi"
# get the metadata from the czi file
md, addmd = imf.get_metadata(filename)

# show some metainformation
print('------------------   Show Metainformation ------------------')

# shape and dimension entry from CZI file as returned by czifile.py
print('Array Shape (czifile)          : ', md['Shape'])
print('Dimension Entry (czifile)      : ', md['Axes'])
print('Array Shape (aicsimageio)      : ', md['Shape_aics'])
print('Dimension Entry (aicsimageio)  : ', md['Axes_aics'])
print('------------------------------------------------------------')
print('SizeS : ', md['SizeS'])
print('SizeT : ', md['SizeT'])
print('SizeZ : ', md['SizeZ'])
print('SizeC : ', md['SizeC'])
print('SizeX (czifile) : ', md['SizeX'])
print('SizeY (czifile) : ', md['SizeY'])
print('SizeY (aicsimageio) : ', md['SizeX_aics'])
print('SizeY (aicsimageio) : ', md['SizeY_aics'])
print('XScale : ', md['XScale'])
print('Yscale : ', md['YScale'])
print('Zscale : ', md['ZScale'])


####################

czi = CziFile(filename)

# Get the shape of the data, the coordinate pairs are (start index, size)
dimensions = czi.dims_shape()
print(dimensions)
print(czi.dims)
print(czi.size)
print(czi.is_mosaic())  # True
# Mosaic files ignore the S dimension and use an internal mIndex to reconstruct
# the scale factor allows one to generate a manageable image
mosaic_data = czi.read_mosaic(C=0, scale_factor=1)
print('CZI Mosaic Data Shape : ', mosaic_data.shape)


plt.figure(figsize=(8, 8))
image2d = mosaic_data[0, :, 0, :, :]
image2d = np.moveaxis(image2d, 0, -1)

# convert ZEN BGR into RGB
image2d = image2d[..., ::-1]

"""
plt.imshow(image2d)
plt.axis('off')
plt.show()
"""

"""
# Load the image slice I want from the file
for m in range(0, 4):
    img, shp = czi.read_image(M=m, C=0)
    print('CZI Single Tile Shape : ', img.shape)
    # print(shp)

    bgr = img[0, 0, :, 0, :, :]
    bgr = np.moveaxis(bgr, 0, -1)
    # convert ZEN BGR into RGB
    rgb = bgr[..., ::-1]

    plt.imshow(rgb)
    plt.axis('off')
    plt.show()

################
"""


# get the current plane indices and store them
values = {'S': 0, 'T': 0, 'Z': 0, 'C': 0, 'Number': 0}


n_channel = 1 if image2d.ndim == 2 else image2d.shape[-1]
axis_norm = (0, 1)   # normalize channels independently
# axis_norm = (0, 1, 2)  # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

"""Predict instance segmentation from input image.
Parameters
----------
img : :class:`numpy.ndarray`
    Input image
axes : str or None
    Axes of the input ``img``.
    ``None`` denotes that axes of img are the same as denoted in the config.
normalizer : :class:`csbdeep.data.Normalizer` or None
    (Optional) normalization of input image before prediction.
    Note that the default (``None``) assumes ``img`` to be already normalized.
prob_thresh : float or None
    Consider only object candidates from pixels with predicted object probability
    above this threshold (also see `optimize_thresholds`).
nms_thresh : float or None
    Perform non-maximum suppression that considers two objects to be the same
    when their area/surface overlap exceeds this threshold (also see `optimize_thresholds`).
n_tiles : iterable or None
    Out of memory (OOM) errors can occur if the input image is too large.
    To avoid this problem, the input image is broken up into (overlapping) tiles
    that are processed independently and re-assembled.
    This parameter denotes a tuple of the number of tiles for every image axis (see ``axes``).
    ``None`` denotes that no tiling should be used.
show_tile_progress: bool
    Whether to show progress during tiled prediction.
predict_kwargs: dict
    Keyword arguments for ``predict`` function of Keras model.
nms_kwargs: dict
    Keyword arguments for non-maximum suppression.
overlap_label: scalar or None
    if not None, label the regions where polygons overlap with that value
Returns
-------
(:class:`numpy.ndarray`, dict)
    Returns a tuple of the label instances image and also
    a dictionary with the details (coordinates, etc.) of all remaining polygons/polyhedra.
"""


demo_model = True

if demo_model:
    print(
        "NOTE: This is loading a previously trained demo model!\n"
        "      Please set the variable 'demo_model = False' to load your own trained model.",
        file=sys.stderr, flush=True
    )
    model = StarDist2D.from_pretrained('Versatile (H&E nuclei)')

else:
    model = StarDist2D(None, name='stardist', basedir='models')
None

img = normalize(image2d,
                pmin=1,
                pmax=99.8,
                axis=axis_norm,
                clip=False,
                eps=1e-20,
                dtype=np.float32)

mask, details = model.predict_instances(img,
                                        axes=None,
                                        normalizer=None,
                                        prob_thresh=0.7,
                                        nms_thresh=0.3,
                                        n_tiles=None,
                                        show_tile_progress=True,
                                        overlap_label=None
                                        )

#plt.figure(figsize=(8, 8))
#plt.imshow(img if img.ndim == 2 else img[..., 0], clim=(0, 1), cmap='gray')
#plt.imshow(mask, cmap=lbl_cmap, alpha=0.5)
# plt.axis('off')


# define measure region properties
to_measure = ('label',
              'area',
              'centroid',
              'max_intensity',
              'mean_intensity',
              'min_intensity',
              'bbox')

# measure the specified parameters store in dataframe
props = pd.DataFrame(
    measure.regionprops_table(
        mask,
        # intensity_image=rgb2gray(image2d),
        intensity_image=image2d[:, :, 2],
        properties=to_measure
    )
).set_index('label')

# filter objects by size and intensity

maxR = 120
maxG = 130
maxB = 220

max_meanint = 0.2125 * maxR + 0.7154 * maxG + 0.0721 * maxB
print('MeanIntensty (max) : ', max_meanint)


props = props[(props['area'] >= 50) & (props['area'] <= 1000)]
props = props[(props['mean_intensity'] <= max_meanint)]

# add plane indices
props['S'] = 0
props['T'] = 0
props['Z'] = 0
props['C'] = 0

# count the number of objects
values['Number'] = props.shape[0]

print(values)
print(props)

ax = sgt.plot_results(image2d, mask, props, add_bbox=True)

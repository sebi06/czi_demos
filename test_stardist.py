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

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imgfileutils as imf
import segmentation_tools as sgt
from aicsimageio import AICSImage, imread
from skimage import measure, segmentation
from skimage.measure import regionprops
from skimage.color import label2rgb
import progressbar
from IPython.display import display, HTML
from MightyMosaic import MightyMosaic

np.random.seed(6)
lbl_cmap = random_label_cmap()

# specify the filename of the CZI file
filename = r'/datadisk1/tuxedo/temp/input/segment_nuclei_CNN.czi'
# get the metadata from the czi file
md = imf.get_metadata_czi(filename, dim2none=False)

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

s = 0
t = 0
z = 0
chindex = 0
minsize = 50
maxsize = 5000


# get AICSImageIO object using the python wrapper for libCZI
img = AICSImage(filename)

# get the current plane indicies and store them
values = {'S': s, 'T': t, 'Z': z, 'C': chindex, 'Number': 0}


# read out a single 2D image planed using AICSImageIO
image2d = img.get_image_data("YX", S=s, T=t, Z=z, C=chindex)

n_channel = 1 if image2d.ndim == 2 else X[0].shape[-1]
axis_norm = (0, 1)   # normalize channels independently
# axis_norm = (0,1,2) # normalize channels jointly
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
    model = StarDist2D.from_pretrained('Versatile (fluorescent nuclei)')

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
                                        prob_thresh=0.4,
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
        intensity_image=image2d,
        properties=to_measure
    )
).set_index('label')

# filter objects by size
props = props[(props['area'] >= minsize) & (props['area'] <= maxsize)]

# add plane indices
props['S'] = s
props['T'] = t
props['Z'] = z
props['C'] = chindex

# count the number of objects
values['Number'] = props.shape[0]

print(values)

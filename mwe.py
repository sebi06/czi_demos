# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
from aicsimageio import AICSImage, imread
import progressbar
import shutil
import imgfileutils as imf
import tifffile
from lxml import etree
#from aicspylibczi import CziFile

#filename = r"testdata\WP96_4Pos_B4-10_DAPI.czi"
filename = r'testdata\WP96_2Pos_B2+B4_S=2_T=2_Z=4_C=3_X=512_Y=256.czi'
savename = filename.split('.')[0] + '.ome.tiff'

#czi = CziFile(filename)

# Get the shape of the data, the coordinate pairs are (start index, size)
# print(czi.dims_shape())
# print(czi.dims)
# print(czi.size)

# get the metadata
md, additional_mdczi = imf.get_metadata(filename)

# remove the S dimension from the dimstring
ometiff_dimstring = md['Axes_aics'].replace('S', '')

# get AICSImageIO object using the python wrapper for libCZI
img = AICSImage(filename)

with tifffile.TiffWriter(savename, append=False) as tif:

    for s in progressbar.progressbar(range(img.shape[0]), redirect_stdout=True):

        # get the 5d image stack
        image5d = img.get_image_data("TZCYX", S=s)

        # write scene as OME-TIFF series
        tif.save(image5d,
                 photometric='minisblack',
                 metadata={'axes': ometiff_dimstring,
                           'PhysicalSizeX': np.round(md['XScale'], 3),
                           'PhysicalSizeXUnit': md['XScaleUnit'],
                           'PhysicalSizeY': np.round(md['YScale'], 3),
                           'PhysicalSizeYUnit': md['YScaleUnit'],
                           'PhysicalSizeZ': np.round(md['ZScale'], 3),
                           'PhysicalSizeZUnit': md['ZScaleUnit']
                           }
                 )

    # close the AICSImage object at the end
    img.close()

print('Done')

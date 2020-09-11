# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import imgfileutils as imf
from aicsimageio import AICSImage, imread
import progressbar
import shutil
from apeer_ometiff_library import io, omexmlClass
import tifffile
import itertools as it

###################################################################

# filename = r"testdata\WP96_4Pos_B4-10_DAPI.czi"
filename = r'testdata\WP96_2Pos_B2+B4_S=2_T=2_Z=4_C=3_X=512_Y=256.czi'
savename = filename.split('.')[0] + '.ome.tiff'

# get the metadata
md, additional_mdczi = imf.get_metadata(filename)
numimg = md['SizeS'] * md['SizeT'] * md['SizeZ'] * md['SizeC']

# get AICSImageIO object using the python wrapper for libCZI
img = AICSImage(filename)

dims_dict, dimindex_list, numvalid_dims = imf.get_dimorder(md['Axes_aics'])
shape5d = list(md['Shape_aics'])
shape5d.pop(dims_dict['S'])

# create image5d for the current scene
image5d = np.zeros(shape5d, dtype=md['NumPy.dtype'])

# remove the S dimension from the dimstring
dimstring5d = md['Axes_aics'].replace('S', '')

# open the TiffWriter
with tifffile.TiffWriter(savename, append=False) as tif:

    for s in progressbar.progressbar(range(md['SizeS']), redirect_stdout=True):
        for t in range(md['SizeT']):
            for z in range(md['SizeZ']):
                for c in range(md['SizeC']):
                    image2d = img.get_image_data("YX", S=s, T=t, Z=z, C=c)

                    # do some processing with the image2d
                    # ....

        # update the 5d stack
        image5d = imf.update5dstack(image5d, image2d,
                                    dimstring5d=dimstring5d,
                                    t=t,
                                    z=z,
                                    c=c)

        # write scene as OME-TIFF series
        tif.save(image5d,
                 photometric='minisblack',
                 metadata={'axes': dimstring5d,
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

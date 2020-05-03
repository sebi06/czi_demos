# -*- coding: utf-8 -*-

#################################################################
# File        : test_write_ometiff.py
# Version     : 0.1
# Author      : czsrh
# Date        : 20.04.2020
# Institution : Carl Zeiss Microscopy GmbH
#
# Copyright (c) 2020 Carl Zeiss AG, Germany. All Rights Reserved.
#################################################################

import imgfileutils as imf
import os
import numpy as np
from aicsimageio import AICSImage, imread, imread_dask
from aicsimageio.writers import ome_tiff_writer
import ometifftools as ott


#filename = r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/celldivision/CellDivision_T=10_Z=15_CH=2_DCV_small.czi'
#filename = r'/datadisk1/tuxedo/temp/input/nuctest01.ome.tiff'
#filename = r'/datadisk1/tuxedo/temp/input/A01.czi'
filename = r'/datadisk1/tuxedo/temp/input/WP384_4Pos_B4-10_DAPI.czi'
fileout = r'/datadisk1/tuxedo/temp/output/testwrite.ome.tiff'


md, addmd = imf.get_metadata(filename)

# read CZI using AICSImageIO library
img = AICSImage(filename)
stack = img.get_image_data()

# save file as OME-TIFF
fs = ott.write_ometiff_aicsimageio(fileout, stack, md,
                                   reader='aicsimageio',
                                   overwrite=True)

print('Save stack as:', fs)

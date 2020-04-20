# -*- coding: utf-8 -*-

#################################################################
# File        : test_xmlwrite.py
# Version     : 0.1
# Author      : czsrh
# Date        : 20.04.2020
# Institution : Carl Zeiss Microscopy GmbH
#
# Copyright (c) 2020 Carl Zeiss AG, Germany. All Rights Reserved.
#################################################################

import imgfileutils as imf

# define your testfiles here
filename_ometiff = r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/celldivision/CellDivision_T=15_Z=20_CH=2_DCV_small.ome.tiff'
filename_czi = r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/wellplate/testwell96.czi'

# get metadata for czi
md, addmd = imf.get_metadata(filename_czi)
xmlczi = imf.writexml_czi(filename_czi)
print(xmlczi)

# get metadata for OME-TIFF
xmlometiff = imf.writexml_ometiff(filename_ometiff)
print(xmlometiff)


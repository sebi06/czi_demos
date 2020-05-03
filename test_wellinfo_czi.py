# -*- coding: utf-8 -*-

#################################################################
# File        : test_wellinfo_czi.py
# Version     : 0.1
# Author      : czsrh
# Date        : 20.04.2020
# Institution : Carl Zeiss Microscopy GmbH
#
# Copyright (c) 2020 Carl Zeiss AG, Germany. All Rights Reserved.
#################################################################

import imgfileutils as imf
import pandas as pd

# filename = r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/wellplate//testwell96.czi'
# filename = r'WP384_4Pos_B4-10_DAPI.czi'
filename = r'input/A01.czi'

# get the metadata from the czi file
md, additional_md = imf.get_metadata(filename)

# convert metadata dictionary to a pandas dataframe
mdframe = imf.md2dataframe(md)

# shape and dimension entry from CZI file as returned by czifile.py
print('CZI Array Shape : ', md['Shape'])
print('CZI Dimension Entry : ', md['Axes'])

# show dimensions
print('--------   Show Dimensions --------')
print('SizeS : ', md['SizeS'])
print('SizeT : ', md['SizeT'])
print('SizeZ : ', md['SizeZ'])
print('SizeC : ', md['SizeC'])

well2check = 'A1'
isids = imf.getImageSeriesIDforWell(md['Well_ArrayNames'], well2check)

print('WellList            : ', md['Well_ArrayNames'])
print('Well Column Indices : ', md['Well_ColId'])
print('Well Row Indices    : ', md['Well_RowId'])
print('WellCounter         : ', md['WellCounter'])
print('Different Wells     : ', md['NumWells'])
print('ImageSeries Ind. Well ', well2check, ' : ', isids)

# print(mdframe[:10])

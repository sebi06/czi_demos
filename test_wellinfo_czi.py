# -*- coding: utf-8 -*-
"""
@author: Sebi

File: test_wellinfo_czi.py
Date: 19.12.2019
Version. 0.1
"""

import imgfileutils as imf

filenames = [r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/wellplate//B4_B5_S=8_4Pos_perWell_T=2_Z=1_CH=1.czi',
             r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/wellplate//96well-SingleFile-Scene-05-A5-A5.czi',
             r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/wellplate//testwell96.czi',
             r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/wellplate/96Well_CH=1_1P.czi']

filename = filenames[3]


# get the metadata from the czi file
md = imf.get_metadata_czi(filename, dim2none=False)

# shape and dimension entry from CZI file as returned by czifile.py
print('CZI Array Shape : ', md['Shape'])
print('CZI Dimension Entry : ', md['Axes'])

# show dimensions
print('--------   Show Dimensions --------')
print('SizeS : ', md['SizeS'])
print('SizeT : ', md['SizeT'])
print('SizeZ : ', md['SizeZ'])
print('SizeC : ', md['SizeC'])

well2check = 'B4'
isids = imf.getImageSeriesIDforWell(md['Well_ArrayNames'], well2check)

print('WellList            : ', md['Well_ArrayNames'])
print('Well Column Indices : ', md['Well_ColId'])
print('Well Row Indices    : ', md['Well_RowId'])
print('WellCounter         : ', md['WellCounter'])
print('Different Wells     : ', md['NumWells'])
print('ImageSeries Ind. Well ', well2check, ' : ', isids)

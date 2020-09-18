import sys
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import imgfileutils as imf
from aicsimageio import AICSImage, imread
from aicspylibczi import CziFile
import czifile as zis
import xmltodict

# select plotting backend
plt.switch_backend('Qt5Agg')

# filename = r'/datadisk1/tuxedo/temp/input/WP96_T=3_Z=4_Ch=2_3x3_A4-A5.czi'
# filename = r'/datadisk1/tuxedo/temp/input/WP96_T=3_Z=4_Ch=2_5x5_A4.czi'
#filename = r'/datadisk1/tuxedo/temp/input/DTScan_ID4.czi'
#filename = r"C:\Temp\input\DTScan_ID4.czi"
#filename = r"C:\Users\m1srh\Downloads\New version with alpha channel to avoid tile dissaperance2.czi"
#filename = r"D:\ImageData\BrainSlide\DTScan_ID2.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\Atomic\Nuclei\nuclei_RGB\H&E\Tumor_H+E_small2.czi"
#filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\Atomic\Nuclei\nuclei_RGB\H&E\Tumor_H+E.czi"
#filename = r'D:\OneDrive - Carl Zeiss AG\Testdata_Zeiss\Atomic\Nuclei\nuclei_RGB\H&E\Tumor_H+E.czi'
#filename = r"D:\ImageData\Castor\Z-Stack_DCV\CellDivision_T=10_Z=15_CH=2_DCV_small.czi"
filename = r'testdata\WP96_2Pos_B2+B4_S=2_T=2_Z=4_C=3_X=512_Y=256.czi'

# get the metadata from the czi file
md, addmd = imf.get_metadata(filename)

# show some metainformation
print('------------------   Show Metainformation ------------------')

# shape and dimension entry from CZI file as returned by czifile.py
print('Array Shape (czifile)          : ', md['Shape_czifile'])
print('Dimension Entry (czifile)      : ', md['Axes_czifile'])
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


####################

czi = CziFile(filename)

# Get the shape of the data, the coordinate pairs are (start index, size)
dimensions = czi.dims_shape()
print('CZI Dimensions : ', dimensions)
print('CZI DimeString : ', czi.dims)
print('CZI Size       : ', czi.size)
print('CZI IsMosaic   : ', czi.is_mosaic())
print('CZI Scene Shape consistent : ', czi.shape_is_consistent)

if md['ImageType'] == 'czi':
    isCZI = True
else:
    isCZI = False

if md['isRGB']:
    isRGB= True
else:
    isRGB = False

if md['czi_ismosaic']:
    isMosaic = True
else:
    isMosaic = False


#output = czi.read_image(S=0, C=0)
#image = output[0]
#image_dims = output[1]

mosaic_data = czi.read_mosaic(C=0, Z=0, scale_factor=0.2)
image2d = np.squeeze(mosaic_data)
#image2d = np.moveaxis(image2d, 0, -1)
# convert ZEN BGR into RGB
image2d = image2d[..., ::-1]

plt.figure(figsize=(12, 12))
plt.imshow(image2d)
plt.axis('off')
plt.show()

"""
# Mosaic files ignore the S dimension and use an internal mIndex to reconstruct
# the scale factor allows one to generate a manageable image
mosaic_data = czi.read_mosaic(C=0, scale_factor=1)
print('CZI Mosaic Data Shape : ', mosaic_data.shape)


plt.figure(figsize=(8, 8))
image2d = mosaic_data[0, :, 0, :, :]
image2d = np.moveaxis(image2d, 0, -1)

# convert ZEN BGR into RGB
image2d = image2d[..., ::-1]

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
"""

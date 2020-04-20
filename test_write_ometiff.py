import imgfileutils as imf
import os
import numpy as np
from aicsimageio import AICSImage, imread, imread_dask
from aicsimageio.writers import ome_tiff_writer
import ometifftools as ott


filename = r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/celldivision/CellDivision_T=10_Z=15_CH=2_DCV_small.czi'
fileout = r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/celldivision/testwrite.ome.tiff'

md, addmd = imf.get_metadata(filename)

# read CZI using AICSImageIO library
img = AICSImage(filename)
stack = img.get_image_data()

# save file as OME-TIFF
fs = ott.write_ometiff_aicsimageio(fileout, stack, md,
                                   czireader='aicsimageio',
                                   overwrite=True)

print('Save CZI file as:', fs)

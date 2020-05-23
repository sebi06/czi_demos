import numpy as np
import imgfileutils as imf
from aicspylibczi import CziFile
import czifile as zis
import xmltodict

# filename = r'/datadisk1/tuxedo/temp/input/WP96_T=3_Z=4_Ch=2_3x3_A4-A5.czi'
# filename = r'/datadisk1/tuxedo/temp/input/WP96_T=3_Z=4_Ch=2_5x5_A4.czi'
filename = r'/datadisk1/tuxedo/temp/input/DTScan_ID4.czi'

print('----------czifile array ----------')
czi_czifile_array, md, addmd = imf.get_array_czi(filename, remove_HDim=False)
print('Shape czi_czifile', czi_czifile_array.shape)
czi_aics = CziFile(filename)
czi_aics_out = czi_aics.read_image(S=0)
czi_aics_array = czi_aics_out[0]
czi_aics_dims = czi_aics_out[1]

print('Shape czi_aics_array', czi_aics_array.shape)
print('Shape czi_aics_dims', czi_aics_dims)
print('CZI Mosaic', md['czi_ismosaic'])
print('SizeX', md['SizeX'])
print('SizeY', md['SizeY'])
print('SizeC', md['SizeC'])
print('SizeZ', md['SizeZ'])
print('SizeT', md['SizeT'])
print('SizeS', md['SizeS'])
print('SizeM', md['SizeM'])
print('SizeB', md['SizeB'])
print('SizeH', md['SizeH'])

print('------------- aics pylibczi -------_-----')
print('dims_aicspylibczi', md['dims_aicspylibczi'])
print('dimorder_aicspylibczi', md['dimorder_aicspylibczi'])
print('size_aicspylibczi', md['size_aicspylibczi'])
print('czi_ismosaic', md['czi_ismosaic'])

print('--, ----------- czifile -------------')
print('Shape_czifile', md['Shape_czifile'])
print('Axes_czifile', md['Axes_czifile'])

print('------------- aicsimageio -------------')
print('Axes_aics', md['Axes_aics'])
print('Shape_aics', md['Shape_aics'])

"""
# Mosaic files ignore the S dimension and use an internal mIndex to reconstruct, the scale factor allows one to generate a manageable image
czi_aics_mosaicsize = czi_aics.read_mosaic_size()
czi_aics_mosaic_array = czi_aics.read_mosaic(C=0, scalefactor=1.0)
print('czi_aics_mosaicsize', czi_aics_mosaicsize)
print('czi_aics_mosaic_array', czi_aics_mosaic_array.shape)

czi_aics_scenebb = czi_aics.scene_bounding_box()
czi_aics_scenehw = czi_aics.scene_height_by_width()
print('czi_aics_scenebb', czi_aics_scenebb)
print('czi_aics_scenehw', czi_aics_scenehw)
"""

dims_dict, dimindex_list, numvalid_dims = imf.get_dimorder(md['dimorder_aicspylibczi'])

print(dims_dict)
print(dimindex_list)

a = [x for x in dimindex_list if x >= 0]
new_dict = {key: val for key, val in dims_dict.items() if val >= 0}

print(a)
print(new_dict)

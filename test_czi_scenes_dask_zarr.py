from aicspylibczi import CziFile
import imgfile_tools as imf
import czifile_tools as czt
import numpy as np
import zarr
import dask
import dask.array as da
from dask import delayed
from itertools import product
import napari

# filename = r"C:\Temp\input\DTScan_ID4.czi"
#filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\S=1_3x3_T=1_Z=1_CH=2.czi"
#filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=1_Z=1_CH=2.czi"
filename = r"/datadisk1/tuxedo/testpictures/Testdata_Zeiss/CZI_Testfiles/S=2_3x3_T=1_Z=1_CH=2.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=1_Z=4_CH=2.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=3_Z=1_CH=2.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\S=1_3x3_T=3_Z=4_CH=2.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=3_Z=4_CH=2.czi"

######################################################################

# get the metadata from the czi file
md, additional_mdczi = imf.get_metadata(filename)

# check if CZI has T or Z dimension
hasT = False
hasZ = False
if 'T' in md['dims_aicspylibczi']:
    hasT = True
if 'Z' in md['dims_aicspylibczi']:
    hasZ = True

# read CZI using aicslibczi
czi = CziFile(filename)

"""
# read info for 1st scene
sid = 0
scene1 = czt.CZIScene(czi, sceneindex=sid)

# array_shape = (1, 1, 1, md['SizeC'], s.height, s.width)
dimpos = md['dimpos_aics']
posS = dimpos['S']
posT = dimpos['T']
posZ = dimpos['Z']
posC = dimpos['C']

array_shape_list = []
posdict = {'S': 'SizeS', 'T': 'SizeT', 'C': 'SizeC', 'Z': 'SizeZ'}

# find key based upon value
kk = []
for v in range(4):
    # get the corresponding dim_id, e.g. 'S'
    dim_id = imf.get_key(dimpos, v)
    # get the correspong string to access the size of tht dimension
    dimstr = posdict[dim_id]
    # append size for this dimension to list containing the shape
    array_shape_list.append(md[dimstr])

# add width and height of scene to the required shape list
array_shape_list.append(scene1.height)
array_shape_list.append(scene1.width)
"""

# get the required shape for all and singel scenes
shape_all, shape_single, same_shape = czt.get_scene_arraydims(czi, md)
print(' Required_Array Shape for all scenes: ', shape_all)
for sh in shape_single:
    print('Required Array Shape for single scenes: ', sh)


# define array to store all channels
# scene_array = zarr.create(tuple(array_shape_list),
#                          dtype = md['NumPy.dtype'],
#                          chunks = True)

scene_array = np.empty(shape_all, dtype=md['NumPy.dtype'])

# loop over all scenes
for s in range(md['SizeS']):

    # get the CZIscene for the current scene
    scene = czt.CZIScene(czi, sceneindex=s)

    for ch in range(md['SizeC']):
        print('Reading Mosaic for Channel:', ch)
        slice = czi.read_mosaic(region=(scene.xstart,
                                        scene.ystart,
                                        scene.width,
                                        scene.height),
                                scale_factor=1.0,
                                C=ch)
        # insert the slice at the corrent index
        if md['dimpos_aics']['C'] == 1:
            scene_array[s, ch, 0, 0, :, :] = np.squeeze(slice)
        if md['dimpos_aics']['C'] == 2:
            scene_array[s, 0, ch, 0, :, :] = np.squeeze(slice)
        if md['dimpos_aics']['C'] == 3:
            scene_array[s, 0, 0, ch, :, :] = np.squeeze(slice)


print(scene_array.shape)

with napari.gui_qt():
    # specify contrast_limits and is_pyramid=False with big data
    # to avoid unnecessary computations
    # napari.view_image(scene_array, contrast_limits=[0, 2000], multiscale=False)

    viewer = napari.Viewer()

    layers = imf.show_napari(viewer, scene_array, md,
                             blending='additive',
                             gamma=0.85,
                             add_mdtable=True,
                             rename_sliders=True)

###########################################################

"""
def load_image(czi, md, ch=0):

    # get the array for a specifc scene, the BBox and the updated metadata
    scene, bbox, md = czt.read_scene_bbox(czi, md,
                                          sceneindex=s,
                                          channel=ch,
                                          timepoint=t,
                                          zplane=z,
                                          scalefactor=1.0)
    print('Reading one scene')

    return scene


width = md['BBoxes_Scenes'][0].width
height = md['BBoxes_Scenes'][0].height

print(md['SizeS'], md['SizeT'], md['SizeZ'], md['SizeC'], height, width)
sp = [md['SizeT'], md['SizeZ'], md['SizeC'], height, width]

# z = zarr.create(sp)
# print(z.shape)

# create dask stack of lazy image readers
lazy_process_image = dask.delayed(load_image)  # lazy reader

lazy_arrays = [lazy_process_image(cziobject, md, s=s, t=t, z=z, ch=ch)
               for s, t, z, ch in product(range(md['SizeS']),
                                          range(md['SizeT']),
                                          range(md['SizeZ']),
                                          range(md['SizeC']))
               ]

dask_arrays = [da.from_delayed(lazy_array,
                               shape=sp,
                               dtype=md['NumPy.dtype'])
               for lazy_array in lazy_arrays
               ]

# Stack into one large dask.array
dask_stack = da.stack(dask_arrays, axis=0)
print(dask_stack.shape)
"""

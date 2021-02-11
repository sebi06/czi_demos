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

# filename = r"testdata\Tumor_H+E_small2.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\Well_B2-4_S=4_T=1_Z=1_C=1.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\W96_B2+B4_S=2_T=1=Z=1_C=1_Tile=5x9.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\W96_B2+B4_S=2_T=2=Z=4_C=3_Tile=5x9.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\Castor\Z-Stack_DCV\CellDivision_T=10_Z=15_CH=2_DCV_small.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\Castor\Mouse Kidney_40x0.95_3CD_JK_comp.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\S=1_3x3_T=1_Z=1_CH=2.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=1_Z=1_CH=2.czi"
# filename = r"/datadisk1/tuxedo/testpictures/Testdata_Zeiss/CZI_Testfiles/S=2_3x3_T=1_Z=1_CH=2.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=1_Z=4_CH=2.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=3_Z=1_CH=2.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\S=1_3x3_T=3_Z=4_CH=2.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=3_Z=4_CH=2.czi"
# filename = r"C:\Temp\input\DTScan_ID4_small.czi"
filename = r"C:\Temp\input\DTScan_ID4.czi"
# filename = r"C:\Temp\input\OverViewScan_8Brains.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\Castor\testwell96.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\Multiscene_CZI_3Scenes.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=3_CH=2.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_CH=2.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_Z=4_CH=2.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\Z=4_CH=2.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\T=3_Z=4_CH=2.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\T=3_CH=2.czi"

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


# get the required shape for all and single scenes
shape_all, shape_single, same_shape = czt.get_shape_allscenes(czi, md)
print('Required_Array Shape for all scenes: ', shape_all)
for sh in shape_single:
    print('Required Array Shape for single scenes: ', sh)


#array_type = 'dask'
array_type = 'zarr'
#array_type = 'numpy'

if array_type == 'zarr':

    # define array to store all channels
    all_scenes_array = zarr.create(tuple(shape_all),
                                   dtype=md['NumPy.dtype'],
                                   chunks=True)

if array_type == 'numpy':
    all_scenes_array = np.empty(shape_all, dtype=md['NumPy.dtype'])


if array_type == 'zarr' or array_type == 'numpy':

    # loop over all scenes
    for s in range(md['SizeS']):
        # get the CZIscene for the current scene
        single_scene = czt.CZIScene(czi, md, sceneindex=s)
        out = czt.read_czi_scene(czi, single_scene, md)
        all_scenes_array[s, :, :, :, :, :] = np.squeeze(out, axis=0)

    print(all_scenes_array.shape)


elif array_type == 'dask':

    def dask_load_sceneimage(czi, s, md):

        # get the CZIscene for the current scene
        single_scene = czt.CZIScene(czi, md, sceneindex=s)
        out = czt.read_czi_scene(czi, single_scene, md)
        return out

    sp = shape_all[1:]

    # create dask stack of lazy image readers
    lazy_process_image = dask.delayed(dask_load_sceneimage)  # lazy reader

    lazy_arrays = [lazy_process_image(czi, s, md) for s in range(md['SizeS'])]

    dask_arrays = [
        da.from_delayed(lazy_array, shape=sp, dtype=md['NumPy.dtype'])
        for lazy_array in lazy_arrays
    ]
    # Stack into one large dask.array
    all_scenes_array = da.stack(dask_arrays, axis=0)
    print(all_scenes_array.shape)


# show array inside napari viewer
with napari.gui_qt():
    # specify contrast_limits and is_pyramid=False with big data
    # to avoid unnecessary computations
    # napari.view_image(scene_array, contrast_limits=[0, 2000], multiscale=False)

    viewer = napari.Viewer()

    layers = imf.show_napari(viewer, all_scenes_array, md,
                             blending='additive',
                             gamma=0.85,
                             add_mdtable=True,
                             rename_sliders=True)

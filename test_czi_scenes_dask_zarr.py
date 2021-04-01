from aicspylibczi import CziFile
from aicsimageio import AICSImage, imread, imread_dask
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
#filename = r"C:\Testdata_Zeiss\CD7\Z-Stack_DCV\CellDivision_T=10_Z=15_CH=2_DCV_small.czi"
#filename = r"C:\Testdata_Zeiss\CD7\Mouse Kidney_40x0.95_3CD_JK_comp.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\S=1_3x3_T=1_Z=1_CH=2.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=1_Z=1_CH=2.czi"
# filename = r"/datadisk1/tuxedo/testpictures/Testdata_Zeiss/CZI_Testfiles/S=2_3x3_T=1_Z=1_CH=2.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=1_Z=4_CH=2.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=3_Z=1_CH=2.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\S=1_3x3_T=3_Z=4_CH=2.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=3_Z=4_CH=2.czi"
#filename = r"C:\Testdata_Zeiss\DTScan_ID4_small.czi"
#filename = r"C:\Testdata_Zeiss\DTScan_ID4.czi"
#filename = r"D:\Temp\input\DTScan_ID4-nokeeptiles.czi"
#filename = r"D:\Testdata_Zeiss\unmix_bug436511\Raw_nokeeptiles.czi"
#filename = r"D:\Testdata_Zeiss\unmix_bug436511\Raw_keeptiles.czi"
#filename = r"D:\Testdata_Zeiss\unmix_bug436511\Raw_Uncompressed.czi"
#filename = r"D:\Temp\input\OverViewScan_8Brains.czi"
#filename = r"D:\Temp\input\OverViewScan_8Brains-keeptile.czi"
#filename = r"D:\Temp\input\OverViewScan_8Brains-nokeeptile.czi"
#filename = r"C:\Testdata_Zeiss\CD7\testwell96.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\Multiscene_CZI_3Scenes.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=3_CH=2.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_CH=2.czi"
#filename = r"C:\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_Z=4_CH=2.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\Z=4_CH=2.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\T=3_Z=4_CH=2.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\T=3_CH=2.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\LatticeLightSheet\LS_Mitosis_T=150-300.czi"
#filename = r"/datadisk1/tuxedo/testpictures/Testdata_Zeiss/BrainSlide/DTScan_ID4.czi"
#filename = r"/datadisk1/tuxedo/testpictures/Testdata_Zeiss/CZI_Testfiles/96well_S=192_2pos_CH=3.czi"
#filename = r"/datadisk1/tuxedo/testpictures/Testdata_Zeiss/CZI_Testfiles/W96_B2+B4_S=2_T=2=Z=4_C=3_Tile=5x9.czi"
#filename = r"/datadisk1/tuxedo/testpictures/Testdata_Zeiss/Nuclei/nuclei_RGB/H+E/Tumor_H+E_small2.czi"
#filename = r"/datadisk1/tuxedo/testpictures/Testdata_Zeiss/Nuclei/nuclei_RGB/H+E/Tumor_H+E.czi"
#filename = r"/datadisk1/tuxedo/testpictures/Testdata_Zeiss/celldivision/CellDivision_T=10_Z=15_CH=2_DCV_small.czi"
#filename = r"/datadisk1/tuxedo/testpictures/Testdata_Zeiss/CZI_Testfiles/S=2_3x3_T=3_Z=4_CH=2.czi"
#filename = r"C:\Users\m1srh\Downloads\Halo_CZI_small.czi"
filename = r"C:\Testdata_Zeiss\OverViewScan.czi"
#filename = r"d:\Testdata_Zeiss\CZI_Testfiles\aicspylibczi\color_lines.czi"
#filename = r"d:\Testdata_Zeiss\CZI_Testfiles\aicspylibczi\test2.czi"
#filename = r"d:\Testdata_Zeiss\CZI_Testfiles\aicspylibczi\test4.czi"

######################################################################

# get the metadata from the czi file
md, additional_mdczi = imf.get_metadata(filename)

use_aicsimageio = True
use_pylibczi = False
use_dask_delayed = True

# decide which tool to use to read the image
if md['ImageType'] != 'czi':
    use_aicsimageio = True
elif md['ImageType'] == 'czi' and md['czi_isMosaic'] is False:
    use_aicsimageio = True
elif md['ImageType'] == 'czi' and md['czi_isMosaic'] is True:
    use_aicsimageio = False
    use_pylibczi = True


# check if CZI has T or Z dimension
hasT = False
hasZ = False
if 'T' in md['dims_aicspylibczi']:
    hasT = True
if 'Z' in md['dims_aicspylibczi']:
    hasZ = True


if use_aicsimageio:

    img = AICSImage(filename)

    if use_dask_delayed:
        print('Using AICSImageIO to read the image (Dask Delayed Reader).')
        all_scenes_array = img.get_image_data()
    if not use_dask_delayed:
        print('Using AICSImageIO to read the image.')
        all_scenes_array = img.get_image_dask_data()

if not use_aicsimageio and use_pylibczi is True:

    # read CZI using aicspylibczi
    czi = CziFile(filename)

    # for testing
    # Get the shape of the data
    print('Dimensions   : ', czi.dims)
    print('Size         : ', czi.size)
    print('Shape        : ', czi.dims_shape())
    print('IsMoasic     : ', czi.is_mosaic())
    if czi.is_mosaic():
        print('Mosaic Size  : ', czi.read_mosaic_size())

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
        print('Using aicspylibCZI to read the image (ZARR array).')

        # option 1
        # all_scenes_array = zarr.create(tuple(shape_all),
        #                               dtype=md['NumPy.dtype'],
        #                               chunks=True)

        # option 2
        all_scenes_array = zarr.open(r'c:\Temp\czi_scene_all.zarr', mode='w',
                                     shape=shape_all,
                                     chunks=True,
                                     dtype=md['NumPy.dtype'])

    if array_type == 'numpy':
        print('Using aicspylibCZI to read the image (Numpy.Array).')
        all_scenes_array = np.empty(shape_all, dtype=md['NumPy.dtype'])

    if array_type == 'zarr' or array_type == 'numpy':

        # loop over all scenes
        for s in range(md['SizeS']):
            # get the CZIscene for the current scene
            single_scene = czt.CZIScene(czi, md, sceneindex=s)
            out = czt.read_czi_scene(czi, single_scene, md, array_type=array_type)
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
        print('Using aicspylibCZI to read the image (Dask.Array) + Delayed Reading.')
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

    viewer = napari.Viewer()

    layers = imf.show_napari(viewer, all_scenes_array, md,
                             blending='additive',
                             gamma=0.85,
                             add_mdtable=True,
                             rename_sliders=True)

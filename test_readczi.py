from aicspylibczi import CziFile
from aicsimageio import AICSImage, imread, imread_dask
from czifiletools import czifile_tools as czt
#import tools.fileutils as czt
from czifiletools import napari_tools as nap
import numpy as np
import zarr
import dask
import dask.array as da
from dask import delayed
from itertools import product
import napari


#filename = r"d:\Testdata_Zeiss\CZI_Testfiles\DTScan_ID4.czi" # OK
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\testwell96.czi"
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\Z=4_CH=2.czi' # OK
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\T=3_Z=4_CH=2.czi' # OK
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\T=3_CH=2.czi' # OK
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=3_CH=2.czi' # OK
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_CH=2.czi' # OK
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_Z=4_CH=2.czi' # OK
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\S=1_3x3_T=1_Z=1_CH=2.czi' # OK
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=1_Z=1_CH=2.czi' # OK
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=1_Z=4_CH=2.czi' # OK
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=3_Z=1_CH=2.czi' # OK
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\S=1_3x3_T=3_Z=4_CH=2.czi' # OK
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=3_Z=4_CH=2.czi' # OK
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\T=3_Z=4_CH=2.czi' # OK
#filename = r"D:\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_Z=4_CH=2.czi" # OK
#filename = r"d:\Testdata_Zeiss\CZI_Testfiles\OverviewScan.czi" # OK
#filename = r"D:\Testdata_Zeiss\CZI_Testfiles\OverviewScan_M=9_CH=3.czi" # OK
#filename = r"D:\Testdata_Zeiss\CZI_Testfiles\CellDivision_T=10_Z=20_CH=1_DCV.czi"
#filename = r"D:\Testdata_Zeiss\CZI_Testfiles\S=1_HE_Slide_RGB_small.czi"
filename = r"D:\Testdata_Zeiss\CZI_Testfiles\FoLu_mCherryEB3_GFPMito_2_Airyscan Processing.czi"

######################################################################

# get the metadata as a dictionary
md, md_add = czt.get_metadata_czi(filename)

# read CZI using aicspylibczi
czi = CziFile(filename)

# for testing
# Get the shape of the data
print('czi_dims       : ', czi.dims)
print('czi_dims_shape : ', czi.get_dims_shape())
print('czi_size       : ', czi.size)
print('IsMoasic       : ', czi.is_mosaic())

"""
if not czi.is_mosaic():

    # get the shape for the 1st scene
    scene = czt.CZIScene(czi, sceneindex=0)
    shape_all = scene.shape_single_scene

    # only update the shape for the scene if the CZI has an S-Dimension
    if scene.hasS:
        shape_all[scene.posS] = md['SizeS']

    print('Shape all Scenes : ', shape_all)
    print('DimString all Scenes : ', scene.single_scene_dimstr)

    # create an empty array with the correct dimensions
    all_scenes = np.empty(czi.size, dtype=md['NumPy.dtype'])

    # loop over scenes if CZI is not a mosaic image
    for s in range(md['SizeS']):

        # read the image stack for the current scene
        current_scene, shp = czi.read_image(S=s)

        # create th index lists containing the slice objects
        if scene.hasS:
            idl_scene = [slice(None, None, None)] * (len(all_scenes.shape) - 2)
            idl_scene[czi.dims.index('S')] = 0
            idl_all = [slice(None, None, None)] * (len(all_scenes.shape) - 2)
            idl_all[czi.dims.index('S')] = s

            # cast current stack into the stack for all scenes
            all_scenes[tuple(idl_all)] = current_scene[tuple(idl_scene)]

        # if there is no S-Dimension use the stack directly
        if not scene.hasS:
            all_scenes = current_scene

    print('Shape all (no mosaic)', all_scenes.shape)


if czi.is_mosaic():

    # get data for 1st scene and create the required shape for all scenes
    scene = czt.CZIScene(czi, sceneindex=0)
    shape_all = scene.shape_single_scene
    shape_all[scene.posS] = md['SizeS']
    print('Shape all Scenes : ', shape_all)
    print('DimString all Scenes : ', scene.single_scene_dimstr)

    # create empty array to hold all scenes
    all_scenes = np.empty(shape_all, dtype=md['NumPy.dtype'])

    # loop over scenes if CZI is not Mosaic
    for s in range(md['SizeS']):
        scene = czt.CZIScene(czi, sceneindex=s)

        # create a slice object for all_scenes array
        if not scene.isRGB:
            idl_all = [slice(None, None, None)] * (len(all_scenes.shape) - 2)
        if scene.isRGB:
            idl_all = [slice(None, None, None)] * (len(all_scenes.shape) - 3)


        # update the entry with the current S index
        idl_all[scene.posS] = s

        # in case T-Z-H dimension are found
        if scene.hasT is True and scene.hasZ is True and scene.hasH is True:

            # read array for the scene
            for h, t, z, c in product(range(scene.sizeH),
                                      range(scene.sizeT),
                                      range(scene.sizeZ),
                                      range(scene.sizeC)):

                # read the array for the 1st scene using the ROI
                scene_array_htzc = czi.read_mosaic(region=(scene.xstart,
                                                          scene.ystart,
                                                          scene.width,
                                                          scene.height),
                                                  scale_factor=1.0,
                                                  H=h,
                                                  T=t,
                                                  Z=z,
                                                  C=c)

                print('Shape Single Scene : ', scene_array_htzc.shape)
                print('Min-Max Single Scene : ', np.min(scene_array_htzc), np.max(scene_array_htzc))

                # create slide object for the current mosaic scene
                #idl_scene = [slice(None, None, None)] * (len(scene.shape_single_scene) - 2)
                idl_all[scene.posS] = s
                idl_all[scene.posH] = h
                idl_all[scene.posT] = t
                idl_all[scene.posZ] = z
                idl_all[scene.posC] = c

                # cast the current scene into the stack for all scenes
                all_scenes[tuple(idl_all)] = scene_array_htzc
                #print('Min-Max all Scenes : ', np.min(all_scenes), np.max(all_scenes))

        if scene.hasT is False and scene.hasZ is False:

            # create an array for the scene
            for c in range(scene.sizeC):

                scene_array_c = czi.read_mosaic(region=(scene.xstart,
                                                        scene.ystart,
                                                        scene.width,
                                                        scene.height),
                                                scale_factor=1.0,
                                                C=c)

                print('Shape Single Scene : ', scene_array_c.shape)

                # create slide object for the current mosaic scene
                #if not scene.isRGB:
                #    idl_scene = [slice(None, None, None)] * (len(scene.shape_single_scene) - 2)
                #if scene.isRGB:
                #    idl_scene = [slice(None, None, None)] * (len(scene.shape_single_scene) - 3)
                idl_all[scene.posS] = s
                idl_all[scene.posC] = c

                # cast the current scene into the stack for all scenes
                all_scenes[tuple(idl_all)] = scene_array_c
"""

# show stack inside the Napari viewer

# show array inside napari viewer
viewer = napari.Viewer()

# specify the index of the Channel inside the array
cpos = scene.single_scene_dimstr.find('C')

layers = nap.show_napari(viewer, all_scenes, md,
                         use_dimstr=True,
                         dimstr=scene.single_scene_dimstr,
                         blending='additive',
                         adjust_contrast=False,
                         gamma=0.85,
                         add_mdtable=True,
                         rename_sliders=True)

# viewer.run()



"""
# loop over all scenes
for s in range(md['SizeS']):
    # get the CZIscene for the current scene
    single_scene = czt.CZIScene(czi, sceneindex=s)
    out = czt.read_czi_scene(czi, single_scene, md, array_type=array_type)

    index_list_out = [slice(None, None, None)] * (len(all_scenes_array.shape) - 2)
    index_list_out[single_scene.posS] = 0
    index_list = [slice(None, None, None)] * (len(all_scenes_array.shape) - 2)
    index_list[single_scene.posS] = s
    all_scenes_array[tuple(index_list)] = out[tuple(index_list_out)]


        #all_scenes_array[:, s, :, :, :] = out

        #all_scenes_array[s, :, :, :, :, :] = np.squeeze(out, axis=0)

    print(all_scenes_array.shape)
"""
"""
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
"""



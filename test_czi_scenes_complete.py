from aicspylibczi import CziFile
import imgfile_tools as imf
import czifile_tools as czt
import numpy as np
import zarr
import napari
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
filename = r"C:\Temp\input\DTScan_ID4_small.czi"
# filename = r"C:\Temp\input\OverViewScan_8Brains.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\Castor\testwell96.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\Multiscene_CZI_3Scenes.czi"

# get the metadata from the czi file
md, additional_mdczi = imf.get_metadata(filename)

# read CZI using aicslibczi
czi = CziFile(filename)
size = czi.read_mosaic_size()


# get all bboxes for all scenes from the CZI file
all_bboxes = czt.getbboxes_allscenes(czi, numscenes=md['SizeS'])

# read data for first scene
for s in all_bboxes:

    scene_array = zarr.create((1,
                               md['SizeT'],
                               md['SizeZ'],
                               md['SizeC'],
                               s.height,
                               s.width),
                              chunks=True)
    # read a complete scene
    for t, z, c in product(range(md['SizeT']),
                           range(md['SizeZ']),
                           range(md['SizeC'])):
        scene_slice, bbox, md = czt.read_scene_bbox(czi, md,
                                                    sceneindex=s.sceneid,
                                                    timepoint=t,
                                                    zplane=z,
                                                    channel=c,
                                                    scalefactor=1.0)

        # insert scence into zarr array
        scene_array[0, t, z, c, :, :] = np.squeeze(scene_slice)

with napari.gui_qt():

    viewer = napari.Viewer()

    layers = imf.show_napari(viewer, scene_array[:], md,
                             blending='additive',
                             gamma=0.85,
                             add_mdtable=True,
                             rename_sliders=True)

import napari
from aicspylibczi import CziFile
from czifiletools import czifile_tools as czt
from czifiletools import napari_tools as nap
from aicsimageio import AICSImage
import dask.array as da
import time

#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\Z=4_CH=2.czi'
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\T=3_Z=4_CH=2.czi'
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\T=3_CH=2.czi'
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=3_CH=2.czi'
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_CH=2.czi'
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_Z=4_CH=2.czi'
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\S=1_3x3_T=1_Z=1_CH=2.czi'
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=1_Z=1_CH=2.czi'
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=1_Z=4_CH=2.czi'
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=3_Z=1_CH=2.czi'
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\S=1_3x3_T=3_Z=4_CH=2.czi'
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=3_Z=4_CH=2.czi'
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\W96_B2+B4_S=2_T=1=Z=1_C=1_Tile=5x9.czi'
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\Multiscene_CZI_3Scenes.czi'
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\96well_S=192_2pos_CH=3.czi'
#filename = r'D:\Testdata_Zeiss\CD7\384well_DAPI.czi'
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\CellDivision_T=10_Z=20_CH=1_DCV.czi'
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\S=1_HE_Slide_RGB.czi'
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\OverViewScan.czi'
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\DTScan_ID4.czi'
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\w96_A1+A2.czi'
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\testwell96.czi'
filename = r"/datadisk1/tuxedo/testpictures/Testdata_Zeiss/celldivision/CellDivision_T=10_Z=15_CH=2_DCV_small.czi"
#filename = r"/datadisk1/tuxedo/testpictures/Testdata_Zeiss/wellplate/testwell96_woatt_S1-5.czi"
#filename = r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/CZI_Testfiles/S=1_3x3_T=1_Z=1_CH=2.czi'
#filename = r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/CZI_Testfiles/S=2_3x3_T=3_Z=4_CH=2.czi'
#filename = r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/CZI_Testfiles/S=2_3x3_T=1_Z=4_CH=2.czi'
#filename = r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/CZI_Testfiles/S=2_3x3_T=3_Z=1_CH=2.czi'
#filename = r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/CZI_Testfiles/S=2_3x3_T=3_Z=4_CH=2.czi'
#filename = r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/CZI_Testfiles/S=1_HE_Slide_RGB.czi'
#filename = r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/CZI_Testfiles/Multiscene_CZI_3Scenes.czi'
#filename = r'c:\Users\m1srh\Downloads\Overview.czi'
#filename = r'd:\Testdata_Zeiss\LatticeLightSheet\LS_Mitosis_T=150-300.czi'
#filename = r"D:\Testdata_Zeiss\CD7\96well_CD7_25x_Lam488_Act647_CO2Y_5x5_subset_10wells_16z.czi"
filename = r"D:\Testdata_Zeiss\CD7\fixed endpoint 3C 20 384well.czi"

# get the metadata as a dictionary
md, md_add = czt.get_metadata_czi(filename)

"""
print('---------------------------------------------------------')
print('Number of Scenes : ', len(md['bbox_all_scenes']))
for k, v in md['bbox_all_scenes'].items():
    print(k, v.w, v.h, v.x, v.y)
if md['isMosaic']:
    print('---------------------------------------------------------')
    print('Number of Scenes : ', len(md['bbox_all_mosaic_scenes']))
    #for k, v in md['bbox_all_mosaic_scenes'].items():
    #    print(k, v.w, v.h, v.x, v.y)
    print('---------------------------------------------------------')
    print('Number of Tiles : ', len(md['bbox_all_mosaic_tiles']))
    #for k, v in md['bbox_all_mosaic_tiles'].items():
    #    print(k.m_index, k.dimension_coordinates, v.w, v.h, v.x, v.y)
    print('---------------------------------------------------------')
    print('Number of Tiles : ', len(md['bbox_all_tiles']))
    #for k, v in md['bbox_all_tiles'].items():
    #    print(k.m_index, k.dimension_coordinates, v.w, v.h, v.x, v.y)
df = czt.md2dataframe(md, paramcol='Parameter', keycol='Value')
print(df)
# open the CZI file reading image data
czi = CziFile(filename)
sceneindex = 0
sc = czt.CZIScene(czi, md, sceneindex)
print(sc.__dict__)

array_size_all_scenes, shape_single_scenes, same_shape = czt.get_shape_allscenes(czi, md)

[img, out] = czi.read_image(S=0)
print('Shape Numpy Array : ', img.shape)
print(out)

img_sc = czt.read_czi_scene(czi, sc, md,
                            scalefactor=1.0,
                            array_type='numpy')
print('Shape Single Scene : ', img_sc.shape)

napari.run()


"""
print('-------------   AICSImageIO   --------------------------------------------')

# test using AICSImageIO 4.0.2 and aicspylibczi 3.0.2
aics_img = AICSImage(filename)
print(aics_img.shape)
for k,v in aics_img.dims.items():
    print(k,v)

# get the stack as dask array
stack = czt.get_daskstack(aics_img)

# start the napari viewer and show the image
viewer = napari.Viewer()
layerlist = nap.show_napari(viewer, stack, md,
                            blending='additive',
                            adjust_contrast=False,
                            gamma=0.85,
                            add_mdtable=True,
                            rename_sliders=True)

viewer.run()


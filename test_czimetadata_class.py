from czifiletools import czimetadata as czimd
import napari
from czifiletools import napari_czitools as nap
from aicspylibczi import CziFile

#filename = r"C:\Testdata_Zeiss\CZI_Testfiles\CellDivision_T=10_Z=20_CH=1_DCV.czi"
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=3_CH=2.czi'
#filename = r"/datadisk1/tuxedo/testpictures/Testdata_Zeiss/celldivision/CellDivision_T=15_Z=20_CH=2_DCV.czi"
#filename = r"d:\Testdata_Zeiss\CZI_Testfiles\testwell96.czi"
#filename = r"/datadisk1/tuxedo/testpictures/Testdata_Zeiss/wellplate/testwell96.czi"
#filename = r"D:\Testdata_Zeiss\CZI_Testfiles\S=1_HE_Slide_RGB_small.czi"
#filename = r"D:\Testdata_Zeiss\CZI_Testfiles\OverviewScan_M=9_CH=3.czi"
#filename = r"d:\Testdata_Zeiss\CZI_Testfiles\DTScan_ID4.czi"
filename = r"d:\Testdata_Zeiss\CZI_Testfiles\OverviewScan.czi"
#filename = r"D:\Testdata_Zeiss\CZI_Testfiles\OverViewScan_2x3_1brain.czi"
#filename = r"D:\Testdata_Zeiss\CZI_Testfiles\OverViewScan_2x3_1brain2.czi"
#filename= r"C:\Users\m1srh\Downloads\strange.czi"

mdata = czimd.CZIMetadata(filename)
mdict = czimd.create_metadata_dict(filename)
for k,v in mdict.items():
    print(k, ' : ', v)

out = czimd.CZIMetadata.get_dtype_fromstring('gray16')
print(out)
scf = czimd.CziScaling.get_scale_ratio(scalex=mdict['XScale'],
                                       scaley=mdict['YScale'],
                                       scalez=mdict['ZScale'])
print("Scale Factors : ", scf)

# get as pandas dataframe

df = czimd.md2dataframe(mdict)
print(df[:10])

# write XML to disk
xmlfile = czimd.writexml_czi(filename)

"""
pt = czimd.Planes.get_planetable(filename,
                                 savetable=True,
                                 separator=',',
                                 index=True)

print(pt[:5])
"""

# show array inside napari viewer
viewer = napari.Viewer()

# specify the index of the Channel inside the array
aicsczi = CziFile(filename)
all_scenes = czimd.readczi(filename)
scene = czimd.CZIScene(aicsczi, 0)
cpos = scene.single_scene_dimstr.find('C')

layers = nap.show_napari(viewer, all_scenes, mdata,
                         blending='additive',
                         adjust_contrast=False,
                         gamma=0.85,
                         add_mdtable=True,
                         md_dict=mdict,
                         rename_sliders=True)

napari.run()
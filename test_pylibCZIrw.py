from aicsimageio import AICSImage
from aicspylibczi import CziFile
import pylibCZIrw_tools as cztrw
from pylibCZIrw import czi as pyczi
#from pylibCZIrw import Rectangle
from matplotlib import pyplot as plt


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
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\CellDivision_T=10_Z=20_CH=1_DCV.czi'
#filename = r'D:\Testdata_Zeiss\CZI_Testfiles\CellDivision_T=3_Z=5_CH=2_X=240_Y=170.czi'
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\S=1_HE_Slide_RGB.czi'
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\OverViewScan.czi'
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\DTScan_ID4.czi'
#filename = r'd:\Testdata_Zeiss\CZI_Testfiles\w96_A1+A2.czi'
filename = r'd:\Testdata_Zeiss\CZI_Testfiles\testwell96.czi'
#filename = r"D:\Testdata_Zeiss\Mitochondria_EM_with_DNN\original_data\mitochondria_train_01_seg_ov_small.czi"
#filename = r"/datadisk1/tuxedo/testpictures/Testdata_Zeiss/celldivision/CellDivision_T=10_Z=15_CH=2_DCV_small.czi"
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

# get all the metadata as a dictionary
md, md_add = cztrw.get_metadata_czi(filename, dim2none=True, convert_scunit=True)

# open the CZI document to read the
czidoc = pyczi.open_czi(filename)

# show all dimensions
total_bbox = czidoc.total_bounding_box
for k,v in total_bbox.items():
    print(k, v)

# get information about the scenes etc.
sc_bbox = czidoc.scenes_bounding_rectangle
total_rect = czidoc.total_bounding_rectangle
pixeltype_ch = czidoc.get_channel_pixel_type(0)
pixeltypes = czidoc.pixel_types
print('Real Pixeltypes in CZI file : ', pixeltypes)

# read a simple 2d image plane
roi = (300, 300, 300, 600)
image2d_C0 = czidoc.read(plane={'C': 0}, scene=0, roi=roi, pixel_type="Gray8")
image2d_C1 = czidoc.read(plane={'C': 1}, scene=0, roi=roi, pixel_type="Gray8")
print(image2d_C0.shape, image2d_C1.shape)
print('Pixeltype after conversion during reading : ', image2d_C0.dtype, image2d_C1.dtype)

# Create two subplots and unpack the output array immediately
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(image2d_C0, interpolation='nearest', cmap='Reds_r')
ax2.imshow(image2d_C1, interpolation='nearest', cmap='Greens_r')
plt.show()

# store metadata inside Pandas dataframe
mdf = cztrw.md2dataframe(md)
print(mdf)

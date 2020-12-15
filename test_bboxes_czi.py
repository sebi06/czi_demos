from aicspylibczi import CziFile
import imgfileutils as imf
import czi_tools as czt
import matplotlib.pyplot as plt
import numpy as np

# filename = r"testdata\Tumor_H+E_small2.czi"
filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\Well_B2-4_S=4_T=1_Z=1_C=1.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\W96_B2+B4_S=2_T=1=Z=1_C=1_Tile=5x9.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\W96_B2+B4_S=2_T=2=Z=4_C=3_Tile=5x9.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\Castor\Z-Stack_DCV\CellDivision_T=10_Z=15_CH=2_DCV_small.czi"
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\Castor\Mouse Kidney_40x0.95_3CD_JK_comp.czi"
# filename = r"C:\Temp\input\DTScan_ID4.czi"
# filename = r"C:\Temp\input\OverViewScan_8Brains.czi"

# get the metadata from the czi file
md, additional_mdczi = imf.get_metadata(filename)

# read CZI using aicslibczi
cziobject = CziFile(filename)
size = cziobject.read_mosaic_size()

dimsizes = imf.getdims_pylibczi(cziobject)


print('Mosaic Size: ', size)
print('-----------------------------------------------------')

# define channel to read
ch = 0
t = 0
z = 0
scalefactor = 0.5

# create figure later plotting
fig, ax = plt.subplots(1, 4, figsize=(16, 6))


# read sizes for all scenes
for s in range(md['SizeS']):

    """
    # get bbox for the specific scene
    xmin, ymin, width, height = czt.get_bbox_scene(cziobject, sceneindex=s)

    # read the specific part of the CZI
    scene = cziobject.read_mosaic(region=(xmin, ymin, width, height),
                                   scale_factor=scalefactor,
                                   T=t,
                                   Z=z,
                                   C=ch)
    """

    # get the
    scene, bbox, md = czt.read_scene_bbox(cziobject, md,
                                          sceneindex=s,
                                          channel=ch,
                                          timepoint=t,
                                          zplane=z,
                                          scalefactor=scalefactor)

    print('Scene Shape : ', s, bbox[0], bbox[1], bbox[2], bbox[3])
    print('BBox Scene  : ', s, scene.shape)
    print('-----------------------------------------------------')

    ax[s].imshow(np.squeeze(scene),
                 cmap=plt.cm.gray,
                 interpolation='nearest',
                 clim=[scene.min(), scene.max() * 0.5])

    #ax[s].set_title('Scene : ' + str(s), fontsize=12)
    ax[s].set_title('BBox Scene: ' + str(s) + str(np.squeeze(scene).shape), fontsize=12)

del cziobject

plt.show()

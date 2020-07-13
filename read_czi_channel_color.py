import czifile as zis
import numpy as np

filename_czi = r'E:\tuxedo\testpictures\Testdata_Zeiss\celldivision\CellDivision_T=10_Z=15_CH=2_DCV_small.czi'


# get CZI object and read array
czi = zis.CziFile(filename_czi)

# parse the XML into a dictionary
channels_colors = []
metadatadict_czi = czi.metadata(raw=False)
sizeC = np.int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeC'])

if sizeC == 1:
    channels_colors.append(metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
                                  ['Channels']['Channel']['Color'])

if sizeC > 1:
    for ch in range(sizeC):
        channels_colors.append(metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
                                  ['Channels']['Channel'][ch]['Color'])

czi.close()

print(channels_colors)

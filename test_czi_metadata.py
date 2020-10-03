import imgfileutils as imf

#filename = r"C:\Users\m1srh\Downloads\sub-6368_20x_MsGcg_RbCol4_SMACy3_islet1.czi"
#filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\Castor\Z-Stack_DCV\CellDivision_T=10_Z=15_CH=2_DCV_small.czi"
filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\Experiment-09.czi"


# get the metadata from the czi file
md, additional_mdczi = imf.get_metadata(filename)

# show some metainformation
print('------------------   Show Metainformation ------------------')

# shape and dimension entry from CZI file as returned by czifile.py
# be aware of the difference bewteen czifile.py and AICSImageIO
print('Array Shape (czifile)          : ', md['Shape_czifile'])
print('Dimension Entry (czifile)      : ', md['Axes_czifile'])
print('Array Shape (aicsimageio)      : ', md['Shape_aics'])
print('Dimension Entry (aicsimageio)  : ', md['Axes_aics'])
print('------------------------------------------------------------')
print('SizeS : ', md['SizeS'])
print('SizeT : ', md['SizeT'])
print('SizeZ : ', md['SizeZ'])
print('SizeC : ', md['SizeC'])
print('SizeX (czifile) : ', md['SizeX'])
print('SizeY (czifile) : ', md['SizeY'])
print('SizeY (aicsimageio) : ', md['SizeX_aics'])
print('SizeY (aicsimageio) : ', md['SizeY_aics'])
print('XScale : ', md['XScale'])
print('Yscale : ', md['YScale'])
print('Zscale : ', md['ZScale'])

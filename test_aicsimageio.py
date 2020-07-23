from aicsimageio import AICSImage, imread, imread_dask
import czifile as zis

filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\Atomic\Nuclei\nuclei_RGB\H&E\Tumor_H&E_uncompressed_TSeries_3.czi"

# get CZI object
czi = zis.CziFile(filename)
# add axes and shape information using czifile package
print(czi.axes)
print(czi.shape)

# add axes and shape information using aicsimageio package
czi_aics = AICSImage(filename)
print(czi_aics.dims)
print(czi_aics.shape)
print(czi_aics.size_x)
print(czi_aics.size_y)
print(czi_aics.size_c)
print(czi_aics.size_t)
print(czi_aics.size_t)
print(czi_aics.size_s)

czi.close()
czi_aics.close()

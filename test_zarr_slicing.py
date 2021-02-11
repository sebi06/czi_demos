import zarr
import numpy as np
import imgfile_tools as imf

filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\S=2_3x3_T=1_Z=1_CH=2.czi"
# get the metadata from the czi file
md, additional_mdczi = imf.get_metadata(filename)


shape = [2, 3, 5, 2, 10, 20]

np_array = np.ones(shape, dtype=np.int16)
print("Shape NP Array:", np_array.shape)

z_array = zarr.create(shape)
print("Shape ZARR Array:", z_array.shape)

print('Is ZARR array: ', isinstance(z_array, zarr.Array))

s0 = slice(shape[0])
s1 = slice(shape[1])
s2 = slice(shape[2])
s3 = slice(1)
#s4 = slice(shape[4])
s4 = slice(None)
#s5 = slice(shape[5])
s5 = slice(None)

sliceNd = (s0, s1, s2, s3, s4, s5)

z_array_sliced = z_array[sliceNd]
print("Shape ZARR Array Sliced:", z_array_sliced.shape)

#array.take(ch, axis=dimpos['C'])


def slicedim_zarr(array, dimindex, posdim):

    if posdim == 0:
        array_sliced = array[dimindex, :, :, :, :, :]
    if posdim == 1:
        array_sliced = array[:, dimindex, :, :, :, :]
    if posdim == 2:
        array_sliced = array[:, :, dimindex, :, :, :]
    if posdim == 3:
        array_sliced = array[:, :, :, dimindex, :, :]

    return array_sliced

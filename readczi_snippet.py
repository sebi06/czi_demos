import imgfileutils as imf
from aicsimageio import AICSImage, imread

# get the metadata from the czi file
filename = 'myimage.czi'
md = imf.get_metadata_czi(filename, dim2none=False)
readmethod = 'perscene'

# get the stack using the specified method
if readmethod == 'chunked':
    img = AICSImage(filename, chunk_by_dims=["S"])
    stack = img.get_image_data()

if readmethod == 'chunked_dask':
    img = AICSImage(filename, chunk_by_dims=["S"])
    stack = img.get_image_dask_data()

if readmethod == 'fullstack':
    img = AICSImage(filename)
    stack = img.get_image_data()

for s in range(md['SizeS']):
    for t in range(md['SizeT']):
        for z in range(md['SizeZ']):
            for ch in range(md['SizeZ']):

                if readmethod == 'chunked_dask':
                    image2d = stack[s, t, z, ch, :, :].compute()

                if readmethod == 'fullstack' or readmethod == 'chunked':
                    image2d = stack[s, t, z, ch, :, :]

                if readmethod == 'perscene':
                    image2d = img.get_image_data("YX", S=s, T=t, Z=z, C=ch)

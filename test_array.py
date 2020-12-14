import numpy as np
import imgfileutils as imf
from aicspylibczi import CziFile


def get_dimorder(dimstring):
    """Get the order of dimensions from dimension string

    :param dimstring: string containing the dimensions
    :type dimstring: str
    :return: dims_dict - dictionary with the dimensions and its positions
    :rtype: dict
    :return: dimindex_list - list with indices of dimensions
    :rtype: list
    :return: numvalid_dims - number of valid dimensions
    :rtype: integer
    """

    dimindex_list = []
    dims = ['R', 'I', 'M', 'H', 'V', 'B', 'S', 'T', 'C', 'Z', 'Y', 'X', '0']
    dims_dict = {}

    for d in dims:

        dims_dict[d] = dimstring.find(d)
        dimindex_list.append(dimstring.find(d))

    numvalid_dims = sum(i > 0 for i in dimindex_list)

    return dims_dict, dimindex_list, numvalid_dims


filename = r"C:\Temp\input\DTScan_ID4.czi"

md, addmd = imf.get_metadata(filename)

czi = CziFile(filename)

# Get the shape of the data, the coordinate pairs are (start index, size)
dimensions = czi.dims_shape()
print(dimensions)
print(czi.dims)
print(czi.size)
print(czi.is_mosaic())  # True
# Mosaic files ignore the S dimension and use an internal mIndex to reconstruct, the scale factor allows one to generate a manageable image
mosaic_data = czi.read_mosaic(C=0, scale_factor=1)
print('CZI Mosaic Data Shape : ', mosaic_data.shape)



md = {}
md['SizeS'] = 1
md['SizeT'] = 3
md['SizeZ'] = 5
md['SizeC'] = 2
md['SizeY'] = 100
md['SizeX'] = 200


dimorder = 'STCYX'

dims_dict, dimindex_list, numvalid_dims = get_dimorder(dimorder)

new = {k: v for k, v in dims_dict.items() if v != -1}
new = {value: key for key, value in new.items()}
print(new)


ar = np.array(np.zeros([md['SizeY'], md['SizeX']]))

out = np.resize(3, (ar.shape[0], ar.shape[1]))


# in case of 2 dimensions
if dimorder == 'YX':
    ar = np.array(np.zeros([md['SizeY'], md['SizeX']]))

# in case of 3 dimensions

if dimorder == 'SYX':
    ar = np.array(np.zeros([md['SizeS'], md['SizeY'], md['SizeX']]))

if dimorder == 'TYX':
    ar = np.array(np.zeros([md['SizeT'], md['SizeY'], md['SizeX']]))

if dimorder == 'ZYX':
    ar = np.array(np.zeros([md['SizeZ'], md['SizeY'], md['SizeX']]))

if dimorder == 'CYX':
    ar = np.array(np.zeros([md['SizeC'], md['SizeY'], md['SizeX']]))

# in case of 4 dimensions
if dimorder == 'SCYX':
    ar = np.array(np.zeros([md['SizeS'], md['SizeC'], md['SizeY'], md['SizeX']]))

if dimorder == 'STYX':
    ar = np.array(np.zeros([md['SizeS'], md['SizeC'], md['SizeY'], md['SizeX']]))

if dimorder == 'SZYX':
    ar = np.array(np.zeros([md['SizeS'], md['SizeC'], md['SizeY'], md['SizeX']]))

if dimorder == 'TCYX':
    ar = np.array(np.zeros([md['SizeS'], md['SizeC'], md['SizeY'], md['SizeX']]))

if dimorder == 'TZYX':
    ar = np.array(np.zeros([md['SizeS'], md['SizeC'], md['SizeY'], md['SizeX']]))

if dimorder == 'ZCYX':
    ar = np.array(np.zeros([md['SizeS'], md['SizeC'], md['SizeY'], md['SizeX']]))

if dimorder == 'ZTYX':
    ar = np.array(np.zeros([md['SizeS'], md['SizeC'], md['SizeY'], md['SizeX']]))

if dimorder == 'SCYX':
    ar = np.array(np.zeros([md['SizeS'], md['SizeC'], md['SizeY'], md['SizeX']]))

if dimorder == 'CTYX':
    ar = np.array(np.zeros([md['SizeS'], md['SizeC'], md['SizeY'], md['SizeX']]))

if dimorder == 'CZYX':
    ar = np.array(np.zeros([md['SizeS'], md['SizeC'], md['SizeY'], md['SizeX']]))


ar = np.array(np.zeros([md['SizeY'], md['SizeX']]))


print(dims_dict)

for d in range(0, 6):

    dim2search = dimorder[d]
    print(dim2search, dims_dict[dim2search])

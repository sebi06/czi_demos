import numpy as np
import zarr
import dask
import dask.array as da


shape = [2, 3, 5, 4, 20, 40]


nparray = np.empty(shape, dtype=np.int16)
print('Shape Numpy Array : ', nparray.shape)

darray = da.empty(shape, dtype=np.int16)
print('Shape Dask Array : ', darray.shape)

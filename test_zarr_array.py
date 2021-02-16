import numpy as np
import zarr
import dask
import dask.array as da

posC = 3
shape = [2, 3, 5, 4, 20, 40]
#shape = [3, 20, 40]

n = scene_array = np.ones(shape, dtype=np.int16)
print('Shape Numpy Array : ', n.shape)

# slice out 1st channel
n1 = n.take(0, axis=posC)
n2 = n[:, :, :, 0, :, :]
print('Shape Numpy Array (Sliced) - 1: ', n1.shape)
print('Shape Numpy Array (Sliced) - 2: ', n2.shape)


z = zarr.create(tuple(shape), dtype=np.int16)
print('Shape Zarr Array : ', z.shape)
z1 = z.oindex[[3], :]
print('Shape Zarr Array (Sliced): ', z1.shape)

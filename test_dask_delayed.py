from aicspylibczi import CziFile
import imgfileutils as imf
from aicsimageio import AICSImage
from skimage import data
import napari
import dask
import dask.array as da
from IPython.display import display, HTML
from dask import delayed

filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\LatticeLightSheet\LS_Mitosis_T=150-300.czi"

# get the metadata
md, addmd = imf.get_metadata(filename)
czi = CziFile(filename)
czi = AICSImage(filename)


def load_image(czi, t=0):

    zstack = czi.get_image_data("CZYX", S=0, T=t)
    #zstack = czi.read_image(T=t)
    return zstack


sp = [md['SizeC'], md['SizeZ'], md['SizeY'], md['SizeX']]

# create dask stack of lazy image readers
lazy_process_image = dask.delayed(load_image)  # lazy reader

lazy_arrays = [lazy_process_image(czi, t=t) for t in range(0, md['SizeT'])]

dask_arrays = [
    da.from_delayed(lazy_array, shape=sp, dtype=md['NumPy.dtype'])
    for lazy_array in lazy_arrays
]
# Stack into one large dask.array
dask_stack = da.stack(dask_arrays, axis=0)
print(dask_stack.shape)
dask_stack

# configure napari automatically based on metadata and show the OME-TIFF stack
layers = imf.show_napari(dask_stack, md,
                         blending='additive',
                         gamma=0.85,
                         add_mdtable=True,
                         rename_sliders=True)

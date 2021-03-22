from aicspylibczi import CziFile
from aicsimageio import AICSImage, imread, imread_dask
import imgfile_tools as imf
import czifile_tools as czt
import numpy as np
import zarr
import dask
import dask.array as da
from dask import delayed
from itertools import product
import segmentation_tools as sgt
import visu_tools as vst
from skimage import measure, segmentation
from skimage.measure import regionprops
from skimage.color import label2rgb
import pandas as pd
import tifffile
import progressbar

# filename = r"E:\tuxedo\testpictures\Testdata_Zeiss\BrainSlide\OverViewScan_analyzed.czi"
filename = r"C:\Users\m1srh\Downloads\Halo_CZI.czi"

# get the metadata from the czi file
md, additional_mdczi = imf.get_metadata(filename)

# toggle additional printed output
verbose = True

# define columns names for dataframe
cols = ['S', 'T', 'Z', 'C', 'Number']
objects = pd.DataFrame(columns=cols)

# optional dipslay of "some" results - empty list = no display
show_image = [0]

# threshold parameters - will be used depending on the segmentation method
filtermethod = 'median'
filtersize = 3
threshold = 'global_otsu'
# use watershed for splitting - ws or ws_adv
use_ws = False
ws_method = 'ws_adv'
min_distance = 5
radius_dilation = 1
chindex = 0
minsize = 1
maxsize = 10000000

# read the czi mosaic image
czi = CziFile(filename)
# Get the shape of the data
print('Dimensions   : ', czi.dims)
print('Size         : ', czi.size)
print('Shape        : ', czi.dims_shape())
print('IsMoasic     : ', czi.is_mosaic())


mosaic = czi.read_mosaic(C=0, scale_factor=0.2)

# read the mosaic pixel data
image2d = np.squeeze(mosaic, axis=0)
print('Mosaic Shape :', image2d.shape)

image_counter = 0
results = pd.DataFrame()
# create the savename for the OME-TIFF
savename = filename.split('.')[0] + '.ome.tiff'

# open the TiffWriter in order to save as Multi-Series OME-TIFF
with tifffile.TiffWriter(savename, append=False) as tif:

    for s in progressbar.progressbar(range(md['SizeS']), redirect_stdout=True):

        values = {'S': s,
                  'C': chindex,
                  'Number': 0}

        # segment the image
        mask = sgt.segment_threshold(image2d, filtermethod=filtermethod,
                                     filtersize=filtersize,
                                     threshold=threshold,
                                     split_ws=use_ws,
                                     min_distance=min_distance,
                                     ws_method=ws_method,
                                     radius=radius_dilation)

        # clear the border
        mask = segmentation.clear_border(mask)

        # measure region properties
        to_measure = ('label',
                      'area',
                      'centroid',
                      'bbox')

        # measure the specified parameters store in dataframe
        props = pd.DataFrame(
            measure.regionprops_table(
                mask,
                intensity_image=image2d,
                properties=to_measure
            )
        ).set_index('label')

        # filter objects by size
        #props = props[(props['area'] >= minsize) & (props['area'] <= maxsize)]

        # add well information for CZI metadata
        try:
            props['WellId'] = md['Well_ArrayNames'][s]
            props['Well_ColId'] = md['Well_ColId'][s]
            props['Well_RowId'] = md['Well_RowId'][s]
        except (IndexError, KeyError) as error:
            # Output expected ImportErrors.
            print('Key not found:', error)
            print('Well Information not found. Using S-Index.')
            props['WellId'] = s
            props['Well_ColId'] = s
            props['Well_RowId'] = s

        # add plane indices
        props['S'] = s
        props['C'] = chindex

        # count the number of objects
        values['Number'] = props.shape[0]

        # update dataframe containing the number of objects
        objects = objects.append(pd.DataFrame(values, index=[0]), ignore_index=True)
        results = results.append(props, ignore_index=True)

        image_counter += 1
        # optional display of results
        if image_counter - 1 in show_image:
            print('Well:', props['WellId'].iloc[0], 'Index S-C:', s, chindex, 'Objects:', values['Number'])

            ax = vst.plot_segresults(image2d, mask, props,
                                     add_bbox=True)

        """
        # write scene as OME-TIFF series
        tif.save(mask,
                 photometric='minisblack',
                 metadata={'axes': 'YX',
                           'PhysicalSizeX': np.round(md['XScale'], 3),
                           'PhysicalSizeXUnit': md['XScaleUnit'],
                           'PhysicalSizeY': np.round(md['YScale'], 3),
                           'PhysicalSizeYUnit': md['YScaleUnit'],
                           'PhysicalSizeZ': np.round(md['ZScale'], 3),
                           'PhysicalSizeZUnit': md['ZScaleUnit']
                           }
                 )
        """

# rename colums in pandas datatable
results.rename(columns={'bbox-0': 'ystart',
                        'bbox-1': 'xstart',
                        'bbox-2': 'yend',
                        'bbox-3': 'xend'},
               inplace=True)

# create ne colums
results['bbox_width'] = results['xend'] - results['xstart']
results['bbox_height'] = results['yend'] - results['ystart']

print(objects)
print(results)
print('Done.')

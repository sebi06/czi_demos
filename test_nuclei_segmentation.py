# -*- coding: utf-8 -*-

#################################################################
# File        : test_nuclei_segmentation.py
# Version     : 0.6
# Author      : czsrh
# Date        : 20.10.2020
# Institution : Carl Zeiss Microscopy GmbH
#
# Copyright (c) 2020 Carl Zeiss AG, Germany. All Rights Reserved.
#################################################################

import sys
from time import process_time, perf_counter
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imgfileutils as imf
import segmentation_tools as sgt
import visutools as vst
from aicsimageio import AICSImage, imread
from scipy import ndimage
from skimage import measure, segmentation
from skimage.measure import regionprops
from skimage.color import label2rgb
from MightyMosaic import MightyMosaic
import progressbar
import shutil
import tifffile
import itertools as it


# select plotting backend
plt.switch_backend('Qt5Agg')
verbose = False


###############################################################################
# filename = r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/wellplate/testwell96.czi'
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\Castor\testwell96.czi"
# filename = r'WP384_4Pos_B4-10_DAPI.czi'
# filename = r'nuctest01.ome.tiff'
# filename = 'A01.czi'
# filename = 'testwell96_A9_1024x1024_Nuc.czi'
# filename = r'/datadisk1/tuxedo/temp/input/Osteosarcoma_01.czi'
# filename = r'c:\Temp\input\Osteosarcoma_02.czi'
# filename = r'c:\Temp\input\well96_DAPI.czi'
#filename = r"C:\Temp\input\WP96_4Pos_B4-10_DAPI.czi"
# filename = r'c:\Temp\input\Translocation_comb_96_5ms.czi'
# filename = r'C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\Atomic\Nuclei\nuclei_RGB\H&E\Tumor_H&E_small2.czi'
filename = r"testdata/WP96_4Pos_B4-10_DAPI.czi"
#filename = r'testdata/WP96_2Pos_B2+B4_S=2_T=2_Z=4_C=3_X=512_Y=256.czi'
#filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\Atomic\Nuclei\nuclei_RGB\H+E\Tumor_H+E_uncompressed_TSeries_cleaned.czi"

# create the savename for the OME-TIFF
savename = filename.split('.')[0] + '.ome.tiff'

# get the metadata
md, additional_mdczi = imf.get_metadata(filename)

# define platetype and get number of rows and columns
show_heatmap = False
if show_heatmap:
    platetype = 96
    nr, nc = vst.getrowandcolumn(platetype=platetype)

chindex = 0  # channel containing the objects, e.g. the nuclei
minsize = 100  # minimum object size [pixel]
maxsize = 5000  # maximum object size [pixel]

# define cutout size for subimage
cutimage = False
startx = 500
starty = 500
width = 640
height = 640

# define columns names for dataframe
cols = ['S', 'T', 'Z', 'C', 'Number']
objects = pd.DataFrame(columns=cols)

# optional dipslay of "some" results - empty list = no display
show_image = [0]

# toggle additional printed output
verbose = True

# threshold parameters - will be used depending on the choice for the segmentation method
filtermethod = 'median'
# filtermethod = None
filtersize = 3
threshold = 'global_otsu'
# threshold = 'triangle'

# use watershed for splitting - ws or ws_adv
use_ws = False
ws_method = 'ws_adv'
min_distance = 5
radius_dilation = 1

# define segmentation method
# use_method = 'scikit'
use_method = 'cellpose'
# use_method = 'zentf'
# use_method = 'stardist2d'

#######################################################

if use_method == 'stardist2d':

    # load pretrained model
    # define StarDist 2D model for nucleus detection
    # 'Versatile (fluorescent nuclei)'
    # 'Versatile (H&E nuclei)' - Not supported by this script yet !
    # 'DSB 2018 (from StarDist 2D paper)'

    # define and load the stardist model
    if not md['czi_isRGB']:
        sdmodel = sgt.load_stardistmodel(modeltype='Versatile (fluorescent nuclei)')
    if md['czi_isRGB']:
        sdmodel = sgt.load_stardistmodel(modeltype='Versatile (H&E nuclei)')

    # define the model parameters
    print('Setting model parameters.')
    stardist_prob_thresh = 0.5
    stardist_overlap_thresh = 0.3
    stardist_norm = True
    stardist_norm_pmin = 1
    stardist_norm_pmax = 99.8
    stardist_norm_clip = False


# load the ML model from cellpose when needed
if use_method == 'cellpose':

    model = sgt.load_cellpose_model(model_type='nuclei')

    # define list of channels for cellpose
    # channels = SizeS * SizeT * SizeZ * [0, 0]
    channels = [0, 0]  # when applying it to a single image
    diameter = 30

# define model path and load TF2 model when needed
if use_method == 'zentf':

    # define tile overlap factor for MightyMosaic
    overlapfactor = 1

    # Load the model
    MODEL_FOLDER = 'model_folder'
    model, tile_height, tile_width = sgt.load_tfmodel(modelfolder=MODEL_FOLDER)
    print('ZEN TF Model Tile Dimension : ', tile_height, tile_width)

###########################################################################

# start the timer for the total pipeline
startp = perf_counter()
readtime_allscenes = 0

# set number of Scenes for testing
md['SizeS'] = 5

# get AICSImageIO object using the python wrapper for libCZI (if file is CZI)
img = AICSImage(filename)


dims_dict, dimindex_list, numvalid_dims = imf.get_dimorder(md['Axes_aics'])
try:
    shape5d = list(md['Shape_aics'])
    shape5d.pop(dims_dict['S'])
except TypeError as e:
    print(e)
    shape5d = list(md['size_aicspylibczi'])

# create image5d for the current scene
image5d = np.zeros(shape5d, dtype=md['NumPy.dtype'])

# set channel dimension = 1 because we are only interested in the nuclei
shape5d[dims_dict['C'] - 1] = 1

# remove the S dimension from the dimstring
dimstring5d = md['Axes_aics'].replace('S', '')

# readmethod: fullstack, chunked, chunked_dask, perscene
readmethod = 'perscene'

if readmethod == 'chunked':
    # start the timer
    start = process_time()
    img = AICSImage(filename, chunk_by_dims=["S"])
    stack = img.get_image_data()
    end = process_time()
    # if verbose:
    print('Runtime CZI Reading using method: ', readmethod, end - start)

if readmethod == 'chunked_dask':
    # start the timer
    start = process_time()
    img = AICSImage(filename, chunk_by_dims=["S"])
    stack = img.get_image_dask_data()
    end = process_time()
    if verbose:
        print('Runtime CZI Reading using method: ', readmethod, str(end - start))

if readmethod == 'fullstack':
    # start the timer
    start = process_time()
    img = AICSImage(filename)
    stack = img.get_image_data()
    end = process_time()
    if verbose:
        print('Runtime CZI Reading using method: ', readmethod, str(end - start))

image_counter = 0
results = pd.DataFrame()

# open the TiffWriter in order to save as Multi-Series OME-TIFF
with tifffile.TiffWriter(savename, append=False) as tif:

    for s in progressbar.progressbar(range(md['SizeS']), redirect_stdout=True):
        for t in range(md['SizeT']):
            for z in range(md['SizeZ']):

                values = {'S': s,
                          'T': t,
                          'Z': z,
                          'C': chindex,
                          'Number': 0}

                if verbose:
                    print('Analyzing S-T-Z-C: ', s, t, z, chindex)

                if readmethod == 'chunked_dask':
                    # start the timer
                    start = perf_counter()
                    image2d = stack[s, t, z, chindex, :, :].compute()
                    end = perf_counter()
                    readtime_allscenes = readtime_allscenes + (end - start)

                if readmethod == 'fullstack' or readmethod == 'chunked':
                    image2d = stack[s, t, z, chindex, :, :]

                if readmethod == 'perscene':
                    # start the timer
                    start = perf_counter()
                    image2d = img.get_image_data("YX",
                                                 S=s,
                                                 T=t,
                                                 Z=z,
                                                 C=chindex)
                    end = perf_counter()
                    readtime_allscenes = readtime_allscenes + (end - start)

                # cutout subimage
                if cutimage:
                    image2d = sgt.cutout_subimage(image2d,
                                                  startx=startx,
                                                  starty=starty,
                                                  width=width,
                                                  height=height)

                if use_method == 'cellpose':
                    # get the mask for the current image
                    mask = sgt.segment_nuclei_cellpose2d(image2d, model,
                                                         rescale=None,
                                                         channels=channels,
                                                         diameter=diameter,
                                                         verbose=True,
                                                         autotune=False)

                if use_method == 'scikit':
                    mask = sgt.segment_threshold(image2d,
                                                 filtermethod=filtermethod,
                                                 filtersize=filtersize,
                                                 threshold=threshold,
                                                 split_ws=use_ws,
                                                 min_distance=min_distance,
                                                 ws_method=ws_method,
                                                 radius=radius_dilation)

                if use_method == 'zentf':

                    classlabel = 1

                    # check if tiling is required
                    if image2d.shape[0] > tile_height or image2d.shape[1] > tile_width:
                        binary = sgt.segment_zentf_tiling(image2d, model,
                                                          tilesize=tile_height,
                                                          classlabel=classlabel,
                                                          overlap_factor=overlapfactor)

                    elif image2d.shape[0] == tile_height and image2d.shape[1] == tile_width:
                        if verbose:
                            print('No Tiling or padding required')
                        binary = sgt.segment_zentf(image2d, model,
                                                   classlabel=classlabel)

                    elif image2d.shape[0] < tile_height or image2d.shape[1] < tile_width:

                        # do padding
                        image2d_padded, pad = sgt.add_padding(image2d,
                                                              input_height=tile_height,
                                                              input_width=tile_width)

                        # run prediction on padded image
                        binary_padded = sgt.segment_zentf(image2d_padded, model,
                                                          classlabel=classlabel)

                        # remove padding from result
                        binary = binary_padded[pad[0]:tile_height - pad[1], pad[2]:tile_width - pad[3]]

                    # apply watershed
                    if use_ws:
                        if ws_method == 'ws':
                            mask = sgt.apply_watershed(binary,
                                                       min_distance=min_distance)
                        if ws_method == 'ws_adv':
                            mask = sgt.apply_watershed_adv(image2d, binary,
                                                           min_distance=min_distance,
                                                           radius=radius_dilation)
                    if not use_ws:
                        # label the objects
                        mask, num_features = ndimage.label(binary)
                        mask = mask.astype(np.int)

                if use_method == 'stardist2d':

                    mask = sgt.segment_nuclei_stardist(image2d, sdmodel,
                                                       prob_thresh=stardist_prob_thresh,
                                                       overlap_thresh=stardist_overlap_thresh,
                                                       norm=stardist_norm,
                                                       norm_pmin=stardist_norm_pmin,
                                                       norm_pmax=stardist_norm_pmax,
                                                       norm_clip=stardist_norm_clip)

                # clear the border
                mask = segmentation.clear_border(mask)

                # add mask to the label5d stack
                #label5d[t, z, 0, :, :] = mask

                # update the 5d stack
                image5d = imf.update5dstack(image5d, mask,
                                            dimstring5d=dimstring5d,
                                            t=t,
                                            z=z,
                                            c=chindex)

                # measure region properties
                to_measure = ('label',
                              'area',
                              'centroid',
                              'max_intensity',
                              'mean_intensity',
                              'min_intensity',
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
                props = props[(props['area'] >= minsize) & (props['area'] <= maxsize)]
                # props = [r for r in props if r.area >= minsize]

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
                    show_heatmap = False

                # add plane indices
                props['S'] = s
                props['T'] = t
                props['Z'] = z
                props['C'] = chindex

                # count the number of objects
                values['Number'] = props.shape[0]
                # values['Number'] = len(regions) - 1
                if verbose:
                    print('Well:', props['WellId'].iloc[0], ' Objects: ', values['Number'])

                # update dataframe containing the number of objects
                objects = objects.append(pd.DataFrame(values, index=[0]),
                                         ignore_index=True)

                results = results.append(props, ignore_index=True)

                image_counter += 1
                # optional display of results
                if image_counter - 1 in show_image:
                    print('Well:', props['WellId'].iloc[0],
                          'Index S-T-Z-C:', s, t, z, chindex,
                          'Objects:', values['Number'])
                    ax = vst.plot_segresults(image2d, mask, props,
                                             add_bbox=True)

        # write scene as OME-TIFF series
        tif.save(image5d,
                 photometric='minisblack',
                 metadata={'axes': dimstring5d,
                           'PhysicalSizeX': np.round(md['XScale'], 3),
                           'PhysicalSizeXUnit': md['XScaleUnit'],
                           'PhysicalSizeY': np.round(md['YScale'], 3),
                           'PhysicalSizeYUnit': md['YScaleUnit'],
                           'PhysicalSizeZ': np.round(md['ZScale'], 3),
                           'PhysicalSizeZUnit': md['ZScaleUnit']
                           }
                 )
    # close the AICSImage object at the end
    img.close()

if verbose and readmethod == 'perscene':
    print('Total time CZI Reading using method: ', readmethod, readtime_allscenes)

# reorder dataframe with single objects
new_order = list(results.columns[-7:]) + list(results.columns[:-7])
results = results.reindex(columns=new_order)


# get the end time for the total pipeline
if verbose:
    endp = perf_counter()
    print('Runtime Segmentation Pipeline : ' + str(endp - startp))

# optional display of a heatmap
if show_heatmap:

    # create heatmap array with NaNs
    heatmap_numobj = vst.create_heatmap(platetype=platetype)
    heatmap_param = vst.create_heatmap(platetype=platetype)

    for well in md['WellCounter']:
        # extract all entries for specific well
        well_results = results.loc[results['WellId'] == well]

        # get the descriptive statistics for specific well
        stats = well_results.describe(include='all')

        # get the column an row indices for specific well
        col = np.int(stats['Well_ColId']['mean'])
        row = np.int(stats['Well_RowId']['mean'])

        # add value for number of objects to heatmap_numobj, e.g. 'count'
        heatmap_numobj[row - 1, col - 1] = stats['WellId']['count']

        # add value for specific params to heatmap, e.g. 'area'
        heatmap_param[row - 1, col - 1] = stats['area']['mean']

    df_numobjects = vst.convert_array_to_heatmap(heatmap_numobj, nr, nc)
    df_params = vst.convert_array_to_heatmap(heatmap_param, nr, nc)

    # define parameter to display a single heatmap
    parameter2display = 'ObjectNumbers'
    # parameter2display = 'Area'
    # colormap = 'YlGnBu'
    colormap = 'cividis_r'

    # show the heatmap for a single parameter
    # use 'df_numobjects' or 'df_params' here
    savename_single = vst.showheatmap(df_numobjects, parameter2display,
                                      fontsize_title=16,
                                      fontsize_label=16,
                                      colormap=colormap,
                                      linecolor='black',
                                      linewidth=3.0,
                                      save=False,
                                      filename=filename,
                                      dpi=100)

    print(objects)
    print(results[:5])

    plt.show()

print('Done')

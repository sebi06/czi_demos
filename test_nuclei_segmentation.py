# -*- coding: utf-8 -*-

#################################################################
# File        : test_nuclei_segmentation.py
# Version     : 0.5
# Author      : czsrh
# Date        : 20.04.2020
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
from czitools import imgfileutils as imf
from czitools import segmentation_tools as sgt
from czitools import visutools as vst
from aicsimageio import AICSImage, imread
from scipy import ndimage
from skimage import measure, segmentation
from skimage.measure import regionprops
from skimage.color import label2rgb
from MightyMosaic import MightyMosaic
import progressbar
import shutil


# select plotting backend
plt.switch_backend('Qt5Agg')
verbose = False


def save_scene(savefolder, imagefile, scene, label5d, metadata, correct_ome=True):

    print('Saving Scene: ', scene, 'as 5D OME-TIFF stack')

    # save stack for every scene
    sid = imf.addzeros(s)
    # keep in mind to use the "pure" filename because inside the container
    # writing to /input/... is forbidden
    name_scene = os.path.basename(imagefile).split('.')[0] + \
        '_S' + sid + '.' + 'ome.tiff'

    complete_savepath = os.path.join(savefolder, name_scene)

    # save file as OME-TIFF
    fs = imf.write_ometiff_aicsimageio(complete_savepath, label5d, md,
                                       reader='aicsimageio',
                                       overwrite=True)

    if correct_ome:
        old = ("2012-03", "2013-06", r"ome/2016-06")
        new = ("2016-06", "2016-06", r"OME/2016-06")
        imf.correct_omeheader(complete_savepath, old, new)

    print('Saved scene: ', complete_savepath)

    return complete_savepath


###############################################################################
# filename = r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/wellplate/testwell96.czi'
# filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\Castor\testwell96.czi"
# filename = r'WP384_4Pos_B4-10_DAPI.czi'
# filename = r'nuctest01.ome.tiff'
# filename = 'A01.czi'
# filename = 'testwell96_A9_1024x1024_Nuc.czi'
# filename = r'/datadisk1/tuxedo/temp/input/Osteosarcoma_01.czi'
# filename = r'c:\Temp\input\Osteosarcoma_02.czi'
#filename = r'c:\Temp\input\well96_DAPI.czi'
filename = r"C:\Temp\input\WP96_4Pos_B4-10_DAPI.czi"
# filename = r'c:\Temp\input\Translocation_comb_96_5ms.czi'
# filename = r'C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\Atomic\Nuclei\nuclei_RGB\H&E\Tumor_H&E_small2.czi'

# define platetype and get number of rows and columns
show_heatmap = False
if show_heatmap:
    platetype = 96
    nr, nc = vst.getrowandcolumn(platetype=platetype)

# save every scene as separate OME-TIFF
save_per_scene = True
if save_per_scene:
    resultfolder = os.path.basename(filename).split('.')[0] + '_Results'
    resultfolder = os.path.join(os.path.dirname(filename), resultfolder)
    # create result folder
    if os.path.exists(resultfolder):
        shutil.rmtree(resultfolder)

    # create folder
    try:
        os.mkdir(resultfolder)
    except OSError:
        print("Creation of the directory %s failed" % resultfolder)
    else:
        print("Successfully created the directory %s " % resultfolder)


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
# use_method = 'cellpose'
# use_method = 'zentf'
use_method = 'stardist2d'

#######################################################

if use_method == 'stardist2d':

    # load pretrained model
    # define StarDist 2D model for nucleus detection
    # 'Versatile (fluorescent nuclei)'
    # 'Versatile (H&E nuclei)' - Not supported by this script yet !
    # 'DSB 2018 (from StarDist 2D paper)'

    # define and load the stardist model
    sdmodel = sgt.load_stardistmodel(modeltype='Versatile (fluorescent nuclei)')

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

    # model = sgt.load_cellpose_model(model_type='nuclei', device=sgt.set_device())
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

# get the metadata
md, additional_mdczi = imf.get_metadata(filename, omeseries=0)

# set number of Scenes for testing
# md['SizeS'] = 1

# get AICSImageIO object using the python wrapper for libCZI (if file is CZI)
img = AICSImage(filename)
# SizeS = img.size_s
# SizeT = img.size_t
# SizeZ = img.size_z

dims_dict, dimindex_list, numvalid_dims = imf.get_dimorder(md['Axes_aics'])
shape_labelstack = list(md['Shape_aics'])
shape_labelstack.pop(dims_dict['S'])

# set channel dimension = 1 because we are only interested in the nuclei
shape_labelstack[dims_dict['C'] - 1] = 1

# create labelstack for the current scene
label5d = np.zeros(shape_labelstack, dtype=np.int16)

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

for s in progressbar.progressbar(range(md['SizeS']), redirect_stdout=True):
    # for s in range(md['SizeS']):
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
            label5d[t, z, 0, :, :] = mask

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
                ax = vst.plot_segresults(image2d, mask, props, add_bbox=True)

    # save the scene
    if save_per_scene:
        complete_savepath = save_scene(resultfolder, filename, s, label5d, md, correct_ome=True)


if verbose and readmethod == 'perscene':
    print('Total time CZI Reading using method: ', readmethod, readtime_allscenes)

# reorder dataframe with single objects
new_order = list(results.columns[-7:]) + list(results.columns[:-7])
results = results.reindex(columns=new_order)

# close the AICSImage object at the end
img.close()

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

    # print(objects)
    # print(results[:5])

    plt.show()

print('Done')

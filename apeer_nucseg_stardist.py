# -*- coding: utf-8 -*-

#################################################################
# File        : apeer_nucseg_stardist.py
# Version     : 0.1
# Author      : czsrh
# Date        : 04.05.2020
# Institution : Carl Zeiss Microscopy GmbH
#
# Copyright (c) 2020 Carl Zeiss AG, Germany. All Rights Reserved.
#################################################################

import sys
from time import process_time, perf_counter
import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import imgfileutils as imf
import ometifftools as ott
import segmentation_tools as sgt
from aicsimageio import AICSImage, imread
from scipy import ndimage
from skimage import measure, segmentation
from skimage.measure import regionprops
from skimage.color import label2rgb
from MightyMosaic import MightyMosaic
import progressbar
import tifffile

verbose = True
save_resultstack = True


def execute(imagefile,
            chindex_nucleus=0,
            sd_modelbasedir='stardist_models',
            sd_modelfolder='2D_versatile_fluo',
            prob_th=0.5,
            ov_th=0.3,
            minsize_nuc=20,
            maxsize_nuc=5000,
            norm=True,
            norm_pmin=1,
            norm_pmax=99.8,
            norm_clip=False,
            correct_ome=True):
    """Main function to run the stardist object segmentation and measure the object parameters.

    :param imagefile: filename of the current image
    :type imagefile: str
    :param chindex_nucleus: channel index with nucleus, defaults to 1
    :type chindex_nucleus: int, optional
    :param sd_modelbasedir: stardist model basedirectory, defaults to 'stardist_models'
    :type sd_modelbasedir: str, optional
    :param sd_modelfolder: stardist model folder inside basedirectory, defaults to '2D_versatile_fluo'
    :type sd_modelfolder: str, optional
    :param prob_th: probability threshold, defaults to 0.5
    :type prob_th: float, optional
    :param ov_th: overlap threshold, defaults to 0.3
    :type ov_th: float, optional
    :param minsize_nuc: minimum size of objects [pixel], defaults to 20
    :type minsize_nuc: int, optional
    :param maxsize_nuc:  maximum size of objects [pixel], defaults to 5000
    :type maxsize_nuc: int, optional
    :param norm: normalize images, defaults to True
    :type norm: bool, optional
    :param norm_pmin: minimum percentile for normalization, defaults to 1
    :type norm_pmin: int, optional
    :param norm_pmax: maximum percentile for normalization, defaults to 99.8
    :type norm_pmax: float, optional
    :param norm_clip: clipping for normalization, defaults to False
    :type norm_clip: bool, optional file
    :return: (obj_csv, objparams_csv, label_stacks) - filenames for data tables and label5d stack
    :rtype: tuple
    """

    # show current image path
    print('Current Image: ', imagefile)

    # check if file exits
    print('File : ', imagefile, ' exists: ', os.path.exists(imagefile))

    # get the metadata
    md, additional_mdczi = imf.get_metadata(imagefile, omeseries=0)

    # check the channel number for the nucleus
    if chindex_nucleus + 1 > md['SizeC']:
        print('Selected Channel for nucleus does not exit. Use channel = 1.')
        chindex_nucleus = 0

    # get AICSImageIO object using the python wrapper for libCZI
    img = AICSImage(imagefile)

    # define columns names for dataframe
    cols = ['S', 'T', 'Z', 'C', 'Number']
    objects = pd.DataFrame(columns=cols)
    results = pd.DataFrame()
    label_stacks = []

    # load the stardist model from web
    # sd_model = sgt.load_stardistmodel(modeltype=sd_modelname)

    # define basefolder for StarDist models
    sd_model = sgt.stardistmodel_from_folder(sd_modelbasedir,
                                             mdname=sd_modelfolder)

    dims_dict, dimindex_list, numvalid_dims = imf.get_dimorder(md['Axes_aics'])
    shape_labelstack = list(md['Shape_aics'])
    shape_labelstack.pop(dims_dict['S'])

    # set channel dimension = 1 because we are only interested in the nuclei
    shape_labelstack[dims_dict['C'] - 1] = 1

    # create labelstack for the current scene
    label5d = np.zeros(shape_labelstack, dtype=np.int16)

    for s in progressbar.progressbar(range(md['SizeS']), redirect_stdout=False):
        # for s in range(md['SizeS']):
        for t in range(md['SizeT']):
            for z in range(md['SizeZ']):

                values = {'S': s, 'T': t, 'Z': z, 'C': chindex_nucleus, 'Number': 0}

                # read a single 2d image
                image2d = img.get_image_data("YX", S=s, T=t, Z=z, C=chindex_nucleus)

                # get the segmented image using StarDist 2D
                mask = sgt.segment_nuclei_stardist(image2d, sd_model,
                                                   prob_thresh=prob_th,
                                                   overlap_thresh=ov_th,
                                                   norm=norm,
                                                   norm_pmin=norm_pmin,
                                                   norm_pmax=norm_pmax,
                                                   norm_clip=norm_clip)

                # clear border objects
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
                props = props[(props['area'] >= minsize_nuc) & (props['area'] <= maxsize_nuc)]

                # add well information for CZI metadata
                try:
                    props['WellId'] = md['Well_ArrayNames'][s]
                    props['Well_ColId'] = md['Well_ColId'][s]
                    props['Well_RowId'] = md['Well_RowId'][s]
                except (IndexError, KeyError) as error:
                    print('Error:', error)
                    print('Well Information not found. Using S-Index.')
                    props['WellId'] = s
                    props['Well_ColId'] = s
                    props['Well_RowId'] = s
                    show_heatmap = False

                # add plane indices
                props['S'] = s
                props['T'] = t
                props['Z'] = z
                props['C'] = chindex_nucleus

                # count the number of objects
                values['Number'] = props.shape[0]
                # values['Number'] = len(regions) - 1
                if verbose:
                    print('Well:', props['WellId'].iloc[0], ' Objects: ', values['Number'])

                # update dataframe containing the number of objects
                objects = objects.append(pd.DataFrame(values, index=[0]),
                                         ignore_index=True)

                results = results.append(props, ignore_index=True)

        # save stack for every scene
        sid = imf.addzeros(s)
        # keep in mind to use the "pure" filename because inside the container
        # writing to /input/... is forbidden
        name_scene = os.path.basename(imagefile).split('.')[0] + \
            '_S' + sid + '.' + 'ome.tiff'

        label_stacks.append(name_scene)

        if save_resultstack:
            # save file as OME-TIFF
            fs = ott.write_ometiff_aicsimageio(name_scene, label5d, md,
                                               reader='aicsimageio',
                                               overwrite=True)

            if correct_ome:
                old = ("2012-03", "2013-06", r"ome/2016-06")
                new = ("2016-06", "2016-06", r"OME/2016-06")
                ott.correct_omeheader(name_scene, old, new)

    # reorder dataframe with single objects
    new_order = list(results.columns[-7:]) + list(results.columns[:-7])
    results = results.reindex(columns=new_order)

    # close the AICSImage object at the end
    img.close()

    # save the results
    # get filename without extension
    basename_woext = os.path.basename(imagefile).split('.')[0]

    # define name for CSV tables
    obj_csv = basename_woext + '_obj.csv'
    objparams_csv = basename_woext + '_objparams.csv'

    # save the DataFrames as CSV tables
    objects.to_csv(obj_csv, index=False, header=True, decimal='.', sep=',')
    print('Saved Object Table as CSV :', obj_csv)
    results.to_csv(objparams_csv, index=False, header=True, decimal='.', sep=',')
    print('Saved Object Parameters Table as CSV :', objparams_csv)

    return (obj_csv, objparams_csv, label_stacks)


# Test Code locally
if __name__ == "__main__":

    # filename = r'input/A01.czi'
    # filename = 'input/Osteosarcoma_02.czi'
    # filename = r'input/WP384_4Pos_B4-10_DAPI.czi'
    #filename = r'c:\Temp\input\Osteosarcoma_02.czi'
    #filename = r'c:\Temp\input\well96_DAPI.czi'
    #filename = r'c:\Temp\input\Translocation_comb_96_5ms.czi'
    filename = r'c:\Temp\input\A01.czi'

    outputs = execute(filename,
                      chindex_nucleus=0,
                      sd_modelbasedir='stardist_models',
                      sd_modelfolder='2D_versatile_fluo',
                      prob_th=0.5,
                      ov_th=0.3,
                      minsize_nuc=20,
                      maxsize_nuc=5000,
                      norm=True,
                      norm_pmin=1,
                      norm_pmax=99.8,
                      norm_clip=False,
                      correct_ome=True)

    print(outputs[0])
    print(outputs[1])
    print(outputs[2])

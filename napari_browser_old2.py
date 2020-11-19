# -*- coding: utf-8 -*-

#################################################################
# File        : napari_browser_adv.py
# Version     : 0.0.1
# Author      : czsrh
# Date        : 18.11.2020
# Institution : Carl Zeiss Microscopy GmbH
#
# Copyright (c) 2020 Carl Zeiss AG, Germany. All Rights Reserved.
#################################################################

from PyQt5.QtWidgets import (
    QPushButton,
    QComboBox,
    QTabWidget,
    QHBoxLayout,
    QFileDialog,
    QDialogButtonBox,
    QWidget,
    QSlider,
)
from PyQt5.QtCore import Qt
import napari
import numpy as np

#from czitools import imgfileutils as imf
import imgfileutils as imf
from aicsimageio import AICSImage, imread
import dask.array as da
import os
from pathlib import Path


def add_napari(array, metadata,
               blending='additive',
               gamma=0.75,
               rename_sliders=False):
    """Show the multidimensional array using the Napari viewer

    :param array: multidimensional NumPy.Array containing the pixeldata
    :type array: NumPy.Array
    :param metadata: dictionary with CZI or OME-TIFF metadata
    :type metadata: dict
    :param blending: NapariViewer option for blending, defaults to 'additive'
    :type blending: str, optional
    :param gamma: NapariViewer value for Gamma, defaults to 0.85
    :type gamma: float, optional
    :param verbose: show additional output, defaults to True
    :type verbose: bool, optional
    :param rename_sliders: name slider with correct labels output, defaults to False
    :type verbose: bool, optional
    """

    # create scalefcator with all ones
    scalefactors = [1.0] * len(array.shape)
    dimpos = imf.get_dimpositions(metadata['Axes_aics'])

    # get the scalefactors from the metadata
    scalef = imf.get_scalefactor(metadata)

    # modify the tuple for the scales for napari
    scalefactors[dimpos['Z']] = scalef['zx']

    # remove C dimension from scalefactor
    scalefactors_ch = scalefactors.copy()
    del scalefactors_ch[dimpos['C']]

    if metadata['SizeC'] > 1:
        # add all channels as layers
        for ch in range(metadata['SizeC']):

            try:
                # get the channel name
                chname = metadata['Channels'][ch]
            except KeyError as e:
                print(e)
                # or use CH1 etc. as string for the name
                chname = 'CH' + str(ch + 1)

            # cut out channel
            # use dask if array is a dask.array
            if isinstance(array, da.Array):
                print('Extract Channel using Dask.Array')
                channel = array.compute().take(ch, axis=dimpos['C'])
                new_dimstring = metadata['Axes_aics'].replace('C', '')

            else:
                # use normal numpy if not
                print('Extract Channel NumPy.Array')
                channel = array.take(ch, axis=dimpos['C'])
                new_dimstring = metadata['Axes_aics'].replace('C', '')

            # actually show the image array
            print('Adding Channel  : ', chname)
            print('Shape Channel   : ', ch, channel.shape)
            print('Scaling Factors : ', scalefactors_ch)

            # get min-max values for initial scaling
            clim = imf.calc_scaling(channel,
                                    corr_min=1.0,
                                    offset_min=0,
                                    corr_max=0.85,
                                    offset_max=0)

            # add channel to napari viewer
            viewer.add_image(channel,
                             name=chname,
                             scale=scalefactors_ch,
                             contrast_limits=clim,
                             blending=blending,
                             gamma=gamma)

    if metadata['SizeC'] == 1:

        # just add one channel as a layer
        try:
            # get the channel name
            chname = metadata['Channels'][0]
        except KeyError:
            # or use CH1 etc. as string for the name
            chname = 'CH' + str(ch + 1)

        # actually show the image array
        print('Adding Channel: ', chname)
        print('Scaling Factors: ', scalefactors)

        # get min-max values for initial scaling
        clim = imf.calc_scaling(array)

        viewer.add_image(array,
                         name=chname,
                         scale=scalefactors,
                         contrast_limits=clim,
                         blending=blending,
                         gamma=gamma)

    if rename_sliders:

        print('Renaming the Sliders based on the Dimension String ....')

        # get the position of dimension entries after removing C dimension
        dimpos_viewer = imf.get_dimpositions(new_dimstring)

        # get the label of the sliders
        sliders = viewer.dims.axis_labels

        # update the labels with the correct dimension strings
        slidernames = ['B', 'S', 'T', 'Z']
        for s in slidernames:
            if dimpos_viewer[s] >= 0:
                sliders[dimpos_viewer[s]] = s
        # apply the new labels to the viewer
        viewer.dims.axis_labels = sliders


class Open_files(QWidget):

    def __init__(self):
        super(QWidget, self).__init__()
        self.layout = QHBoxLayout(self)
        self.file_dialog = QFileDialog()
        self.file_dialog.setWindowFlags(Qt.Widget)
        self.file_dialog.setModal(False)
        self.file_dialog.setOption(QFileDialog.DontUseNativeDialog)

        # Remove open and cancel button from widget
        self.buttonBox = self.file_dialog.findChild(QDialogButtonBox, "buttonBox")
        self.buttonBox.clear()

        # Only open following file types
        # self.file_dialog.setNameFilter("Images (*.czi *.nd2 *.tiff *.tif *.jpg *.png)")
        self.file_dialog.setNameFilter("Images (*.czi *.ome.tiff *ome.tif *.tiff *.tif)")
        self.layout.addWidget(self.file_dialog)
        self.file_dialog.currentChanged.connect(self.open_path)

    def open_path(self, path):

        if os.path.isfile(path):

            # remove exiting layers from napari
            viewer.layers.select_all()
            viewer.layers.remove_selected()

            # get the metadata
            md, addmd = imf.get_metadata(path)

            # get AICSImageIO object
            img = AICSImage(path)
            stack = img.get_image_data()

            add_napari(stack, md,
                       blending='additive',
                       gamma=0.85,
                       rename_sliders=True)


with napari.gui_qt():

    # create a viewer and add some images
    viewer = napari.Viewer()
    # add the gui to the viewer as a dock widget
    viewer.window.add_dock_widget(Open_files(), area="right")
    # viewer.window.add_dock_widget(Processsing())

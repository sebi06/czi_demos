# -*- coding: utf-8 -*-

#################################################################
# File        : napari_tools.py
# Version     : 0.0.4
# Author      : czsrh
# Date        : 10.06.2021
# Institution : Carl Zeiss Microscopy GmbH
#
# Disclaimer: This tool is purely experimental. Feel free to
# use it at your own risk.
#
# Copyright (c) 2021 Carl Zeiss AG, Germany. All Rights Reserved.
#################################################################

try:
    import napari
except ModuleNotFoundError as error:
    print(error.__class__.__name__ + ": " + error.msg)

from PyQt5.QtWidgets import (

    QHBoxLayout,
    QVBoxLayout,
    QFileSystemModel,
    QFileDialog,
    QTreeView,
    QDialogButtonBox,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QCheckBox,
    QAbstractItemView,
    QComboBox,
    QPushButton,
    QLineEdit,
    QLabel,
    QGridLayout

)

from PyQt5.QtCore import Qt, QDir, QSortFilterProxyModel
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QFont

#import tools.czifile_tools as czt
from czifiletools import czifile_tools as czt
#import tools.imgfile_tools as imf
import zarr
import dask
import dask.array as da
import numpy as np
import time


class TableWidget(QWidget):

    def __init__(self):

        super(QWidget, self).__init__()

        self.layout = QVBoxLayout(self)
        self.mdtable = QTableWidget()
        self.layout.addWidget(self.mdtable)
        self.mdtable.setShowGrid(True)
        self.mdtable.setHorizontalHeaderLabels(['Parameter', 'Value'])
        header = self.mdtable.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignLeft)

    def update_metadata(self, metadata):

        # number of rows is set to number of metadata entries
        row_count = len(metadata)
        col_count = 2
        self.mdtable.setColumnCount(col_count)
        self.mdtable.setRowCount(row_count)

        row = 0

        # update the table with the entries from metadata dictionary
        for key, value in metadata.items():
            newkey = QTableWidgetItem(key)
            self.mdtable.setItem(row, 0, newkey)
            newvalue = QTableWidgetItem(str(value))
            self.mdtable.setItem(row, 1, newvalue)
            row += 1

        # fit columns to content
        self.mdtable.resizeColumnsToContents()

    def update_style(self):

        # define font size and type
        fnt = QFont()
        fnt.setPointSize(11)
        fnt.setBold(True)
        fnt.setFamily('Arial')

        # update both header items
        fc = (25, 25, 25)
        item1 = QtWidgets.QTableWidgetItem('Parameter')
        #item1.setForeground(QtGui.QColor(25, 25, 25))
        item1.setFont(fnt)
        self.mdtable.setHorizontalHeaderItem(0, item1)

        item2 = QtWidgets.QTableWidgetItem('Value')
        #item2.setForeground(QtGui.QColor(25, 25, 25))
        item2.setFont(fnt)
        self.mdtable.setHorizontalHeaderItem(1, item2)


def show_napari(viewer, array, metadata,
                use_dimstr = True,
                dimstr = 'STZCYX',
                blending='additive',
                adjust_contrast=True,
                gamma=0.85,
                add_mdtable=True,
                rename_sliders=False):
    """Show the multidimensional array using the Napari viewer

    :param viewer: Instance of the napari viewer
    :type array: NapariViewer
    :param array: multidimensional NumPy.Array containing the pixel data
    :type array: NumPy.Array
    :param metadata: dictionary with CZI or OME-TIFF metadata
    :type metadata: dict
    :param blending: NapariViewer option for blending, defaults to 'additive'
    :type blending: str, optional
    :param gamma: NapariViewer value for Gamma, defaults to 0.85
    :type gamma: float, optional
    :param rename_sliders: name slider with correct labels output, defaults to False
    :type verbose: bool, optional
    """

    # create list for the napari layers
    napari_layers = []

    # create scalefcator with all ones
    scalefactors = [1.0] * len(array.shape)

    # use the dimension string from AICSImageIO 6D
    #dimpos = czt.get_dimpositions(metadata['Axes_aics'])
    #dimpos = czt.get_dimpositions(metadata['czi_dims'])
    if not use_dimstr:
        dimpos = czt.get_dimpositions('S' + metadata['czi_dims5D_aics'])
    if use_dimstr:
        dimpos = czt.get_dimpositions(dimstr)


    # get the scalefactors from the metadata
    scalef = czt.get_scalefactor(metadata)

    # modify the tuple for the scales for napari
    scalefactors[dimpos['Z']] = scalef['zx']

    # remove C dimension from scalefactor
    scalefactors_ch = scalefactors.copy()
    del scalefactors_ch[dimpos['C']]

    # add widget for metadata
    if add_mdtable:

        # create widget for the metadata
        mdbrowser = TableWidget()

        viewer.window.add_dock_widget(mdbrowser,
                                      name='mdbrowser',
                                      area='right')

        # add the metadata and adapt the table display
        mdbrowser.update_metadata(metadata)
        mdbrowser.update_style()

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
        channel = slicedim(array, ch, dimpos['C'])

        # actually show the image array
        print('Adding Channel  :', chname)
        print('Shape Channel   :', ch, channel.shape)
        print('Scaling Factors :', scalefactors_ch)

        if adjust_contrast:
            sc = calc_scaling(channel, corr_max=0.5)
            print('Display Scaling', sc)

            # add channel to napari viewer
            new_layer = viewer.add_image(channel,
                                         name=chname,
                                         scale=scalefactors_ch,
                                         contrast_limits=sc,
                                         blending=blending,
                                         gamma=gamma)

        if not adjust_contrast:
            # add channel to napari viewer
            new_layer = viewer.add_image(channel,
                                         name=chname,
                                         scale=scalefactors_ch,
                                         blending=blending,
                                         gamma=gamma)

        napari_layers.append(new_layer)

    if rename_sliders:

        print('Renaming the Sliders based on the Dimension String ....')

        # get the label of the sliders (as a tuple) ad rename it
        #viewer.dims.axis_labels = napari_rename_sliders(viewer.dims.axis_labels, metadata['Axes_aics'])
        #sliderlabels = napari_rename_sliders(viewer.dims.axis_labels, metadata['czi_dims'])
        #sliderlabels = napari_rename_sliders(viewer.dims.axis_labels, metadata['czi_dims5D_aics'])
        if not use_dimstr:
            sliderlabels = napari_rename_sliders(viewer.dims.axis_labels, 'S' + metadata['czi_dims5D_aics'])
        if use_dimstr:
            sliderlabels = napari_rename_sliders(viewer.dims.axis_labels, dimstr)


        viewer.dims.axis_labels = sliderlabels

    return napari_layers


def napari_rename_sliders(sliders, dimorder):
    """Rename the sliders of the Napari viewer according to the dimensions.

    :param sliders: Tuple containing the slider label
    :type sliders: tuple
    :param dimorder: Dimension string using AICSImageIO or aicspylibczi
    :type dimorder: str
    :return: Tuple with new slider labels
    :rtype: tuple
    """

    # get the positions of dimension entries after removing C dimension
    dimpos_viewer = czt.get_dimpositions(dimorder)

    # update the labels with the correct dimension strings
    slidernames = ['B', 'H', 'V', 'M', 'S', 'T', 'Z']

    # convert to list()
    tmp_sliders = list(sliders)

    for s in slidernames:
        try:
            if dimpos_viewer[s] >= 0:

                # assign the dimension labels
                tmp_sliders[dimpos_viewer[s]] = s

                # convert back to tuple
                sliders = tuple(tmp_sliders)
        except KeyError:
            print('No', s, 'Dimension found')

    return sliders


def slicedim(array, dimindex, posdim):
    """slice out a specific channel without (!) dropping the dimension
    # of the array to conserve the dimorder string
    # this should work for Numpy.Array, Dask and ZARR ...

    :param array: The array to be sliced
    :type array: Numpy.Array, dask.Array, zarr.Array
    :param dimindex: index to be sliced out at a given dimension
    :type dimindex: int
    :param posdim: index of the dimension where the slicing should take place
    :type posdim: int
    :return: sliced array
    :rtype: Numpy.Array, dask.array, zarr.array
    """

    if posdim == 0:
        array_sliced = array[dimindex:dimindex + 1, ...]
    if posdim == 1:
        array_sliced = array[:, dimindex:dimindex + 1, ...]
    if posdim == 2:
        array_sliced = array[:, :, dimindex:dimindex + 1, ...]
    if posdim == 3:
        array_sliced = array[:, :, :, dimindex:dimindex + 1, ...]
    if posdim == 4:
        array_sliced = array[:, :, :, :, dimindex:dimindex + 1, ...]
    if posdim == 5:
        array_sliced = array[:, :, :, :, :, dimindex:dimindex + 1, ...]

    """
    # old way to it differently

    if isinstance(array, da.Array):
        print('Extract Channel as Dask.Array')
        channel = slicedimC(array, ch, dimpos['C'])
        # channel = array.compute().take(ch, axis=dimpos['C'])
    if isinstance(array, zarr.Array):
        print('Extract Channel as Dask.Array')
        channel = slicedimC(array, ch, dimpos['C'])
    if isinstance(array, np.ndarray):
        # use normal numpy if not
        print('Extract Channel as NumPy.Array')
        channel = array.take(ch, axis=dimpos['C'])
    """

    return array_sliced


def calc_scaling(data, corr_min=1.0,
                 offset_min=0,
                 corr_max=0.85,
                 offset_max=0):
    """Calculate the scaling for better display

    :param data: Calculate min / max scaling
    :type data: Numpy.Array or dask.array or zarr.array
    :param corr_min: correction factor for minvalue, defaults to 1.0
    :type corr_min: float, optional
    :param offset_min: offset for min value, defaults to 0
    :type offset_min: int, optional
    :param corr_max: correction factor for max value, defaults to 0.85
    :type corr_max: float, optional
    :param offset_max: offset for max value, defaults to 0
    :type offset_max: int, optional
    :return: list with [minvalue, maxvalue]
    :rtype: list
    """

    start = time.time()

    # get min-max values for initial scaling
    if isinstance(data, zarr.Array):
        minvalue, maxvalue = np.min(data), np.max(data)
    elif isinstance(data, da.Array):
        # use dask.compute only once since this is faster
        minvalue, maxvalue = da.compute(data.min(), data.max())
    else:
        minvalue, maxvalue = np.min(data, initial=0), np.max(data, initial=0)

    end = time.time()

    minvalue = np.round((minvalue + offset_min) * corr_min, 0)
    maxvalue = np.round((maxvalue + offset_max) * corr_max, 0)

    print('Scaling:', minvalue, maxvalue)
    print('Calculation of Min-Max [s] : ', end - start)

    return [minvalue, maxvalue]

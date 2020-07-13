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
import skimage
from skimage import filters, io
from typing import (
    Dict,
    Any,
)

from czitools import imgfileutils as imf
from aicsimageio import AICSImage, imread
import dask.array as da
from nptyping import NDArray
import os
import javabridge
import bioformats
from pathlib import Path


def metadata(path: str) -> Dict[str, Any]:
    xml = bioformats.get_omexml_metadata(path)
    md = bioformats.omexml.OMEXML(xml)

    meta = {'AcquisitionDate': md.image().AcquisitionDate}
    meta['Name'] = md.image().Name
    meta['SizeC'] = md.image().Pixels.SizeC
    meta['SizeT'] = md.image().Pixels.SizeT
    meta['SizeX'] = md.image().Pixels.SizeX
    meta['SizeY'] = md.image().Pixels.SizeY
    meta['SizeZ'] = md.image().Pixels.SizeZ
    meta['PhysicalSizeX'] = md.image().Pixels.PhysicalSizeX
    meta['PhysicalSizeY'] = md.image().Pixels.PhysicalSizeY
    meta['PhysicalSizeZ'] = md.image().Pixels.PhysicalSizeZ
    meta['PositionX'] = md.image().Pixels.Plane().PositionX
    meta['PositionY'] = md.image().Pixels.Plane().PositionY
    meta['Timepoint'] = md.image().Pixels.Plane().DeltaT

    return(meta)


def load_bioformats(path: str) -> NDArray[int]:
    meta = metadata(path)

    image = np.empty((meta['SizeC'], meta['SizeT'], meta['SizeZ'],
                      meta['SizeX'], meta['SizeY']))
    with bioformats.ImageReader(path) as rdr:
        for c in range(0, meta['SizeC']):
            for t in range(0, meta['SizeT']):
                for z in range(0, meta['SizeZ']):
                    image[c, t, z, :, :] = rdr.read(c=c, z=z, t=t,
                                                    series=None,
                                                    index=None,
                                                    rescale=False,
                                                    wants_max_intensity=False,
                                                    channel_names=None)

    return(np.squeeze(image))


def calc_scaling(data, corr_min=1.0,
                 offset_min=0,
                 corr_max=0.85,
                 offset_max=0):

    # get min-max values for initial scaling
    minvalue = np.round((data.min() + offset_min) * corr_min)
    maxvalue = np.round((data.max() + offset_max) * corr_max)
    print('Scaling: ', minvalue, maxvalue)

    return [minvalue, maxvalue]


def get_dimpositions(dimstring, tocheck=['B', 'S', 'T', 'Z', 'C']):
    """Simple function to get the indices of the dimension identifiers in a string

    :param dimstring: dimension string
    :type dimstring: str
    :param tocheck: list of entries to check, defaults to ['B', 'S', 'T', 'Z', 'C']
    :type tocheck: list, optional
    :return: dictionary with positions of dimensions inside string
    :rtype: dict
    """
    dimpos = {}
    for p in tocheck:
        dimpos[p] = dimstring.find(p)

    return dimpos


def add_napari(array, metadata,
               blending='additive',
               gamma=0.75,
               verbose=True,
               use_pylibczi=True,
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
    :param use_pylibczi: specify if pylibczi was used to read the CZI file, defaults to True
    :type use_pylibczi: bool, optional
    :param rename_sliders: name slider with correct labels output, defaults to False
    :type verbose: bool, optional
    """

    # create scalefcator with all ones
    scalefactors = [1.0] * len(array.shape)

    if metadata['ImageType'] == 'ometiff':

        # find position of dimensions
        dimpos = get_dimpositions(metadata['Axes_aics'])

        # get the scalefactors from the metadata
        scalef = imf.get_scalefactor(metadata)
        # tempoaray workaround for slider / floating point issue
        # https://forum.image.sc/t/problem-with-dimension-slider-when-adding-array-as-new-layer-for-ome-tiff/39092/2?u=sebi06
        scalef['zx'] = np.round(scalef['zx'], 3)

        # modify the tuple for the scales for napari
        scalef['zx'] = np.round(scalef['zx'], 3)

        # remove C dimension from scalefactor
        scalefactors_ch = scalefactors.copy()
        del scalefactors_ch[dimpos['C']]

        # add all channels as layers
        for ch in range(metadata['SizeC']):

            try:
                # get the channel name
                chname = metadata['Channels'][ch]
            except:
                # or use CH1 etc. as string for the name
                chname = 'CH' + str(ch + 1)

            # cutout channel
            channel = array.take(ch, axis=dimpos['C'])
            print('Shape Channel : ', ch, channel.shape)
            new_dimstring = metadata['Axes_aics'].replace('C', '')

            # actually show the image array
            print('Adding Channel  : ', chname)
            print('Shape Channel   : ', ch, channel.shape)
            print('Scaling Factors : ', scalefactors_ch)

            # get min-max values for initial scaling
            clim = calc_scaling(channel,
                                corr_min=1.0,
                                offset_min=0,
                                corr_max=0.85,
                                offset_max=0)

            if verbose:
                print('Scaling: ', clim)

            # add channel to napari viewer
            viewer.add_image(channel,
                             name=chname,
                             scale=scalefactors_ch,
                             contrast_limits=clim,
                             blending=blending,
                             gamma=gamma)

    if metadata['ImageType'] == 'czi':

        # use find position of dimensions
        if not use_pylibczi:
            dimpos = get_dimpositions(metadata['Axes'])

        if use_pylibczi:
            dimpos = get_dimpositions(metadata['Axes_aics'])

        # get the scalefactors from the metadata
        scalef = imf.get_scalefactor(metadata)
        # temporary workaround for slider / floating point issue
        # https://forum.image.sc/t/problem-with-dimension-slider-when-adding-array-as-new-layer-for-ome-tiff/39092/2?u=sebi06
        scalef['zx'] = np.round(scalef['zx'], 3)

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
                except:
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
                    if not use_pylibczi:
                        new_dimstring = metadata['Axes'].replace('C', '')
                    if use_pylibczi:
                        new_dimstring = metadata['Axes_aics'].replace('C', '')

                # actually show the image array
                print('Adding Channel  : ', chname)
                print('Shape Channel   : ', ch, channel.shape)
                print('Scaling Factors : ', scalefactors_ch)

                # get min-max values for initial scaling
                clim = calc_scaling(channel,
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
            except:
                # or use CH1 etc. as string for the name
                chname = 'CH' + str(ch + 1)

            # actually show the image array
            print('Adding Channel: ', chname)
            print('Scaling Factors: ', scalefactors)

            # get min-max values for initial scaling
            clim = calc_scaling(array)

            viewer.add_image(array,
                             name=chname,
                             scale=scalefactors,
                             contrast_limits=clim,
                             blending=blending,
                             gamma=gamma)

    if rename_sliders:

        print('Renaming the Sliders based on the Dimension String ....')

        # get the position of dimension entries after removing C dimension
        dimpos_viewer = get_dimpositions(new_dimstring)

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

            # temporary workaround for slider / floating point issue
            # https://forum.image.sc/t/problem-with-dimension-slider-when-adding-array-as-new-layer-for-ome-tiff/39092/2?u=sebi06

            md['XScale'] = np.round(md['XScale'], 3)
            md['YScale'] = np.round(md['YScale'], 3)
            md['ZScale'] = np.round(md['ZScale'], 3)

            # get AICSImageIO object using the python wrapper for libCZI (if file is CZI)
            img = AICSImage(path)
            stack = img.get_image_data()

            if md['ImageType'] == 'czi':
                use_pylibczi = True
            if md['ImageType'] == 'ometiff':
                use_pylibczi = False

            add_napari(stack, md,
                       blending='additive',
                       gamma=0.85,
                       verbose=True,
                       use_pylibczi=use_pylibczi,
                       rename_sliders=True)


with napari.gui_qt():

    # create a viewer and add some images
    viewer = napari.Viewer()
    # add the gui to the viewer as a dock widget
    viewer.window.add_dock_widget(Open_files(), area="right")
    # viewer.window.add_dock_widget(Processsing())

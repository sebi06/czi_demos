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
from czitools import segmentation_tools as sgt
from czitools import visutools as vst
from aicsimageio import AICSImage, imread
import dask.array as da
from nptyping import NDArray
import os
import javabridge
import bioformats


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


def add_napari(array, metadata,
               blending='additive',
               gamma=0.85,
               verbose=True,
               use_pylibczi=True):
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
    """

    # create scalefcator with all ones
    scalefactors = [1.0] * len(array.shape)

    if metadata['ImageType'] == 'ometiff':

        # find position of dimensions
        posZ = metadata['DimOrder BF Array'].find('Z')
        posC = metadata['DimOrder BF Array'].find('C')
        posT = metadata['DimOrder BF Array'].find('T')

        # get the scalefactors from the metadata
        scalef = imf.get_scalefactor(metadata)

        # modify the tuple for the scales for napari
        scalefactors[posZ] = scalef['zx']

        # add all channels as layers
        for ch in range(metadata['SizeC']):

            try:
                # get the channel name
                chname = metadata['Channels'][ch]
            except:
                # or use CH1 etc. as string for the name
                chname = 'CH' + str(ch + 1)

            # cutout channel
            channel = array.take(ch, axis=posC)
            print('Shape Channel : ', ch, channel.shape)

            # actually show the image array
            print('Scaling Factors: ', scalefactors)

            # get min-max values for initial scaling
            clim = [channel.min(), np.round(channel.max() * 0.85)]
            if verbose:
                print('Scaling: ', clim)

            viewer.add_image(channel,
                             name=chname,
                             scale=scalefactors,
                             contrast_limits=clim,
                             blending=blending,
                             gamma=gamma)

    if metadata['ImageType'] == 'czi':

        if not use_pylibczi:
            # use find position of dimensions
            posZ = metadata['Axes'].find('Z')
            posC = metadata['Axes'].find('C')
            posT = metadata['Axes'].find('T')

        if use_pylibczi:
            posZ = metadata['Axes_aics'].find('Z')
            posC = metadata['Axes_aics'].find('C')
            posT = metadata['Axes_aics'].find('T')

        # get the scalefactors from the metadata
        scalef = imf.get_scalefactor(metadata)

        # modify the tuple for the scales for napari
        scalefactors[posZ] = scalef['zx']

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
                    channel = array.compute().take(ch, axis=posC)

                else:
                    # use normal numpy if not
                    print('Extract Channel NumPy.Array')
                    channel = array.take(ch, axis=posC)
                    scalefactors_czi = scalefactors.pop(posC)

                print('Shape Channel : ', ch, channel.shape)

                # actually show the image array
                print('Adding Channel: ', chname)
                print('Scaling Factors: ', scalefactors)

                # get min-max values for initial scaling
                clim = calc_scaling(channel)

                viewer.add_image(channel,
                                 name=chname,
                                 scale=scalefactors_czi,
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


class Processsing(QWidget):

    def __init__(self):
        super(QWidget, self).__init__()
        self.layout = QHBoxLayout(self)

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()

        # self.tabs.setTabPosition(QTabWidget.North)

        # Add tabs
        self.tabs.addTab(self.tab1, "Gaussian")
        self.tabs.addTab(self.tab2, "Median")
        self.tabs.addTab(self.tab3, "Sobel")

        # tab1
        self.tab1.layout = QHBoxLayout(self)
        self.sld1 = QSlider(Qt.Horizontal, self)
        self.btn1 = QPushButton("create result layer", self)
        # drop down menu
        self.cb = QComboBox()
        self.cb.addItems(["reflect", "constant", "nearest", "mirror", "wrap"])
        # set min and max slider
        self.sld1.setMinimum(0)
        self.sld1.setMaximum(10)
        # add widget to the tab
        self.tab1.layout.addWidget(self.sld1)
        self.tab1.layout.addWidget(self.btn1)
        self.tab1.layout.addWidget(self.cb)
        # set layout to the tab
        self.tab1.setLayout(self.tab1.layout)
        # connect different button/slider
        self.sld1.valueChanged[int].connect(self.updateValue)
        self.btn1.clicked.connect(self.buttonClicked)
        self.cb.currentIndexChanged.connect(self.modeChoice)

        # tab2
        self.tab2.layout = QHBoxLayout(self)
        self.sld2 = QSlider(Qt.Horizontal, self)
        self.btn2 = QPushButton("create result layer", self)
        self.cb2 = QComboBox()
        self.cb2.addItems(["diamond", "disk", "square"])
        self.sld2.setMinimum(0)
        self.sld2.setMaximum(10)
        self.tab2.layout.addWidget(self.sld2)
        self.tab2.layout.addWidget(self.btn2)
        self.tab2.layout.addWidget(self.cb2)
        self.tab2.setLayout(self.tab2.layout)
        self.sld2.valueChanged[int].connect(self.updateValue)
        self.btn2.clicked.connect(self.buttonClicked)
        self.cb2.currentIndexChanged.connect(self.morphoChoice)

        # tab3
        self.tab3.layout = QHBoxLayout(self)
        self.btn3 = QPushButton("create result layer", self)
        self.tab3.layout.addWidget(self.btn3)
        self.tab3.setLayout(self.tab3.layout)
        self.btn3.clicked.connect(self.buttonClicked)
        self.btn4 = QPushButton("sobel", self)
        self.tab3.layout.addWidget(self.btn4)
        self.tab3.setLayout(self.tab3.layout)
        self.btn4.clicked.connect(self.buttonClicked)

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    # return choices from drop down menu
    def morphoChoice(self):
        morpho = self.cb2.currentText()
        return(morpho)

    def modeChoice(self):
        mode = self.cb.currentText()
        return(mode)

    def updateValue(self, value):
        # findwhich slider is sending the value
        if self.sender() == self.sld1:
            result = filters.gaussian(viewer.layers['image_1'].data,
                                      sigma=value, mode=self.modeChoice(), preserve_range=True)
            # update napari layer 'result gauss'
            viewer.layers['result_gauss'].data = result
        elif self.sender() == self.sld2:
            selem = getattr(skimage.morphology.selem, self.morphoChoice())
            result = filters.median(viewer.layers['image_1'].data, selem=selem(value))
            viewer.layers['result_median'].data = result

    # Create a result layer to apply filter
    def buttonClicked(self):
        if self.sender() == self.btn1:
            viewer.add_image(viewer.layers['image_1'].data, name='result_gauss')
            self.sld1.setValue(0)
            self.sld2.setValue(0)
        elif self.sender() == self.btn2:
            viewer.add_image(viewer.layers['image_1'].data, name='result_median')
            self.sld1.setValue(0)
            self.sld2.setValue(0)
        elif self.sender() == self.btn3:
            viewer.add_image(viewer.layers['image_1'].data, name='result_sobel')
            self.sld1.setValue(0)
            self.sld2.setValue(0)
        elif self.sender() == self.btn4:
            result = filters.sobel(viewer.layers['image_1'].data)
            viewer.layers['result_sobel'].data = result
            # Need to adjust napari contrast to result of skimage "as_float"
            viewer.layers['result_sobel'].contrast_limits_range = viewer.layers['result_sobel']._calc_data_range()


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
            md, additional_mdczi = imf.get_metadata(path, omeseries=0)

            # get AICSImageIO object using the python wrapper for libCZI (if file is CZI)
            img = AICSImage(path)
            stack = img.get_image_data()

            # create scalefcator with all ones
            scalefactors = [1.0] * len(stack.shape)

            add_napari(stack, md,
                       blending='additive',
                       gamma=0.85,
                       verbose=True,
                       use_pylibczi=True)

            """
            if path.endswith((".tiff", ".tif", ".czi")):
                # Uses aicsimageio to open these files
                # (image.io but the channel in last dimension which doesn't)
                # work with napari
                image = imread(path)
            if path.endswith((".jpg", ".png")):
                image = io.imread(path)
            if path.endswith((".ome.tiff", "ome.tif")):
                # a little slow but dask_image imread doesn't work well
                javabridge.start_vm(class_path=bioformats.JARS)
                # image = load_bioformats(path)
                image = imread(path)
            viewer.add_image(image, name="image_1")
            """


with napari.gui_qt():
    # create a viewer and add some images
    viewer = napari.Viewer()
    # add the gui to the viewer as a dock widget
    viewer.window.add_dock_widget(Open_files(), area="right")
    viewer.window.add_dock_widget(Processsing())

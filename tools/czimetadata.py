# -*- coding: utf-8 -*-

#################################################################
# File        : czimetadata.py
# Version     : 0.0.3
# Author      : czsrh
# Date        : 19.07.2021
# Institution : Carl Zeiss Microscopy GmbH
#
# Disclaimer: This tool is purely experimental. Feel free to
# use it at your own risk.
#
# Copyright (c) 2021 Carl Zeiss AG, Germany. All Rights Reserved.
#################################################################

import os
import sys
from pathlib import Path
import xmltodict
from collections import Counter, OrderedDict
import xml.etree.ElementTree as ET
#from aicsimageio import AICSImage
from aicspylibczi import CziFile
#import pydash
#import zarr
#import dask.array as da
#import itertools as it
from tqdm.contrib.itertools import product
import pandas as pd
import numpy as np
#from datetime import datetime
import dateutil.parser as dt
#from lxml import etree
import pydash


class CZIMetadata:

    def __init__(self, filename, dim2none=True):

        # get metadata dictionary using aicspylibczi
        aicsczi = CziFile(filename)
        xmlstr = ET.tostring(aicsczi.meta)
        metadata = xmltodict.parse(xmlstr)

        # get directory, filename, SW version and acquisition data
        self.info = CziFileInfo(filename, metadata)

        # check if CZI is a RGB file
        if 'A' in aicsczi.dims:
            self.isRGB = True
        else:
            self.isRGB = False

        # get additional data by using pylibczi directly
        self.aicsczi_dims = aicsczi.dims
        self.aicsczi_dims_shape = aicsczi.get_dims_shape()
        self.aicsczi_size = aicsczi.size
        self.isMosaic = aicsczi.is_mosaic()

        # determine pixel type for CZI array by reading XML metadata
        self.pixeltype = metadata['ImageDocument']['Metadata']['Information']['Image']['PixelType']

        # determine pixel type for CZI array using aicspylibczi
        self.pixeltype_aics = aicsczi.pixel_type

        # determine pixel type for CZI array
        self.npdtype, self.maxrange = self.get_dtype_fromstring(self.pixeltype_aics)

        # get the dimensions and order
        self.dims = CziDimensions(metadata, dim2none=True)
        self.dim_order, self.dim_index, self.dim_valid = self.get_dimorder(aicsczi.dims)

        # get the bounding boxes
        self.bbox = CziBoundingBox(aicsczi, isMosaic=self.isMosaic)

        # get information about channels
        self.channelinfo = CziChannelInfo(metadata)

        # get scaling info
        self.scale = CziScaling(metadata, dim2none=True)

        # get objetive information
        self.objective = CziObjectives(metadata)

        # get detector information
        self.detector = CziDetector(metadata)

        # get detector information
        self.microscope = CziMicroscope(metadata)

        # get information about sample carrier and wells etc.
        self.sample = CziSampleInfo(metadata)

    # can be also used without creating an instance of the class
    @staticmethod
    def get_dtype_fromstring(pixeltype):

        dytpe = None

        if pixeltype == 'gray16' or pixeltype == 'Gray16':
            dtype = np.dtype(np.uint16)
            maxvalue = 65535
        if pixeltype == 'gray8' or pixeltype == 'Gray8':
            dtype = np.dtype(np.uint8)
            maxvalue = 255
        if pixeltype == 'bgr48' or pixeltype == 'Bgr48':
            dtype = np.dtype(np.uint16)
            maxvalue = 65535
        if pixeltype == 'bgr24' or pixeltype == 'Bgr24':
            dtype = np.dtype(np.uint8)
            maxvalue = 255

        return dtype, maxvalue

    @staticmethod
    def get_dimorder(dimstring):
        """Get the order of dimensions from dimension string

        :param dimstring: string containing the dimensions
        :type dimstring: str
        :return: dims_dict - dictionary with the dimensions and its positions
        :rtype: dict
        :return: dimindex_list - list with indices of dimensions
        :rtype: list
        :return: numvalid_dims - number of valid dimensions
        :rtype: integer
        """

        dimindex_list = []
        dims = ['R', 'I', 'M', 'H', 'V', 'B', 'S', 'T', 'C', 'Z', 'Y', 'X', 'A']
        dims_dict = {}

        # loop over all dimensions and find the index
        for d in dims:
            dims_dict[d] = dimstring.find(d)
            dimindex_list.append(dimstring.find(d))

        # check if a dimension really exists
        numvalid_dims = sum(i > 0 for i in dimindex_list)

        return dims_dict, dimindex_list, numvalid_dims


class CziDimensions:

    """
    Information CZI Dimension Characters:
    - 'X':'Width'
    - 'Y':'Height'
    - 'C':'Channel'
    - 'Z':'Slice'        # depth
    - 'T':'Time'
    - 'R':'Rotation'
    - 'S':'Scene'        # contiguous regions of interest in a mosaic image
    - 'I':'Illumination' # direction
    - 'B':'Block'        # acquisition
    - 'M':'Mosaic'       # index of tile for compositing a scene
    - 'H':'Phase'        # e.g. Airy detector fibers
    - 'V':'View'         # e.g. for SPIM
    """

    def __init__(self, md, dim2none=True):

        # get the dimensions
        self.SizeX = np.int(md['ImageDocument']['Metadata']['Information']['Image']['SizeX'])
        self.SizeY = np.int(md['ImageDocument']['Metadata']['Information']['Image']['SizeY'])

        # check C-Dimension
        try:
            self.SizeC = np.int(md['ImageDocument']['Metadata']['Information']['Image']['SizeC'])
            self.hasC = True
        except KeyError as e:
            self.hasC= False
            if dim2none:
                self.SizeC = None
            if not dim2none:
                self.SizeC = 1

        # check Z-Dimension
        try:
            self.SizeZ = np.int(md['ImageDocument']['Metadata']['Information']['Image']['SizeZ'])
            self.hasZ = True
        except KeyError as e:
            self.hasZ = False
            if dim2none:
                self.SizeZ = None
            if not dim2none:
                self.SizeZ = 1

        # check T-Dimension
        try:
            self.SizeT = np.int(md['ImageDocument']['Metadata']['Information']['Image']['SizeT'])
            self.hasT = True
        except KeyError as e:
            self.hasT = False
            if dim2none:
                self.SizeT = None
            if not dim2none:
                self.SizeT = 1

        # check M-Dimension
        try:
            self.SizeM = np.int(md['ImageDocument']['Metadata']['Information']['Image']['SizeM'])
            self.hasM = True
        except KeyError as e:
            self.hasM = False
            if dim2none:
                self.SizeM = None
            if not dim2none:
                self.SizeM = 1

        # check B-Dimension
        try:
            self.SizeB = np.int(md['ImageDocument']['Metadata']['Information']['Image']['SizeB'])
            self.hasB = True
        except KeyError as e:
            self.hasB = False
            if dim2none:
                self.SizeB = None
            if not dim2none:
                self.SizeB = 1

        # check S-Dimension
        try:
            self.SizeS = np.int(md['ImageDocument']['Metadata']['Information']['Image']['SizeS'])
            self.hasS = True
        except KeyError as e:
            self.hasS = False
            if dim2none:
                self.SizeS = None
            if not dim2none:
                self.SizeS = 1

        # check H-Dimension
        try:
            self.SizeH = np.int(md['ImageDocument']['Metadata']['Information']['Image']['SizeH'])
            self.hasH = True
        except KeyError as e:
            self.hasH = False
            if dim2none:
                self.SizeH = None
            if not dim2none:
                self.SizeH = 1

        # check I-Dimension
        try:
            self.SizeI = np.int(md['ImageDocument']['Metadata']['Information']['Image']['SizeH'])
            self.hasI = True
        except KeyError as e:
            self.hasI = False
            if dim2none:
                self.SizeI = None
            if not dim2none:
                self.SizeI = 1

        # check R-Dimension
        try:
            self.SizeR = np.int(md['ImageDocument']['Metadata']['Information']['Image']['SizeR'])
            self.hasR = True
        except KeyError as e:
            self.hasR = False
            if dim2none:
                self.SizeR = None
            if not dim2none:
                self.SizeR = 1


class CziBoundingBox:
    def __init__(self, aicsczi, isMosaic=False):

        self.all_scenes = aicsczi.get_all_scene_bounding_boxes()
        if isMosaic:
            self.all_mosaic_scenes = aicsczi.get_all_mosaic_scene_bounding_boxes()
            self.all_mosaic_tiles = aicsczi.get_all_mosaic_tile_bounding_boxes()
            self.all_tiles = aicsczi.get_all_tile_bounding_boxes()


class CziChannelInfo:
    def __init__(self, md):

        # create empty lists for channel related information
        channels = []
        channels_names = []
        channels_colors = []
        channels_contrast = []
        channels_gamma = []

        try:
            sizeC = np.int(md['ImageDocument']['Metadata']['Information']['Image']['SizeC'])
        except KeyError as e:
            sizeC = 1


        # in case of only one channel
        if sizeC == 1:
            # get name for dye
            try:
                channels.append(md['ImageDocument']['Metadata']['DisplaySetting']['Channels']['Channel']['ShortName'])
            except KeyError as e:
                print('Channel shortname not found :', e)
                try:
                    channels.append(md['ImageDocument']['Metadata']['DisplaySetting']['Channels']['Channel']['DyeName'])
                except KeyError as e:
                    print('Channel dye not found :', e)
                    channels.append('Dye-CH1')

            # get channel name
            try:
                channels_names.append(md['ImageDocument']['Metadata']['DisplaySetting']['Channels']['Channel']['Name'])
            except KeyError as e:
                try:
                    channels_names.append(md['ImageDocument']['Metadata']['DisplaySetting']['Channels']['Channel']['@Name'])
                except KeyError as e:
                    print('Channel name found :', e)
                    channels_names.append('CH1')

            # get channel color
            try:
                channels_colors.append(md['ImageDocument']['Metadata']['DisplaySetting']['Channels']['Channel']['Color'])
            except KeyError as e:
                print('Channel color not found :', e)
                channels_colors.append('#80808000')

            # get contrast setting fro DisplaySetting
            try:
                low = np.float(md['ImageDocument']['Metadata']['DisplaySetting']['Channels']['Channel']['Low'])
            except KeyError as e:
                low = 0.1
            try:
                high = np.float(md['ImageDocument']['Metadata']['DisplaySetting']['Channels']['Channel']['High'])
            except KeyError as e:
                high = 0.5

            channels_contrast.append([low, high])

            # get the gamma values
            try:
                channels_gamma.append(np.float(md['ImageDocument']['Metadata']['DisplaySetting']['Channels']['Channel']['Gamma']))
            except KeyError as e:
                channels_gamma.append(0.85)


        # in case of two or more channels
        if sizeC > 1:
            # loop over all channels
            for ch in range(sizeC):
                # get name for dyes
                try:
                    channels.append(md['ImageDocument']['Metadata']['DisplaySetting']['Channels']['Channel'][ch]['ShortName'])
                except KeyError as e:
                    print('Channel shortname not found :', e)
                    try:
                        channels.append(md['ImageDocument']['Metadata']['DisplaySetting']['Channels']['Channel'][ch]['DyeName'])
                    except KeyError as e:
                        print('Channel dye not found :', e)
                        channels.append('Dye-CH' + str(ch))

                # get channel names
                try:
                    channels_names.append(md['ImageDocument']['Metadata']['DisplaySetting']['Channels']['Channel'][ch]['Name'])
                except KeyError as e:
                    try:
                        channels_names.append(md['ImageDocument']['Metadata']['DisplaySetting']['Channels']['Channel'][ch]['@Name'])
                    except KeyError as e:
                        print('Channel name not found :', e)
                        channels_names.append('CH' + str(ch))

                # get channel colors
                try:
                    channels_colors.append(md['ImageDocument']['Metadata']['DisplaySetting']['Channels']['Channel'][ch]['Color'])
                except KeyError as e:
                    print('Channel color not found :', e)
                    # use grayscale instead
                    channels_colors.append('80808000')

                # get contrast setting fro DisplaySetting
                try:
                    low = np.float(md['ImageDocument']['Metadata']['DisplaySetting']['Channels']['Channel'][ch]['Low'])
                except KeyError as e:
                    low = 0.1
                try:
                    high = np.float(md['ImageDocument']['Metadata']['DisplaySetting']['Channels']['Channel'][ch]['High'])
                except KeyError as e:
                    high = 0.5

                channels_contrast.append([low, high])

                # get the gamma values
                try:
                    channels_gamma.append(np.float(md['ImageDocument']['Metadata']['DisplaySetting']['Channels']['Channel'][ch]['Gamma']))
                except KeyError as e:
                    channels_gamma.append(0.85)

        # write channels information (as lists) into metadata dictionary
        self.shortnames = channels
        self.names = channels_names
        self.colors = channels_colors
        self.clims = channels_contrast
        self.gamma = channels_gamma


class CziScaling:
    def __init__(self, md, dim2none=True):

        # get the XY scaling information
        try:
            self.X = float(md['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][0]['Value']) * 1000000
            self.Y = float(md['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][1]['Value']) * 1000000
            self.X = np.round(self.X, 3)
            self.Y = np.round(self.Y, 3)
            try:
                self.XUnit = md['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][0]['DefaultUnitFormat']
                self.YUnit = md['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][1]['DefaultUnitFormat']
            except (KeyError, TypeError) as e:
                print('Error extracting XY ScaleUnit :', e)
                self.XUnit = None
                self.YUnit = None
        except (KeyError, TypeError) as e:
            print('Error extracting XY Scale  :', e)

        # get the Z scaling information
        try:
            self.Z = float(md['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][2]['Value']) * 1000000
            self.Z = np.round(self.Z, 3)
            # additional check for faulty z-scaling
            if self.Z == 0.0:
                self.Z = 1.0
            try:
                self.ZUnit = md['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][2]['DefaultUnitFormat']
            except (IndexError, KeyError, TypeError) as e:
                print('Error extracting Z ScaleUnit :', e)
                self.ZUnit = self.XScaleUnit
        except (IndexError, KeyError, TypeError) as e:
            print('Error extracting Z Scale  :', e)
            if dim2none:
                self.Z = None
                self.ZUnit = None
            if not dim2none:
                # set to isotropic scaling if it was single plane only
                self.Z = self.XScale
                self.ZUnit = self.XScaleUnit

        # convert scale unit to avoid encoding problems
        if self.XUnit == 'µm':
            self.XUnit = 'micron'
        if self.YUnit == 'µm':
            self.YUnit = 'micron'
        if self.ZUnit == 'µm':
            self.ZUnit = 'micron'

        # get scaling ratio
        self.ratio = self.get_scale_ratio(scalex=self.X,
                                          scaley=self.Y,
                                          scalez=self.Z)

    @staticmethod
    def get_scale_ratio(scalex=1.0, scaley=1.0, scalez=1.0):

        # set default scale factor to 1.0
        scale_ratio = {'xy': 1.0,
                       'zx': 1.0
                       }
        try:
            # get the factor between XY scaling
            scale_ratio['xy'] = np.round(scalex / scaley, 3)
            # get the scalefactor between XZ scaling
            scale_ratio['zx'] = np.round(scalez / scalex, 3)
        except TypeError as e:
            print(e, 'Using defaults = 1.0')

        return scale_ratio


class CziFileInfo:
    def __init__(self, filename, md):

        # get directory and filename etc.
        self.dirname = os.path.dirname(filename)
        self.filename = os.path.basename(filename)

        # get acquisition data and SW version
        try:
            self.software_name = md['ImageDocument']['Metadata']['Information']['Application']['Name']
            self.software_version = md['ImageDocument']['Metadata']['Information']['Application']['Version']
        except KeyError as e:
            print('Key not found:', e)
            self.software_name = None
            self.software_version = None

        try:
            self.acquisition_date = md['ImageDocument']['Metadata']['Information']['Image']['AcquisitionDateAndTime']
        except KeyError as e:
            print('Key not found:', e)
            self.acquisition_date = None


class CziObjectives:
    def __init__(self, md):

        self.NA = []
        self.mag = []
        self.ID = []
        self.name = []
        self.immersion = []
        self.tubelensmag = []
        self.nominalmag = []

        # check if Instrument metadata actually exist
        if pydash.objects.has(md, ['ImageDocument', 'Metadata', 'Information', 'Instrument', 'Objectives']):
            # get objective data
            if isinstance(md['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective'], list):
                num_obj = len(md['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective'])
            else:
                num_obj = 1

            # if there is only one objective found
            if num_obj == 1:
                try:
                    self.name.append(md['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective']['Name'])
                except (KeyError, TypeError) as e:
                    print('No Objective Name :', e)
                    self.name.append(None)

                try:
                    self.immersion = md['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective']['Immersion']
                except (KeyError, TypeError) as e:
                    print('No Objective Immersion :', e)
                    self.immersion = None

                try:
                    self.NA = np.float(md['ImageDocument']['Metadata']['Information']
                                                 ['Instrument']['Objectives']['Objective']['LensNA'])
                except (KeyError, TypeError) as e:
                    print('No Objective NA :', e)
                    self.NA = None

                try:
                    self.ID = md['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective']['Id']
                except (KeyError, TypeError) as e:
                    print('No Objective ID :', e)
                    self.ID = None

                try:
                    self.tubelensmag = np.float(md['ImageDocument']['Metadata']['Information']['Instrument']['TubeLenses']['TubeLens']['Magnification'])
                except (KeyError, TypeError) as e:
                    print('No Tubelens Mag. :', e, 'Using Default Value = 1.0.')
                    self.tubelensmag = 1.0

                try:
                    self.nominalmag = np.float(md['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective']['NominalMagnification'])
                except (KeyError, TypeError) as e:
                    print('No Nominal Mag.:', e, 'Using Default Value = 1.0.')
                    self.nominalmag = 1.0

                try:
                    if self.tubelensmag is not None:
                        self.mag = self.nominalmag * self.tubelensmag
                    if self.tubelensmag is None:
                        print('Using Tublens Mag = 1.0 for calculating Objective Magnification.')
                        self.mag = self.nominalmag * 1.0

                except (KeyError, TypeError) as e:
                    print('No Objective Magnification :', e)
                    self.mag = None

            if num_obj > 1:
                for o in range(num_obj):

                    try:
                        self.name.append(md['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective'][o]['Name'])
                    except KeyError as e:
                        print('No Objective Name :', e)
                        self.name.append(None)

                    try:
                        self.immersion.append(md['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective'][o]['Immersion'])
                    except KeyError as e:
                        print('No Objective Immersion :', e)
                        self.immersion.append(None)

                    try:
                        self.NA.append(np.float(md['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective'][o]['LensNA']))
                    except KeyError as e:
                        print('No Objective NA :', e)
                        self.NA.append(None)

                    try:
                        self.ID.append(md['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective'][o]['Id'])
                    except KeyError as e:
                        print('No Objective ID :', e)
                        self.ID.append(None)

                    try:
                        self.tubelensmag.append(np.float(md['ImageDocument']['Metadata']['Information']['Instrument']['TubeLenses']['TubeLens'][o]['Magnification']))
                    except KeyError as e:
                        print('No Tubelens Mag. :', e, 'Using Default Value = 1.0.')
                        self.tubelensmag.append(1.0)

                    try:
                        self.nominalmag.append(np.float(md['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective'][o]['NominalMagnification']))
                    except KeyError as e:
                        print('No Nominal Mag. :', e, 'Using Default Value = 1.0.')
                        self.nominalmag.append(1.0)

                    try:
                        if self.tubelensmag is not None:
                            self.mag.append(self.nominalmag[o] * self.tubelensmag[o])
                        if self.tubelensmag is None:
                            print('Using Tublens Mag = 1.0 for calculating Objective Magnification.')
                            self.mag.append(self.nominalmag[o] * 1.0)

                    except KeyError as e:
                        print('No Objective Magnification :', e)
                        self.mag.append(None)


class CziDetector:
    def __init__(self, md):

        # get detector information
        self.model = []
        self.name = []
        self.ID = []
        self.modeltype = []
        self.instrumentID = []

        # check if there are any detector entries inside the dictionary
        if pydash.objects.has(md, ['ImageDocument', 'Metadata', 'Information', 'Instrument', 'Detectors']):

            if isinstance(md['ImageDocument']['Metadata']['Information']['Instrument']['Detectors']['Detector'], list):
                num_detectors = len(
                    md['ImageDocument']['Metadata']['Information']['Instrument']['Detectors']['Detector'])
            else:
                num_detectors = 1

            # if there is only one detector found
            if num_detectors == 1:

                # check for detector ID
                try:
                    self.ID.append(md['ImageDocument']['Metadata']['Information']['Instrument']['Detectors']['Detector']['Id'])
                except KeyError as e:
                    print('DetectorID not found :', e)
                    self.ID.append(None)

                # check for detector Name
                try:
                    self.name.append(md['ImageDocument']['Metadata']['Information']['Instrument']['Detectors']['Detector']['Name'])
                except KeyError as e:
                    print('DetectorName not found :', e)
                    self.name.append(None)

                # check for detector model
                try:
                    self.model.append(md['ImageDocument']['Metadata']['Information']['Instrument']['Detectors']['Detector']['Manufacturer']['Model'])
                except KeyError as e:
                    print('DetectorModel not found :', e)
                    self.model.append(None)

                # check for detector modeltype
                try:
                    self.modeltype.append(md['ImageDocument']['Metadata']['Information']['Instrument']['Detectors']['Detector']['Type'])
                except KeyError as e:
                    print('DetectorType not found :', e)
                    self.modeltype.append(None)

            if num_detectors > 1:
                for d in range(num_detectors):

                    # check for detector ID
                    try:
                        self.ID.append(md['ImageDocument']['Metadata']['Information']['Instrument']['Detectors']['Detector'][d]['Id'])
                    except KeyError as e:
                        print('DetectorID not found :', e)
                        self.ID.append(None)

                    # check for detector Name
                    try:
                        self.name.append(md['ImageDocument']['Metadata']['Information']['Instrument']['Detectors']['Detector'][d]['Name'])
                    except KeyError as e:
                        print('DetectorName not found :', e)
                        self.name.append(None)

                    # check for detector model
                    try:
                        self.model.append(md['ImageDocument']['Metadata']['Information']['Instrument']['Detectors']['Detector'][d]['Manufacturer']['Model'])
                    except KeyError as e:
                        print('DetectorModel not found :', e)
                        self.model.append(None)

                    # check for detector modeltype
                    try:
                        self.modeltype.append(md['ImageDocument']['Metadata']['Information']['Instrument']['Detectors']['Detector'][d]['Type'])
                    except KeyError as e:
                        print('DetectorType not found :', e)
                        self.modeltype.append(None)


class CziMicroscope:
    def __init__(self, md):

        self.ID = None
        self.Name = None

        # check if there are any microscope entry inside the dictionary
        if pydash.objects.has(md, ['ImageDocument', 'Metadata', 'Information', 'Instrument', 'Microscopes']):

            # check for detector ID
            try:
                self.ID = md['ImageDocument']['Metadata']['Information']['Instrument']['Microscopes']['Microscope']['Id']
            except KeyError as e:
                try:
                    self.ID = md['ImageDocument']['Metadata']['Information']['Instrument']['Microscopes']['Microscope']['@Id']
                except KeyError as e:
                    print('Microscope ID not found :', e)
                    self.ID = None

            # check for microscope system name
            try:
                self.Name = md['ImageDocument']['Metadata']['Information']['Instrument']['Microscopes']['Microscope']['System']
            except KeyError as e:
                print('Microscope System Name not found :', e)
                self.Name = None


class CziSampleInfo:
    def __init__(self, md):

        # check for well information
        self.well_array_names = []
        self.well_indices = []
        self.well_position_names = []
        self.well_colID = []
        self.well_rowID = []
        self.well_counter = []
        self.scene_stageX = []
        self.scene_stageY = []

        try:
            # get S-Dimension
            sizeS = np.int(md['ImageDocument']['Metadata']['Information']['Image']['SizeS'])
            print('Trying to extract Scene and Well information if existing ...')

            # extract well information from the dictionary
            allscenes = md['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['S']['Scenes']['Scene']

            # loop over all detected scenes
            for s in range(sizeS):

                if sizeS == 1:
                    well = allscenes
                    try:
                        self.well_array_names.append(allscenes['ArrayName'])
                    except KeyError as e:
                        try:
                            self.well_array_names.append(well['Name'])
                        except KeyError as e:
                            # print('Well Name not found :', e)
                            try:
                                self.well_array_names.append(well['@Name'])
                            except KeyError as e:
                                # print('Well @Name not found :', e)
                                print('Well Name not found :', e, 'Using A1 instead')
                                self.well_array_names.append('A1')

                    try:
                        self.well_indices.append(allscenes['Index'])
                    except KeyError as e:
                        try:
                            self.well_indices.append(allscenes['@Index'])
                        except KeyError as e:
                            print('Well Index not found :', e)
                            self.well_indices.append(1)

                    try:
                        self.well_position_names.append(allscenes['Name'])
                    except KeyError as e:
                        try:
                            self.well_position_names.append(allscenes['@Name'])
                        except KeyError as e:
                            print('Well Position Names not found :', e)
                            self.well_position_names.append('P1')

                    try:
                        self.well_colID.append(np.int(allscenes['Shape']['ColumnIndex']))
                    except KeyError as e:
                        print('Well ColumnIDs not found :', e)
                        self.well_colID.append(0)

                    try:
                        self.well_rowID.append(np.int(allscenes['Shape']['RowIndex']))
                    except KeyError as e:
                        print('Well RowIDs not found :', e)
                        self.well_rowID.append(0)

                    try:
                        # count the content of the list, e.g. how many time a certain well was detected
                        self.well_counter = Counter(self.well_array_names)
                    except KeyError:
                        self.well_counter.append(Counter({'A1': 1}))

                    try:
                        # get the SceneCenter Position
                        sx = allscenes['CenterPosition'].split(',')[0]
                        sy = allscenes['CenterPosition'].split(',')[1]
                        self.scene_stageX.append(np.double(sx))
                        self.scene_stageY.append(np.double(sy))
                    except (TypeError, KeyError) as e:
                        print('Stage Positions XY not found :', e)
                        self.scene_stageX.append(0.0)
                        self.scene_stageY.append(0.0)

                if sizeS > 1:
                    try:
                        well = allscenes[s]
                        self.well_array_names.append(well['ArrayName'])
                    except KeyError as e:
                        try:
                            self.well_array_names.append(well['Name'])
                        except KeyError as e:
                            # print('Well Name not found :', e)
                            try:
                                self.well_array_names.append(well['@Name'])
                            except KeyError as e:
                                # print('Well @Name not found :', e)
                                print('Well Name not found. Using A1 instead')
                                self.well_array_names.append('A1')

                    # get the well information
                    try:
                        self.well_indices.append(well['Index'])
                    except KeyError as e:
                        try:
                            self.well_indices.append(well['@Index'])
                        except KeyError as e:
                            print('Well Index not found :', e)
                            self.well_indices.append(None)
                    try:
                        self.well_position_names.append(well['Name'])
                    except KeyError as e:
                        try:
                            self.well_position_names.append(well['@Name'])
                        except KeyError as e:
                            print('Well Position Names not found :', e)
                            self.well_position_names.append(None)

                    try:
                        self.well_colID.append(np.int(well['Shape']['ColumnIndex']))
                    except KeyError as e:
                        print('Well ColumnIDs not found :', e)
                        self.well_colID.append(None)

                    try:
                        self.well_rowID.append(np.int(well['Shape']['RowIndex']))
                    except KeyError as e:
                        print('Well RowIDs not found :', e)
                        self.well_rowID.append(None)

                    # count the content of the list, e.g. how many time a certain well was detected
                    self.well_counter = Counter(self.well_array_names)

                    # try:
                    if isinstance(allscenes, list):
                        try:
                            # get the SceneCenter Position
                            sx = allscenes[s]['CenterPosition'].split(',')[0]
                            sy = allscenes[s]['CenterPosition'].split(',')[1]
                            self.scene_stageX.append(np.double(sx))
                            self.scene_stageY.append(np.double(sy))
                        except KeyError as e:
                            print('Stage Positions XY not found :', e)
                            self.scene_stageX.append(0.0)
                            self.scene_stageY.append(0.0)
                    if not isinstance(allscenes, list):
                        self.scene_stageX.append(0.0)
                        self.scene_stageY.append(0.0)

                # count the number of different wells
                self.number_wells = len(self.well_counter.keys())

        except (KeyError, TypeError) as e:
            print('No valid Scene or Well information found:', e)


class CZIScene:
    def __init__(self, czi, sceneindex):

        if not czi.is_mosaic():
            self.bbox = czi.get_all_scene_bounding_boxes()[sceneindex]
        if czi.is_mosaic():
            self.bbox = czi.get_mosaic_scene_bounding_box(index=sceneindex)

        self.xstart = self.bbox.x
        self.ystart = self.bbox.y
        self.width = self.bbox.w
        self.height = self.bbox.h
        self.index = sceneindex


        if 'C' in czi.dims:
            self.hasC = True
            self.sizeC = czi.get_dims_shape()[0]['C'][1]
            self.posC = czi.dims.index('C')
        else:
            self.hasC = False
            self.sizeC = None
            self.posC = None

        if 'T' in czi.dims:
            self.hasT = True
            self.sizeT = czi.get_dims_shape()[0]['T'][1]
            self.posT = czi.dims.index('T')
        else:
            self.hasT = False
            self.sizeT = None
            self.posT = None

        if 'Z' in czi.dims:
            self.hasZ = True
            self.sizeZ = czi.get_dims_shape()[0]['Z'][1]
            self.posZ = czi.dims.index('Z')
        else:
            self.hasZ = False
            self.sizeZ = None
            self.posZ = None

        if 'S' in czi.dims:
            self.hasS = True
            self.sizeS = czi.get_dims_shape()[0]['S'][1]
            self.posS = czi.dims.index('S')
        else:
            self.hasS = False
            self.sizeS = None
            self.posS = None

        if 'M' in czi.dims:
            self.hasM = True
            self.sizeM = czi.get_dims_shape()[0]['M'][1]
            self.posM = czi.dims.index('M')
        else:
            self.hasM = False
            self.sizeM = None
            self.posM = None

        if 'B' in czi.dims:
            self.hasB = True
            self.sizeB = czi.get_dims_shape()[0]['B'][1]
            self.posB = czi.dims.index('B')
        else:
            self.hasB = False
            self.sizeB = None
            self.posB = None

        if 'H' in czi.dims:
            self.hasH = True
            self.sizeH = czi.get_dims_shape()[0]['H'][1]
            self.posH = czi.dims.index('H')
        else:
            self.hasH = False
            self.sizeH = None
            self.posH = None

        if 'A' in czi.dims:
            self.hasA = True
            self.sizeH = czi.get_dims_shape()[0]['A'][1]
            self.posA = czi.dims.index('A')
            self.isRGB = True
        else:
            self.hasA = False
            self.sizeA = None
            self.posA = None
            self.isRGB = False

        # determine the shape of the scene
        self.shape_single_scene = []
        #self.single_scene_dimstr = 'S'
        self.single_scene_dimstr = ''

        dims_to_ignore = ['M', 'A', 'Y', 'X']

        # get the dimension identifier
        for d in czi.dims:
            if d not in dims_to_ignore:

                if d == 'S':
                    # set size of scene dimension to 1 because shape_single_scene
                    dimsize = 1
                else:
                    # get the position inside string
                    dimpos = czi.dims.index(d)
                    dimsize = czi.size[dimpos]

                # append
                self.shape_single_scene.append(dimsize)
                self.single_scene_dimstr = self.single_scene_dimstr + d

        # add X and Y size to the shape and dimstring for the specific scene
        self.shape_single_scene.append(self.height)
        self.shape_single_scene.append(self.width)
        self.single_scene_dimstr = self.single_scene_dimstr + 'YX'

        # check for the A-Dimension (RGB)
        if 'A' in czi.dims:
            self.single_scene_dimstr = self.single_scene_dimstr + 'A'
            self.shape_single_scene.append(3)


class Planes:
    def __init__(self):
        pass

    @staticmethod
    def get_planetable(czifile,
                       savetable=False,
                       separator=',',
                       index=True):

        # get the czi metadata
        md = CZIMetadata(czifile)
        aicsczi = CziFile(czifile)

        # initialize the plane table
        df_czi = pd.DataFrame(columns=['Subblock',
                                       'Scene',
                                       'Tile',
                                       'T',
                                       'Z',
                                       'C',
                                       'X[micron]',
                                       'Y[micron]',
                                       'Z[micron]',
                                       'Time[s]',
                                       'xstart',
                                       'ystart',
                                       'width',
                                       'height'])

        # define subblock counter
        sbcount = -1

        # check fort non existing dimensions
        if md.dims.SizeS is None:
            sizeS = 1
        else:
            sizeS = md.dims.SizeS

        if md.dims.SizeM is None:
            sizeM = 1
        else:
            sizeM = md.dims.SizeM

        if md.dims.SizeT is None:
            sizeT = 1
        else:
            sizeT = md.dims.SizeT

        if md.dims.SizeZ is None:
            sizeZ = 1
        else:
            sizeZ = md.dims.SizeZ

        if md.dims.SizeC is None:
            sizeC = 1
        else:
            sizeC = md.dims.SizeC


        def getsbinfo(subblock):
            try:
                # time = sb.xpath('//AcquisitionTime')[0].text
                time = subblock.findall(".//AcquisitionTime")[0].text
                timestamp = dt.parse(time).timestamp()
            except IndexError as e:
                timestamp = 0.0

            try:
                # xpos = np.double(sb.xpath('//StageXPosition')[0].text)
                xpos = np.double(subblock.findall("..//StageXPosition")[0].text)
            except IndexError as e:
                xpos = 0.0

            try:
                # ypos = np.double(sb.xpath('//StageYPosition')[0].text)
                ypos = np.double(subblock.findall("..//StageYPosition")[0].text)
            except IndexError as e:
                ypos = 0.0

            try:
                # zpos = np.double(sb.xpath('//FocusPosition')[0].text)
                zpos = np.double(subblock.findall("..//FocusPosition")[0].text)
            except IndexError as e:
                zpos = 0.0

            return timestamp, xpos, ypos, zpos

        # in case the CZI has the M-Dimension
        if md.isMosaic:

            for s, m, t, z, c in product(range(sizeS),
                                         range(sizeM),
                                         range(sizeT),
                                         range(sizeZ),
                                         range(sizeC)):

                sbcount += 1
                # print(s, m, t, z, c)
                #info = aicsczi.read_subblock_rect(S=s,
                #                                  M=m,
                #                                  T=t,
                #                                  Z=z,
                #                                  C=c)

                tilebbox = aicsczi.get_tile_bounding_box(S=s,
                                                         M=m,
                                                         T=t,
                                                         Z=z,
                                                         C=c)

                # read information from subblock
                sb = aicsczi.read_subblock_metadata(unified_xml=True,
                                                    B=0,
                                                    S=s,
                                                    M=m,
                                                    T=t,
                                                    Z=z,
                                                    C=c)

                # get information from subblock
                timestamp, xpos, ypos, zpos = getsbinfo(sb)

                df_czi = df_czi.append({'Subblock': sbcount,
                                        'Scene': s,
                                        'Tile': m,
                                        'T': t,
                                        'Z': z,
                                        'C': c,
                                        'X[micron]': xpos,
                                        'Y[micron]': ypos,
                                        'Z[micron]': zpos,
                                        'Time[s]': timestamp,
                                        #'xstart': info[0],
                                        #'ystart': info[1],
                                        #'xwidth': info[2],
                                        #'ywidth': info[3]},
                                        'xstart': tilebbox.x,
                                        'ystart': tilebbox.y,
                                        'width': tilebbox.w,
                                        'height': tilebbox.h},
                                       ignore_index=True)

        if not md.isMosaic:

            for s, t, z, c in product(range(sizeS),
                                      range(sizeT),
                                      range(sizeZ),
                                      range(sizeC)):

                sbcount += 1
                #info = aicsczi.read_subblock_rect(S=s,
                #                                  T=t,
                #                                  Z=z,
                #                                  C=c)

                tilebbox = aicsczi.get_tile_bounding_box(S=s,
                                                         T=t,
                                                         Z=z,
                                                         C=c)

                # read information from subblocks
                sb = aicsczi.read_subblock_metadata(unified_xml=True,
                                                    B=0,
                                                    S=s,
                                                    T=t,
                                                    Z=z,
                                                    C=c)

                # get information from subblock
                timestamp, xpos, ypos, zpos = getsbinfo(sb)

                df_czi = df_czi.append({'Subblock': sbcount,
                                        'Scene': s,
                                        'Tile': 0,
                                        'T': t,
                                        'Z': z,
                                        'C': c,
                                        'X[micron]': xpos,
                                        'Y[micron]': ypos,
                                        'Z[micron]': zpos,
                                        'Time[s]': timestamp,
                                        'xstart': tilebbox.x,
                                        'ystart': tilebbox.y,
                                        'width': tilebbox.w,
                                        'height': tilebbox.h},
                                        #'xstart': info[0],
                                        #'ystart': info[1],
                                        #'xwidth': info[2],
                                        #'ywidth': info[3]},
                                       ignore_index=True)

        def norm_columns(df, colname='Time [s]', mode='min'):
            """Normalize a specific column inside a Pandas dataframe

            :param df: DataFrame
            :type df: pf.DataFrame
            :param colname: Name of the coumn to be normalized, defaults to 'Time [s]'
            :type colname: str, optional
            :param mode: Mode of Normalization, defaults to 'min'
            :type mode: str, optional
            :return: Dataframe with normalized column
            :rtype: pd.DataFrame
            """
            # normalize columns according to min or max value
            if mode == 'min':
                min_value = df[colname].min()
                df[colname] = df[colname] - min_value

            if mode == 'max':
                max_value = df[colname].max()
                df[colname] = df[colname] - max_value

            return df

        # normalize timestamps
        df_czi = norm_columns(df_czi, colname='Time[s]', mode='min')

        # cast data  types
        df_czi = df_czi.astype({'Subblock': 'int32',
                                'Scene': 'int32',
                                'Tile': 'int32',
                                'T': 'int32',
                                'Z': 'int32',
                                'C': 'int16',
                                'xstart': 'int32',
                                'ystart': 'int32',
                                'width': 'int32',
                                'height': 'int32'},
                               copy=False,
                               errors='ignore')

        if savetable:
            csvfile = os.path.splitext(czifile)[0] + '_planetable.csv'
            print('Saving DataFrame as CSV table to: ', csvfile)
            # write the CSV data table
            df_czi.to_csv(csvfile, sep=separator, index=index)


        return df_czi

    @staticmethod
    def norm_columns(df, colname='Time [s]', mode='min'):
        """Normalize a specific column inside a Pandas dataframe

        :param df: DataFrame
        :type df: pf.DataFrame
        :param colname: Name of the coumn to be normalized, defaults to 'Time [s]'
        :type colname: str, optional
        :param mode: Mode of Normalization, defaults to 'min'
        :type mode: str, optional
        :return: Dataframe with normalized column
        :rtype: pd.DataFrame
        """
        # normalize columns according to min or max value
        if mode == 'min':
            min_value = df[colname].min()
            df[colname] = df[colname] - min_value

        if mode == 'max':
            max_value = df[colname].max()
            df[colname] = df[colname] - max_value

        return df

    @staticmethod
    def filter_planetable(planetable, S=0, T=0, Z=0, C=0):

        # filter planetable for specific scene
        if S > planetable['Scene'].max():
            print('Scene Index was invalid. Using Scene = 0.')
            S = 0
        pt = planetable[planetable['Scene'] == S]

        # filter planetable for specific timepoint
        if T > planetable['T'].max():
            print('Time Index was invalid. Using T = 0.')
            T = 0
        pt = planetable[planetable['T'] == T]

        # filter resulting planetable pt for a specific z-plane
        try:
            if Z > planetable['Z[micron]'].max():
                print('Z-Plane Index was invalid. Using Z = 0.')
                zplane = 0
                pt = pt[pt['Z[micron]'] == Z]
        except KeyError as e:
            if Z > planetable['Z [micron]'].max():
                print('Z-Plane Index was invalid. Using Z = 0.')
                zplane = 0
                pt = pt[pt['Z [micron]'] == Z]

        # filter planetable for specific channel
        if C > planetable['C'].max():
            print('Channel Index was invalid. Using C = 0.')
            C = 0
        pt = planetable[planetable['C'] == C]

        # return filtered planetable
        return pt


def create_metadata_dict(filename):

    mdata = CZIMetadata(filename)

    mdict = {'Directory': mdata.info.dirname,
             'Filename': mdata.info.filename,
             'AcqDate': mdata.info.acquisition_date,
             'SW-Name': mdata.info.software_name,
             'SW-Version' : mdata.info.software_version,
             'czi_dims': mdata.aicsczi_dims,
             'czi_dims_shape' : mdata.aicsczi_dims_shape,
             'czi_size': mdata.aicsczi_size,
             'dim_order': mdata.dim_order,
             'dim_index': mdata.dim_index,
             'dim_valid': mdata.dim_valid,
             'SizeX': mdata.dims.SizeX,
             'SizeY': mdata.dims.SizeY,
             'SizeZ': mdata.dims.SizeZ,
             'SizeC': mdata.dims.SizeC,
             'SizeT': mdata.dims.SizeT,
             'SizeS': mdata.dims.SizeS,
             'SizeB': mdata.dims.SizeB,
             'SizeM': mdata.dims.SizeM,
             'SizeH': mdata.dims.SizeH,
             'SizeI': mdata.dims.SizeI,
             'isRGB': mdata.isRGB,
             'isMosaic': mdata.isMosaic,
             'ObjNA': mdata.objective.NA,
             'ObjMag': mdata.objective.mag,
             'ObjID': mdata.objective.ID,
             'ObjName': mdata.objective.name,
             'ObjImmersion': mdata.objective.immersion,
             'TubelensMag': mdata.objective.tubelensmag,
             'ObjNominalMag': mdata.objective.nominalmag,
             'XScale': mdata.scale.X,
             'YScale': mdata.scale.Y,
             'ZScale': mdata.scale.Z,
             'XScaleUnit': mdata.scale.XUnit,
             'YScaleUnit': mdata.scale.YUnit,
             'ZScaleUnit': mdata.scale.ZUnit,
             'scale_ratio': mdata.scale.ratio,
             'DetectorModel': mdata.detector.model,
             'DetectorName': mdata.detector.name,
             'DetectorID': mdata.detector.ID,
             'DetectorType': mdata.detector.modeltype,
             'InstrumentID': mdata.detector.instrumentID,
             'ChannelsNames': mdata.channelinfo.names,
             'ChannelShortNames': mdata.channelinfo.shortnames,
             'ChannelColors': mdata.channelinfo.colors,
             'bbox_all_scenes': mdata.bbox.all_scenes,
             'WellArrayNames': mdata.sample.well_array_names,
             'WellIndicies': mdata.sample.well_indices,
             'WellPositionNames' : mdata.sample.well_position_names,
             'WellRowID': mdata.sample.well_rowID,
             'WellColumnID': mdata.sample.well_colID,
             'WellCounter': mdata.sample.well_counter,
             'SceneCenterStageX': mdata.sample.scene_stageX,
             'SceneCenterStageY': mdata.sample.scene_stageX
             }

    # check if mosaic
    if mdata.isMosaic:
        mdict['bbox_all_mosaic_scenes'] = mdata.bbox.all_mosaic_scenes
        mdict['bbox_all_mosaic_tiles'] = mdata.bbox.all_mosaic_tiles
        mdict['bbox_all_tiles'] = mdata.bbox.all_tiles

    ordered_dict = sort_dict_by_key(mdict)

    return ordered_dict


def sort_dict_by_key(unsorted_dict):
    sorted_keys = sorted(unsorted_dict.keys(), key=lambda x: x.lower())
    sorted_dict = {}
    for key in sorted_keys:
        sorted_dict.update({key: unsorted_dict[key]})

    return sorted_dict


def md2dataframe(metadata, paramcol='Parameter', keycol='Value'):
    """Convert the metadata dictionary to a Pandas DataFrame.

    :param metadata: MeteData dictionary
    :type metadata: dict
    :param paramcol: Name of Columns for the MetaData Parameters, defaults to 'Parameter'
    :type paramcol: str, optional
    :param keycol: Name of Columns for the MetaData Values, defaults to 'Value'
    :type keycol: str, optional
    :return: Pandas DataFrame containing all the metadata
    :rtype: Pandas.DataFrame
    """
    mdframe = pd.DataFrame(columns=[paramcol, keycol])

    for k in metadata.keys():
        d = {'Parameter': k, 'Value': metadata[k]}
        df = pd.DataFrame([d], index=[0])
        mdframe = pd.concat([mdframe, df], ignore_index=True)

    return mdframe


def writexml_czi(filename, xmlsuffix='_CZI_MetaData.xml'):
    """Write XML imformation of CZI to disk

    :param filename: CZI image filename
    :type filename: str
    :param xmlsuffix: suffix for the XML file that will be created, defaults to '_CZI_MetaData.xml'
    :type xmlsuffix: str, optional
    :return: filename of the XML file
    :rtype: str
    """

    # get metadata dictionary using aicspylibczi
    aicsczi = CziFile(filename)
    metadata_xmlstr = ET.tostring(aicsczi.meta)

    # change file name
    xmlfile = filename.replace('.czi', xmlsuffix)

    # get tree from string
    tree = ET.ElementTree(ET.fromstring(metadata_xmlstr))

    # write XML file to same folder
    tree.write(xmlfile, encoding='utf-8', method='xml')

    return xmlfile


def addzeros(number):
    """Convert a number into a string and add leading zeros.
    Typically used to construct filenames with equal lengths.

    :param number: the number
    :type number: int
    :return: zerostring - string with leading zeros
    :rtype: str
    """

    if number < 10:
        zerostring = '0000' + str(number)
    if number >= 10 and number < 100:
        zerostring = '000' + str(number)
    if number >= 100 and number < 1000:
        zerostring = '00' + str(number)
    if number >= 1000 and number < 10000:
        zerostring = '0' + str(number)

    return zerostring


def get_fname_woext(filepath):
    """Get the complete path of a file without the extension
    It alos will works for extensions like c:\myfile.abc.xyz
    The output will be: c:\myfile

    :param filepath: complete fiepath
    :type filepath: str
    :return: complete filepath without extension
    :rtype: str
    """
    # create empty string
    real_extension = ''

    # get all part of the file extension
    sufs = Path(filepath).suffixes
    for s in sufs:
        real_extension = real_extension + s

    # remover real extension from filepath
    filepath_woext = filepath.replace(real_extension, '')

    return filepath_woext


def readczi(filename):

    # get the CZI metadata
    md = CZIMetadata(filename)

    # read CZI using aicspylibczi
    aicsczi = CziFile(filename)

    if not aicsczi.is_mosaic():

        # get the shape for the 1st scene
        scene = CZIScene(aicsczi, sceneindex=0)
        shape_all = scene.shape_single_scene

        # only update the shape for the scene if the CZI has an S-Dimension
        if scene.hasS:
            shape_all[scene.posS] = md.dims.SizeS

        print('Shape all Scenes : ', shape_all)
        print('DimString all Scenes : ', scene.single_scene_dimstr)

        # create an empty array with the correct dimensions
        all_scenes = np.empty(aicsczi.size, dtype=md.npdtype)

        # loop over scenes if CZI is not a mosaic image
        if md.dims.SizeS is None:
            sizeS = 1
        else:
            sizeS = md.dims.SizeS

        for s in range(sizeS):

            # read the image stack for the current scene
            current_scene, shp = aicsczi.read_image(S=s)

            # create th index lists containing the slice objects
            if scene.hasS:
                idl_scene = [slice(None, None, None)] * (len(all_scenes.shape) - 2)
                idl_scene[aicsczi.dims.index('S')] = 0
                idl_all = [slice(None, None, None)] * (len(all_scenes.shape) - 2)
                idl_all[aicsczi.dims.index('S')] = s

                # cast current stack into the stack for all scenes
                all_scenes[tuple(idl_all)] = current_scene[tuple(idl_scene)]

            # if there is no S-Dimension use the stack directly
            if not scene.hasS:
                all_scenes = current_scene

        print('Shape all (no mosaic)', all_scenes.shape)

    if aicsczi.is_mosaic():

        # get data for 1st scene and create the required shape for all scenes
        scene = CZIScene(aicsczi, sceneindex=0)
        shape_all = scene.shape_single_scene
        shape_all[scene.posS] = md.dims.SizeS
        print('Shape all Scenes : ', shape_all)
        print('DimString all Scenes : ', scene.single_scene_dimstr)

        # create empty array to hold all scenes
        all_scenes = np.empty(shape_all, dtype=md.npdtype)

        # loop over scenes if CZI is not Mosaic
        for s in range(md.dims.SizeS):
            scene = CZIScene(aicsczi, sceneindex=s)

            # create a slice object for all_scenes array
            if not scene.isRGB:
                idl_all = [slice(None, None, None)] * (len(all_scenes.shape) - 2)
            if scene.isRGB:
                idl_all = [slice(None, None, None)] * (len(all_scenes.shape) - 3)

            # update the entry with the current S index
            idl_all[scene.posS] = s

            # in case T-Z-H dimension are found
            if scene.hasT is True and scene.hasZ is True and scene.hasH is True:

                # read array for the scene
                for h, t, z, c in product(range(scene.sizeH),
                                          range(scene.sizeT),
                                          range(scene.sizeZ),
                                          range(scene.sizeC)):
                    # read the array for the 1st scene using the ROI
                    scene_array_htzc = aicsczi.read_mosaic(region=(scene.xstart,
                                                                   scene.ystart,
                                                                   scene.width,
                                                                   scene.height),
                                                           scale_factor=1.0,
                                                           H=h,
                                                           T=t,
                                                           Z=z,
                                                           C=c)

                    print('Shape Single Scene : ', scene_array_htzc.shape)
                    print('Min-Max Single Scene : ', np.min(scene_array_htzc), np.max(scene_array_htzc))

                    # create slide object for the current mosaic scene
                    # idl_scene = [slice(None, None, None)] * (len(scene.shape_single_scene) - 2)
                    idl_all[scene.posS] = s
                    idl_all[scene.posH] = h
                    idl_all[scene.posT] = t
                    idl_all[scene.posZ] = z
                    idl_all[scene.posC] = c

                    # cast the current scene into the stack for all scenes
                    all_scenes[tuple(idl_all)] = scene_array_htzc

            if scene.hasT is False and scene.hasZ is False:

                # create an array for the scene
                for c in range(scene.sizeC):
                    scene_array_c = aicsczi.read_mosaic(region=(scene.xstart,
                                                                scene.ystart,
                                                                scene.width,
                                                                scene.height),
                                                        scale_factor=1.0,
                                                        C=c)

                    print('Shape Single Scene : ', scene_array_c.shape)

                    idl_all[scene.posS] = s
                    idl_all[scene.posC] = c

                    # cast the current scene into the stack for all scenes
                    all_scenes[tuple(idl_all)] = scene_array_c

    return all_scenes



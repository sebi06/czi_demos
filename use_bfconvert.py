import os
from pathlib import Path


def get_fname_woext(filepath):

    # create empty string
    real_extension = ''

    # get all part of the file extension
    sufs = Path(filepath).suffixes
    for s in sufs:
        real_extension = real_extension + s

    # remover real extension from filepath
    filepath_woext = filepath.replace(real_extension, '')

    return filepath_woext


def convert_to_ometiff(imagefilepath,
                       bftoolsdir='/Users/bftools',
                       czi_include_attachments=False,
                       czi_autostitch=True,
                       verbose=True):

    # check if path exits
    if not os.path.exists(bftoolsdir):
        print('No bftools dirctory found. Nothing will be converted')
        file_ometiff = None
        file_omexml = None

    if os.path.exists(bftoolsdir):

        # set working dir
        os.chdir(bftoolsdir)

        # get the imagefile path without extension
        imagefilepath_woext = get_fname_woext(imagefilepath)

        # create imagefile path for OME.TIFF and OME.XML
        file_ometiff = imagefilepath_woext + '.ome.tiff'
        file_omexml = imagefilepath_woext + '.ome.xml'

        # create cmdstring for CZI files- mind the spaces !!!
        if imagefilepath.lower().endswith('.czi'):

            # configure the CZI options
            if czi_include_attachments:
                czi_att = 'true'
            if not czi_include_attachments:
                czi_att = 'false'

            if czi_autostitch:
                czi_stitch = 'true'
            if not czi_autostitch:
                czi_stitch = 'false'

            # create cmdstring - mind the spaces !!!
            cmdstring = 'bfconvert -no-upgrade -option zeissczi.attachments ' + czi_att + ' -option zeissczi.autostitch ' + \
                czi_stitch + ' "' + imagefilepath + '" "' + file_ometiff + '"'

        else:
            # create cmdstring for non-CZIs- mind the spaces !!!
            cmdstring = 'bfconvert -no-upgrade' + ' "' + imagefilepath + '" "' + file_ometiff + '"'

        if verbose:
            print('Original ImageFile : ', imagefilepath_woext)
            print('ImageFile OME.TIFF : ', file_ometiff)
            print('ImageFile OEM.XML  : ', file_omexml)
            print('Use CMD : ', cmdstring)

        # run the bfconvert tool with the specified parameters
        os.system(cmdstring)
        print('Done.')

    return file_ometiff, file_omexml


bfconvert_path = r'c:\Users\m1srh\Documents\Software\Bioformats\6.5.0\bftools\bftools'
file_orig = r'C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\OME_TIFF_Convert\test_bfconvert\CH=3.czi'

file_ometiff, file_omexml = convert_to_ometiff(file_orig,
                                               bftoolsdir=bfconvert_path,
                                               czi_autostitch=False,
                                               czi_include_attachments=False,
                                               verbose=True)

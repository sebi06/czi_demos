import os
from pathlib import Path


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


def convert_to_ometiff(imagefilepath,
                       bftoolsdir='/Users/bftools',
                       czi_include_attachments=False,
                       czi_autostitch=True,
                       verbose=True):
    """Convert image file using bfconvert tool into a OME-TIFF from with a python script.

    :param imagefilepath: path to imagefile
    :type imagefilepath: str
    :param bftoolsdir: bftools directory containing the bfconvert, defaults to '/Users/bftools'
    :type bftoolsdir: str, optional
    :param czi_include_attachments: option convert a CZI attachment (if CZI), defaults to False
    :type czi_include_attachments: bool, optional
    :param czi_autostitch: option stich a CZI, defaults to True
    :type czi_autostitch: bool, optional
    :param verbose: show additional output, defaults to True
    :type verbose: bool, optional
    :return: fileparh of created OME-TIFF file
    :rtype: str
    """
    # check if path exits
    if not os.path.exists(bftoolsdir):
        print('No bftools dirctory found. Nothing will be converted')
        file_ometiff = None

    if os.path.exists(bftoolsdir):

        # set working dir
        os.chdir(bftoolsdir)

        # get the imagefile path without extension
        imagefilepath_woext = get_fname_woext(imagefilepath)

        # create imagefile path for OME-TIFF
        file_ometiff = imagefilepath_woext + '.ome.tiff'

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
            print('Use CMD : ', cmdstring)

        # run the bfconvert tool with the specified parameters
        os.system(cmdstring)
        print('Done.')

    return file_ometiff


bfconvert_path = r'c:\Users\m1srh\Documents\Software\Bioformats\6.5.0\bftools\bftools'
file_orig = r'C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\OME_TIFF_Convert\test_bfconvert\CH=3.czi'

file_ometiff, file_omexml = convert_to_ometiff(file_orig,
                                               bftoolsdir=bfconvert_path,
                                               czi_autostitch=False,
                                               czi_include_attachments=False,
                                               verbose=True)

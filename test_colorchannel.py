import numpy as np
#from czitools import imgfileutils as imf
import imgfileutils as imf


def get_scalefactor(metadata):
    """Add scaling factors to the metadata dictionary

    :param metadata: dictionary with CZI or OME-TIFF metadata
    :type metadata: dict
    :return: dictionary with additional keys for scling factors
    :rtype: dict
    """

    # set default scale factore to 1
    scalefactors = {'xy': 1.0,
                    'zx': 1.0
                    }

    try:
        # get the factor between XY scaling
        scalefactors['xy'] = metadata['XScale'] / metadata['YScale']
        # get the scalefactor between XZ scaling
        scalefactors['zx'] = metadata['ZScale'] / metadata['YScale']
    except KeyError as e:
        print('Key not found: ', e)

    return scalefactors


#filename_czi = r"C:\Users\m1srh\Documents\GitHub\ipy_notebooks\Read_OMETIFF_CZI\testdata\CellDivision_T=10_Z=15_CH=2_DCV_small.czi"
filename_czi = r'E:\tuxedo\testpictures\Testdata_Zeiss\celldivision\CellDivision_T=10_Z=15_CH=2_DCV_small.czi'

print('---------- CZI ----------')
md_czi, addmd_czi = imf.get_metadata(filename_czi)
sf_czi = get_scalefactor(md_czi)

print(md_czi['ChannelColors'])

import numpy as np
from czitools import imgfileutils as imf


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


filename_czi = r"C:\Users\m1srh\Documents\GitHub\ipy_notebooks\Read_OMETIFF_CZI\testdata\CellDivision_T=10_Z=15_CH=2_DCV_small.czi"
filename_ometiff = r"C:\Users\m1srh\Documents\GitHub\ipy_notebooks\Read_OMETIFF_CZI\testdata\CellDivision_T=10_Z=15_CH=2_DCV_small.ome.tiff"

print('---------- CZI ----------')
md_czi, addmd_czi = imf.get_metadata(filename_czi)
sf_czi = get_scalefactor(md_czi)

print('XScale', md_czi['XScale'])
print('YScale', md_czi['YScale'])
print('ZScale', md_czi['ZScale'])
print('SF CZI', sf_czi)

print('---------- OME-TIFF ----------')
md_ometiff, addmd_ometiff = imf.get_metadata(filename_ometiff)
sf_ometiff = get_scalefactor(md_ometiff)

print('XScale', md_ometiff['XScale'])
print('YScale', md_ometiff['YScale'])
print('ZScale', md_ometiff['ZScale'])
print('SF OME-TIFF', sf_ometiff)

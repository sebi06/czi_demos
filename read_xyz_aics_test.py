from aicsimageio import AICSImage, imread, imread_dask
import aicspylibczi
import imgfileutils as imf
import czi_tools as czt
import itertools as it
#import xmltodict
#from lxml import etree
from tqdm import tqdm
#from collections import defaultdict
import nested_dict as nd
import pandas as pd
import numpy as np
from datetime import datetime
import dateutil.parser as dt

"""
def norm_columns(df, colname='Time [s]', mode='min'):

    # normalize columns according to min or max value
    if mode == 'min':
        min_value = df[colname].min()
        df[colname] = df[colname] - min_value

    if mode == 'max':
        max_value = df[colname].max()
        df[colname] = df[colname] - max_value

    return df
"""

#filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\Castor\testwell96.czi"
filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\Atomic\Nuclei\nuclei_RGB\H&E\Tumor_H&E.czi"

df = czt.get_czi_planetable(filename)

print(df[:8])

"""
czi = aicspylibczi.CziFile(filename)

# Get the shape of the data
dimensions = czi.dims  # 'STCZMYX'

czi_libczi_size = czi.size

czi_libczi_shape = czi.dims_shape()

#print('dimensions_libczi : ', dimensions)
#print('czi_libczi_size : ', czi_libczi_size)
#print('czi_libczi_shape : ', czi_libczi_shape)
# print('IsMosaic : ', czi.is_mosaic())  # True

try:
    sizeM = czi_libczi_shape[0]['M'][1]
except KeyError as e:
    print('Key not found: ', e, 'Set SizeM = 1.')
    sizeM = 1

print(sizeM)

md, add = imf.get_metadata(filename)

print('SizeS : ', md['SizeS'])
print('SizeM : ', md['SizeM'])
print('SizeT : ', md['SizeT'])
print('SizeZ : ', md['SizeZ'])
print('SizeC : ', md['SizeC'])

# tileinfo_dict = nd.nested_dict()
sbcount = -1

df = pd.DataFrame(columns=['Subblock',
                           'Scene',
                           'Tile',
                           'T',
                           'Z',
                           'C',
                           'X [micron]',
                           'Y [micron]',
                           'Z [micron]',
                           'Time [s]',
                           'xstart',
                           'ystart',
                           'xwidth',
                           'ywidth'])

pbar = tqdm(total=md['SizeS'] * md['SizeM'] * md['SizeT'] * md['SizeZ'] * md['SizeC'])

if md['czi_ismosaic']:

    for s, m, t, z, c in it.product(range(md['SizeS']),
                                    range(md['SizeM']),
                                    range(md['SizeT']),
                                    range(md['SizeZ']),
                                    range(md['SizeC'])):

        sbcount += 1
        # print(s, m, t, z, c)
        info = czi.read_subblock_rect(S=s, M=m, T=t, Z=z, C=c)
        # tileinfo_dict[s][m][t][z][c] = info

        # read information from subblock
        sb = czi.read_subblock_metadata(unified_xml=True, B=0, S=s, M=m, T=t, Z=z, C=c)

        time = sb.xpath('//AcquisitionTime')[0].text
        timestamp = dt.parse(time).timestamp()
        # sb_exp = sb.xpath('//ExposureTime').text
        # framesize = sb.xpath('//Frame')[0].text
        xpos = np.double(sb.xpath('//StageXPosition')[0].text)
        ypos = np.double(sb.xpath('//StageYPosition')[0].text)
        zpos = np.double(sb.xpath('//FocusPosition')[0].text)

        df = df.append({'Subblock': sbcount,
                        'Scene': s,
                        'Tile': m,
                        'Time': t,
                        'Z': z,
                        'C': c,
                        'X [micron]': xpos,
                        'Y [micron]': ypos,
                        'Z [micron]': zpos,
                        'Time [s]': timestamp,
                        'xstart': info[0],
                        'ystart': info[1],
                        'xwidth': info[2],
                        'ywidth': info[3]},
                       ignore_index=True)

        pbar.update(1)

pbar.close()

if not md['czi_ismosaic']:

    for s, t, z, c in it.product(range(md['SizeS']),
                                 range(md['SizeT']),
                                 range(md['SizeZ']),
                                 range(md['SizeC'])):

        sbcount += 1
        # print(s, t, z, c)
        info = czi.read_subblock_rect(S=s, T=t, Z=z, C=c)
        # tileinfo_dict[s][t][z][c] = info

        # read information from subblocks
        sb = czi.read_subblock_metadata(unified_xml=True, B=0, S=s, T=t, Z=z, C=c)

        time = sb.xpath('//AcquisitionTime')[0].text
        timestamp = dt.parse(time).timestamp()
        # sb_exp = sb.xpath('//ExposureTime').text
        framesize = sb.xpath('//Frame')[0].text
        xpos = np.double(sb.xpath('//StageXPosition')[0].text)
        ypos = np.double(sb.xpath('//StageYPosition')[0].text)
        zpos = np.double(sb.xpath('//FocusPosition')[0].text)

        df = df.append({'Subblock': sbcount,
                        'Scene': s,
                        'Tile': 0,
                        'T': t,
                        'Z': z,
                        'C': c,
                        'X [micron]': xpos,
                        'Y [micron]': ypos,
                        'Z [micron]': zpos,
                        'Time [s]': timestamp,
                        'xstart': info[0],
                        'ystart': info[1],
                        'xwidth': info[2],
                        'ywidth': info[3]},
                       ignore_index=True)

        pbar.update(1)

pbar.close()

print('Done.')

df = imf.norm_columns(df, colname='Time [s]', mode='min')

df = df.astype({'Subblock': 'int32',
                'Scene': 'int32',
                'Tile': 'int32',
                'T': 'int32',
                'Z': 'int32',
                'C': 'int16',
                'xstart': 'int32',
                'xstart': 'int32',
                'ystart': 'int32',
                'xwidth': 'int32',
                'ywidth': 'int32'}, copy=False)


print(df[:8])
"""

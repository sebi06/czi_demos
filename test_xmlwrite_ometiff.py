
#import warnings
# warnings.filterwarnings('ignore')
# warnings.simplefilter('ignore')

#import czifile as zis
# from apeer_ometiff_library import io, processing  # , omexmlClass
#import os
#from matplotlib import pyplot as plt, cm
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#from skimage.viewer import ImageViewer
#import skimage.io
#import matplotlib.colors as colors
#import numpy as np
#import ipywidgets as widgets

import imgfileutils as imf

# define your testfiles here
filename = r'testdata/CellDivision_T=10_Z=15_CH=2_DCV_small.ome.tiff'

md, addmd = imf.get_metadata(filename)

# or much shorter
xmlometiff = imf.writexml_ometiff(filename)
print(xmlometiff)

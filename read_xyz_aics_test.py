import os
import itertools as it
from tqdm import tqdm, trange
from tqdm.contrib.itertools import product
import dateutil.parser as dt
import matplotlib.pyplot as plt
from matplotlib import cm, use
import visutools as vt
import czi_tools as czt

# use specific plotting backend
use('Qt5Agg')

filepath = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\Castor\testwell96.czi"
#filepath = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\Atomic\Nuclei\nuclei_RGB\H+E\Tumor_H+E.czi"

print('--------------------------------------------------')
print('FilePath : ', filepath)
print(os.getcwd())
print('File exists : ', os.path.exists(filepath))
print('--------------------------------------------------')

# define dimension indices
S = 0
T = 0
Z = 0
C = 0

# define the plot type and output type
plot_type = 'html'
separator = ','

# define plot parameters
msz2d = 70
msz3d = 10
normz = False

if plot_type == 'mpl':
    saveformat = 'png'
if plot_type == 'html':
    saveformat = 'html'

# define name for figure to be saved
filename = os.path.basename(filepath)
fig1savename = os.path.splitext(filename)[0] + '_XYZ-Pos' + '.' + saveformat
fig2savename = os.path.splitext(filename)[0] + '_XYZ-Pos3D' + '.' + saveformat

# get the planetable from CZI file
planetable = czt.get_czi_planetable(filepath)

# filter the planetable for S, T, Z and C entry
planetable_filtered = czt.filterplanetable(planetable, S=S, T=T, Z=Z, C=C)

if plot_type == 'mpl':
    # display the XYZ positions using matplotlib
    fig1, fig2 = vt.scatterplot_mpl(planetable_filtered,
                                    S=S, T=T, Z=Z, C=C,
                                    msz2d=msz2d,
                                    normz=normz,
                                    fig1savename=fig1savename,
                                    fig2savename=fig2savename,
                                    msz3d=msz3d)

if plot_type == 'html':
    # display the XYZ positions using plotly
    fig1, fig2 = vt.scatterplot_plotly(planetable_filtered,
                                       S=S, T=T, Z=Z, C=C,
                                       msz2d=msz2d,
                                       normz=normz,
                                       fig1savename=fig1savename,
                                       fig2savename=fig2savename,
                                       msz3d=msz3d)
    fig1.show()
    fig2.show()

# write the planetable to a csv
print('Write to CSV File : ', filename)
csvfile = czt.save_planetable(planetable, filename,
                              separator=separator,
                              index=False)

plt.show()

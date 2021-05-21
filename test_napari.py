
from tools import imgfile_tools as imf
from tools import napari_tools as nap
from aicsimageio import AICSImage, imread, imread_dask
from apeer_ometiff_library import io, processing, omexmlClass
import napari
from magicgui import magicgui
from napari.types import ImageData


@magicgui(call_button='Save Image')
def savelayer(layernames={"choices" : layernames}):
    """
    Save a given layer based on the current slider postion.
    """

    # get the slider position
    sliderpos = viewer.dims.point
    sliderpos = list(map(int, sliderpos))
    print('Current_Slider_Position')

    dimstring = '_'

    # create fanyc image name
    for i in range(len(sliderpos)):

        numstring = imf.addzeros((sliderpos[i]))
        dimstring = dimstring + md['Axes_aics'][i] + '_' + numstring

        print(dimstring)

    return dimstring


#filename = r"C:\Testdata_Zeiss\Castor\Z-Stack_DCV\NeuroSpheres_DCV.czi"
filename = r"D:\Testdata_Zeiss\AiryScan\FoLu_mCherryEB3_GFPMito_2_Airyscan Processing.czi"
#filename = r"C:\Users\m1srh\Documents\GitHub\czi_demos\testdata\CellDivision_T=10_Z=15_CH=2_DCV_small.ome.tiff"

md, addmd = imf.get_metadata(filename)

img = AICSImage(filename)
stack = img.get_image_data()

# Return value is an array of order (T, Z, C, X, Y)
#stack, omexml = io.read_ometiff(filename)

viewer = napari.Viewer()
viewer.window.add_dock_widget(savelayer, area='right')

image_layers = nap.show_napari(viewer, stack, md,
                               blending='additive',
                               gamma=0.85,
                               adjust_contrast=False,
                               add_mdtable=False,
                               rename_sliders=True)

layernames = []
for ln in viewer.layers:
    layernames.append(ln.name)




napari.run()


####################


out = save_image_label_pair(md, viewer.layers, imagelayer='CH1', labellayer='Labels')

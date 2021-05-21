import napari
import numpy as np
from skimage.io import imread
from skimage.filters import gaussian

viewer = napari.Viewer()

image = imread('testdata/nuctest01.ome.tiff')

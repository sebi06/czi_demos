from stardist.models import StarDist2D
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, measure, segmentation
from skimage import exposure
from skimage.exposure import rescale_intensity
from skimage.morphology import white_tophat, black_tophat, disk, square, ball, closing, square, dilation
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.morphology import binary_opening
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu, threshold_triangle, rank, median, gaussian, sobel
from skimage.segmentation import clear_border, watershed, random_walker
from skimage.color import label2rgb
from skimage.util import invert
from skimage.transform import rescale, resize
import scipy.ndimage as ndimage
from pylibCZIrw import czi as pyczi
import pylibczirw_metadata as czimd
from aicsimageio import AICSImage, imread
from aicsimageio.writers import ome_tiff_writer as otw
import os
from tqdm import tqdm
from tqdm.contrib import itertools as it
from typing import List, Dict, Tuple, Optional, Type, Any, Union
from dataclasses import dataclass, field
from cztile.fixed_total_area_strategy import AlmostEqualBorderFixedTotalAreaStrategy2D
import segmentation_tools as sgt
import segmentation_stardist as sg_sd


def process_labels(int_image: np.ndarray,
                   labels: np.ndarray,
                   use_splitobjects=True,
                   use_watershed: bool = True,
                   min_distance=5,
                   erode: bool = False,
                   num_erode: int = 1,
                   relabel: bool = False,
                   verbose=False) -> Tuple[np.ndarray, np.ndarray]:

    if use_splitobjects:
        labels = sgt.split_touching_objects(labels) * 1

    if use_watershed:
        #labels = sgt.apply_watershed(labels, min_distance=min_distance)

        labels = sgt.apply_watershed_adv(np.squeeze(int_image),
                                         labels,
                                         filtermethod_ws='median',
                                         filtersize_ws=7,
                                         min_distance=min_distance,
                                         radius=7)

    if erode:
        labels = sgt.erode_labels(labels, num_erode, relabel=relabel)

    if verbose:
        print("New labels info: ", labels.min(), labels.max(), labels.shape, labels.dtype)

    # convert to desired type
    background = invert(labels)

    # return np.squeeze(labels), np.squeeze(background)
    return labels, background


def show_plot(img: np.ndarray, labels: np.ndarray, new_labels: np.ndarray) -> None:

    # show the results
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("input")
    #ax[1].imshow(render_label(labels.astype(np.uint16), img=img))
    #ax[1].set_title("pred + input")
    #ax[1].imshow(render_label(new_labels, img=img))
    #ax[1].set_title("pred + input + erode")
    ax[1].imshow(new_labels)
    ax[1].set_title("pred + erode")
    plt.show()


def save_OMETIFF(img_FL: np.ndarray,
                 img_TL: np.ndarray,
                 new_labels: np.ndarray,
                 background: np.ndarray,
                 savepath_FL: str = "DAPI.ome.tiff",
                 savepath_TL: str = "PGC.ome.tiff",
                 savepath_NUC: str = "PGC_nuc.ome.tiff",
                 savepath_BGRD: str = "PGC_background.ome.tiff",
                 pixels_physical_sizes: List[float] = [1.0, 1.0, 1.0],
                 channel_names: Dict[str, str] = {"FL": "FL", "TL": "TL", "NUC": "NUC", "BGRD": "BGRD"}) -> None:

    # write the array as an OME-TIFF incl. the metadata for the labels
    otw.OmeTiffWriter.save(img_FL, savepath_FL,
                           channel_names=channel_names["FL"],
                           pixels_physical_sizes=pixels_physical_sizes,
                           dim_order="YX")

    # write the array as an OME-TIFF incl. the metadata for the labels
    otw.OmeTiffWriter.save(img_TL, savepath_TL,
                           channel_names=channel_names["TL"],
                           pixels_physical_sizes=pixels_physical_sizes,
                           dim_order="YX")

    # save the label
    otw.OmeTiffWriter.save(new_labels, savepath_NUC,
                           channel_names=channel_names["NUC"],
                           pixels_physical_sizes=pixels_physical_sizes,
                           dim_order="YX")

    # save the background
    otw.OmeTiffWriter.save(background, savepath_BGRD,
                           channel_names=["BGRD"],
                           pixels_physical_sizes=pixels_physical_sizes,
                           dim_order="YX")


##########################################################################
plot = False
basefolder = r"data"
dir_FL = os.path.join(basefolder, "fluo")
dir_LABEL = os.path.join(basefolder, "label")
dir_TL = os.path.join(basefolder, "trans")

os.makedirs(dir_FL, exist_ok=True)
os.makedirs(dir_LABEL, exist_ok=True)
os.makedirs(dir_TL, exist_ok=True)

suffix_orig = ".ome.tiff"
suffix_NUC = "_nuc.ome.tiff"
suffix_BGRD = "_background.ome.tiff"
use_tiles = False
target_scaleXY = 0.5
tilesize = 400
erode = False
num_erode = 1
rescale_image = False

# prints a list of available models
# StarDist2D.from_pretrained()
model = StarDist2D.from_pretrained('2D_versatile_fluo')

stardist_prob_thresh = 0.5
stardist_overlap_thresh = 0.2
stardist_overlap_label = None  # 0 is not supported yet
stardist_norm = True
stardist_norm_pmin = 1
stardist_norm_pmax = 99.8
stardist_norm_clip = False
area_min = 20
area_max = 1000000

use_watershed = False
min_distance = 10

ch_id_FL = 0
ch_id_TL = 1

ext = ".czi"

# iterating over all files
for file in os.listdir(basefolder):
    if file.endswith(ext):

        print("Processing CZI file:", file)

        cziname = file
        cziname_NUC = file[:-4] + "_onlyFL"
        cziname_TL = file[:-4] + "_onlyTL"

        # get the scaling from the CZI
        cziscale = czimd.CziScaling(os.path.join(basefolder, cziname))
        pixels_physical_sizes = [1, cziscale.X, cziscale.Y]
        scale_forward = target_scaleXY / cziscale.X
        new_shapeXY = int(np.round(tilesize * scale_forward, 0))

        # open a CZI instance to read and in parallel one to write
        with pyczi.open_czi(os.path.join(basefolder, file)) as czidoc_r:

            if use_tiles:

                tilecounter = 0

                # create a "tile" by specifying the desired tile dimension and minimum required overlap
                tiler = AlmostEqualBorderFixedTotalAreaStrategy2D(total_tile_width=tilesize,
                                                                  total_tile_height=tilesize,
                                                                  min_border_width=8)

                # get the size of the bounding rectangle for the scene
                tiles = tiler.tile_rectangle(czidoc_r.scenes_bounding_rectangle[0])

                # show the created tile locations
                for tile in tiles:
                    print(tile.roi.x, tile.roi.y, tile.roi.w, tile.roi.h)

                # loop over all tiles created by the "tiler"
                for tile in tqdm(tiles):

                    # read a specific tile from the CZI using the roi parameter
                    tile2d_FL = czidoc_r.read(plane={"C": ch_id_FL}, roi=(tile.roi.x, tile.roi.y, tile.roi.w, tile.roi.h))
                    tile2d_TL = czidoc_r.read(plane={"C": ch_id_TL}, roi=(tile.roi.x, tile.roi.y, tile.roi.w, tile.roi.h))

                    if rescale_image:
                        # scale the FL image to 0.5 micron per pixel (more or less)
                        tile2d_FL_scaled = resize(tile2d_FL, (new_shapeXY, new_shapeXY), preserve_range=True, anti_aliasing=True)

                        # get the prediction for the current tile
                        # labels, polys = model.predict_instances(normalize(tile2d_FL))  # , n_tiles=(2, 2))  # int32
                        labels_scaled = sg_sd.segment_nuclei_stardist(tile2d_FL, model,
                                                                      prob_thresh=stardist_prob_thresh,
                                                                      overlap_thresh=stardist_overlap_thresh,
                                                                      overlap_label=stardist_overlap_label,
                                                                      # norm=stardist_norm,
                                                                      norm_pmin=stardist_norm_pmin,
                                                                      norm_pmax=stardist_norm_pmax,
                                                                      norm_clip=stardist_norm_clip)

                        # scale the label image back to the original size preserving the label values
                        labels = resize(labels_scaled, (tilesize, tilesize), anti_aliasing=False, preserve_range=True).astype(int)

                    if not rescale_image:

                        # get the prediction for the current tile
                        # labels, polys = model.predict_instances(normalize(tile2d_FL))  # , n_tiles=(2, 2))  # int32
                        labels = sg_sd.segment_nuclei_stardist(tile2d_FL, model,
                                                               prob_thresh=stardist_prob_thresh,
                                                               overlap_thresh=stardist_overlap_thresh,
                                                               overlap_label=stardist_overlap_label,
                                                               # norm=stardist_norm,
                                                               norm_pmin=stardist_norm_pmin,
                                                               norm_pmax=stardist_norm_pmax,
                                                               norm_clip=stardist_norm_clip)

                    # filter labels by size
                    labels = sgt.area_filter(labels, area_min=area_min, area_max=area_max)

                    new_labels, background = process_labels(tile2d_FL, labels,
                                                            erode=erode,
                                                            num_erode=num_erode,
                                                            use_watershed=use_watershed,
                                                            min_distance=min_distance,
                                                            relabel=False,
                                                            verbose=False)

                    show_plot(tile2d_FL, labels, np.squeeze(new_labels))

                    # save the original FL channel as OME-TIFF
                    savepath_FL = os.path.join(dir_FL, cziname_NUC + "_t" + str(tilecounter) + suffix_orig)

                    # save the original TL (PGC etc. ) channel as OME_TIFF
                    savepath_TL = os.path.join(dir_TL, cziname_TL + "_t" + str(tilecounter) + suffix_orig)

                    # save the labels for the nucleus and the background as OME-TIFF
                    savepath_BGRD = os.path.join(dir_LABEL, cziname_TL + "_t" + str(tilecounter) + suffix_BGRD)
                    savepath_NUC = os.path.join(dir_LABEL, cziname_TL + "_t" + str(tilecounter) + suffix_NUC)

                    # save the OME-TIFFs
                    save_OMETIFF(tile2d_FL, tile2d_TL, new_labels, background,
                                 savepath_FL=savepath_FL,
                                 savepath_TL=savepath_TL,
                                 savepath_NUC=savepath_NUC,
                                 savepath_BGRD=savepath_BGRD,
                                 pixels_physical_sizes=pixels_physical_sizes)

                    tilecounter += 1

            if not use_tiles:

                # read a specific tile from the CZI using the roi parameter
                tile2d_FL = czidoc_r.read(plane={"C": ch_id_FL})[..., 0]
                tile2d_TL = czidoc_r.read(plane={"C": ch_id_TL})[..., 0]

                # get the prediction for the current tile
                # labels, polys = model.predict_instances(normalize(tile2d_FL))  # , n_tiles=(2, 2))  # int32
                labels = sg_sd.segment_nuclei_stardist(tile2d_FL, model,
                                                       prob_thresh=stardist_prob_thresh,
                                                       overlap_thresh=stardist_overlap_thresh,
                                                       overlap_label=stardist_overlap_label,
                                                       # norm=stardist_norm,
                                                       norm_pmin=stardist_norm_pmin,
                                                       norm_pmax=stardist_norm_pmax,
                                                       norm_clip=stardist_norm_clip)

                # filter labels by size
                labels = sgt.area_filter(labels, area_min=area_min, area_max=area_max)

                new_labels, background = process_labels(tile2d_FL, labels,
                                                        erode=erode,
                                                        num_erode=num_erode,
                                                        use_watershed=use_watershed,
                                                        min_distance=min_distance,
                                                        relabel=False,
                                                        verbose=False)

                show_plot(tile2d_FL, labels, new_labels)

                # save the original FL channel as OME-TIFF
                savepath_FL = os.path.join(dir_FL, cziname_NUC[:-4] + suffix_orig)

                # save the original TL (PGC etc. ) channel as OME_TIFF
                savepath_TL = os.path.join(dir_TL, cziname_TL[:-4] + suffix_orig)

                # save the labels for the nucleus and the background as OME-TIFF
                savepath_BGRD = os.path.join(dir_LABEL, cziname_TL[:-4] + suffix_BGRD)
                savepath_NUC = os.path.join(dir_LABEL, cziname_TL[:-4] + suffix_NUC)

                # save the OME-TIFFs
                save_OMETIFF(tile2d_FL, tile2d_TL, new_labels, background,
                             savepath_FL=savepath_FL,
                             savepath_TL=savepath_TL,
                             savepath_NUC=savepath_NUC,
                             savepath_BGRD=savepath_BGRD,
                             pixels_physical_sizes=pixels_physical_sizes)

    else:
        continue

print("Done.")

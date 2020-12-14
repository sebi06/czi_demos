from aicspylibczi import CziFile
import imgfileutils as imf

filename = r"input\Tumor_H+E_small2.czi"

czi = CziFile(filename)

bboxes = czi.mosaic_scene_bounding_boxes(index=0)

xmin = 0
ymin = 0
xmax = 0
ymax = 0

for b in range(len(bboxes)):
    box = bboxes[b]
    # check first bbox
    if box[0] < box[0]:
        xmin = box[0]

    if box[1] < ymin:
        ymin = box[1]

    if box[0] + box[2] > xmax:
        xmax = box[0] + box[2]

    if box[1] + box[3] > ymax:
        ymax = box[1] + box[3]

print(xmin, ymin, xmax, ymax)

out = imf.get_scene_extend_czi(czi, sceneindex=0)

print(out)

size = czi.read_mosaic_size()

print(size)

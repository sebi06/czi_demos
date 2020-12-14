from aicspylibczi import CziFile
import imgfileutils as imf
import czi_tools as czt

#filename = r"input\Tumor_H+E_small2.czi"
filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\W96_B2+B4_S=2_T=2=Z=4_C=3_Tile=5x9.czi"

# get the metadata from the czi file
md, additional_mdczi = imf.get_metadata(filename)

# read CZI using aicslibczi
czi = CziFile(filename)

# get the bounding boxes
bboxes = czi.mosaic_scene_bounding_boxes(index=0)

#xmin = 0
#ymin = 0
#xmax = 0
#ymax = 0
"""
for b in range(len(bboxes)):
    box = bboxes[b]
    # check first bbox
    # if box[0] < box[0]:
    if box[0] < xmin:
        xmin = box[0]

    if box[1] < ymin:
        ymin = box[1]

    if box[0] + box[2] > xmax:
        xmax = box[0] + box[2]

    if box[1] + box[3] > ymax:
        ymax = box[1] + box[3]

print(xmin, ymin, xmax, ymax)
"""

xmin = []
ymin = []
xmax = []
ymax = []

for b in range(len(bboxes)):
    # get the bounding box for a tile
    box = bboxes[b]
    xmin.append(box[0])
    ymin.append(box[1])
    xmax.append(box[0] + box[2])
    ymax.append(box[1] + box[3])

XMIN = min(xmin)
YMIN = min(ymin)
XMAX = max(xmax)
YMAX = max(ymax)

out = czt.get_scene_extend_czi(czi, sceneindex=0)
print(out)
size = czi.read_mosaic_size()
print(size)

# read sizes for all scenes
for s in range(md['SizeS']):
    out = czt.get_scene_extend_czi(czi, sceneindex=s)
    print('BBox Scene:', s, ' : ', out)

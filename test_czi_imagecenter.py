import numpy as np
import czifile as zis
import imgfileutils as imf

#filename = r"C:\Temp\input\DTScan_ID4.czi"
#filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\CZI_Testfiles\W96_B2+B4_S=2_T=2=Z=4_C=3_Tile=4x8.czi"
filename = r"C:\Users\m1srh\Documents\ImageSC\P2A_B6_M1_GIN-0004-Scene-2-ScanRegion1_unc.czi"
#filename = r"C:\Users\m1srh\Documents\ImageSC\P2A_B6_M1_GIN-0004-Scene-2-ScanRegion1_AF750.czi"
#filename = r"C:\Users\m1srh\Documents\ImageSC\P2A_B6_M1_GIN-0004-Scene-2-ScanRegion1_AF594.czi"
#filename = r"C:\Users\m1srh\Documents\ImageSC\P2A_B6_M1_GIN-0004-Scene-2-ScanRegion1_AF488.czi"
#filename = r"C:\Users\m1srh\Documents\ImageSC\P2A_B6_M1_GIN-0004-Scene-2-ScanRegion1_DAPI.czi"

# get CZI object
czi = zis.CziFile(filename)

# parse the XML into a dictionary
metadatadict_czi = czi.metadata(raw=False)

# get the complete scene information
allscenes = metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['S']['Scenes']['Scene']
centerposX = []
centerposY = []

# check if there is a list of scenes
if isinstance(allscenes, list):
    for s in range(len(allscenes)):
        for k, v in allscenes[s].items():
            print(k, ':', v)
        # get the SceneCenter Position for all scenes
        centerposX.append(np.double(allscenes[s]['CenterPosition'].split(',')[0]))
        centerposY.append(np.double(allscenes[s]['CenterPosition'].split(',')[1]))
# and in case of only one scence (= allscenes is not a list)
if not isinstance(allscenes, list):
    for k, v in allscenes.items():
        print(k, ':', v)
    centerposX.append(np.double(allscenes['CenterPosition'].split(',')[0]))
    centerposY.append(np.double(allscenes['CenterPosition'].split(',')[1]))

# show the positions
print(centerposX)
print(centerposY)

######################################

# get the metadata from the czi file
md, additional_mdczi = imf.get_metadata(filename)

print('StageCenterX : ', md['SceneStageCenterX'])
print('StageCenterY : ', md['SceneStageCenterY'])

import imgfileutils as imf

filename = r"C:\Users\m1srh\OneDrive - Carl Zeiss AG\Testdata_Zeiss\LatticeLightSheet\LS_Mitosis_T=150-300.czi"

md, addmd = imf.get_metadata(filename)

print(md['ObjNA'])
print(md['ObjMag'])
print(md['ObjID'])
print(md['ObjName'])
print(md['ObjImmersion'])
print(md['TubelensMag'])
print(md['ObjNominalMag'])

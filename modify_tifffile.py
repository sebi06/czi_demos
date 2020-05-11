import ometifftools as ott

filename = r'A01_S0000.ome.tiff'

old = ("2012-03", "2013-06", r"ome/2016-06")
new = ("2016-06", "2016-06", r"OME/2016-06")
ott.correct_omeheader(filename, old, new)

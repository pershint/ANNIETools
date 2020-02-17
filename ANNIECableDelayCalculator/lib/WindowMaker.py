# coding: utf-8
#################################################
# Prints out each channel's LED window in ADC counts #
# INPUT: PeakFitResults.json produced after running #
# main_LEDWindows.py #

import json
with open("./PeakFitResults.json","r") as f:
    dat = json.load(f)

channels = dat["channel"]
peaks = dat["mu"]
import math
for j,c in enumerate(channels):
    if math.isnan(peaks[j]):
        print("%s,%i,%i"%(c,int(peaks[j-6]/2)-10,int(peaks[j-6]/2)+30))
    else:
        print("%s,%i,%i"%(c,int(peaks[j]/2)-10,int(peaks[j]/2)+30))

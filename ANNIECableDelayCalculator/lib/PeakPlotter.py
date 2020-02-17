import numpy as np
import json
import ROOT
import sys
import matplotlib.pyplot as plt
import math

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import numpy as np
import math

sns.set_context('poster')
sns.set(font_scale=2.0)
sns.set_style("whitegrid")
sns.axes_style("darkgrid")

def GetBins(rootfile,hist_name):
    thehist = rootfile.Get(hist_name)
    bin_centers, evts,evts_unc =(), (), () #pandas wants ntuples
    fit_bin_centers, fit_evts,fit_evts_unc =(), (), () #pandas wants ntuples
    for i in xrange(int(thehist.GetNbinsX()+1)):
        if i==0:
            continue
        bin_centers =  bin_centers + ((float(thehist.GetBinWidth(i))/2.0) + float(thehist.GetBinLowEdge(i)),)
        evts = evts + (thehist.GetBinContent(i),)
        evts_unc = evts_unc + (thehist.GetBinError(i),)
        fit_bin_centers =  fit_bin_centers + ((float(thehist.GetBinWidth(i))/2.0) + float(thehist.GetBinLowEdge(i)),)
        fit_evts = fit_evts + (thehist.GetBinContent(i),)
        fit_evts_unc = fit_evts_unc + (thehist.GetBinError(i),)
    bin_centers = np.array(bin_centers)
    evts = np.array(evts)
    evts_unc = np.array(evts_unc)
    return bin_centers, evts, evts_unc


LEDMAP = {0:0, 1:1,2:2,3:3,4:4,5:5}
ROOTFILE = "./data/BeamRunTankOnly1378S0_PMTStability_Run0.root"
channel_list = np.arange(332,464,1)
window_toplot = [0,2000]

if __name__=='__main__':
    print(" ###### WELCOME TO ANNIE LED PEAK FINDER ###### ")
    WM_list = np.array([382,393,404,405])
    HMWBLUXETEL_list = np.setdiff1d(channel_list,WM_list)

    HMWBLUXETELInitialParams = [np.array([100,500,6,6]),np.array([100,640,6,6]),
                     np.array([100,750,6,6]),np.array([100,890,5,6]),
                     np.array([100,1040,6,6]),np.array([100,1180,5,6])]
    HMWBLUXETELLB = [np.array([0,450,0,0]),np.array([0,600,0,0]),
                     np.array([0,720,0,0]),np.array([0,860,0,0]),
                     np.array([0,1010,0,0]),np.array([0,1150,0,0])]
    HMWBLUXETELUB = [np.array([1E4,580,15,15]),np.array([1E4,690,15,15]),
                     np.array([1E4,780,15,15]),np.array([1E4,930,15,15]),
                     np.array([1E4,1070,15,15]),np.array([1E4,1210,15,15])]
    WMInitialParams = [np.array([100,840,6,6]),np.array([100,980,6,6]),
                      np.array([100,1090,6,6]),np.array([100,1225,5,5]),
                      np.array([100,1370,6,6]),np.array([100,1510,5,5])]
    WMLB = [np.array([0,810,0,0]),np.array([0,955,0,0]),
                      np.array([0,1050,0,0]),np.array([0,1200,0,0]),
                      np.array([0,1350,0,0]),np.array([0,1480,0,0])]
    WMUB = [np.array([1E4,860,15,15]),np.array([1E4,1000,15,15]),
                      np.array([1E4,1120,10,10]),np.array([1E4,1240,15,15]),
                      np.array([1E4,1400,15,15]),np.array([1E4,1540,15,15])]
    
    myfile = ROOT.TFile.Open(ROOTFILE)
    thehist_title = "hist_peaktime_CNUM" #FIXME: Make a configurable somehow
    #loop through each channel, show the histogram, and ask whether to use 
    #Default fitting params
    for channel_num in channel_list:
        title = thehist_title.replace("CNUM",str(channel_num))
        print("TITLE: " + str(title))
        ctrs,evts,evts_unc = GetBins(myfile,title)
        mybins = np.where((ctrs<window_toplot[1]) & (ctrs>window_toplot[0]))[0]
        ctrs = ctrs[mybins]
        evts = evts[mybins]
        evts_unc = evts_unc[mybins]
        plt.plot(ctrs,evts,marker='o',markersize=8,label=channel_num)
    plt.xlabel("Peak times (ns)")
    plt.ylabel("Peaks")
    plt.title("Distribution of peak times for channels")
    plt.show()

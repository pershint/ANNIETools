import numpy as np
import json
import ROOT
import sys
import matplotlib.pyplot as plt
import math

from . import HistogramBuilder as hb
from . import Maths as m

LEDMAP = {0:0,1:1,2:2,3:3,4:4,5:5}

#### OVERVIEW ####
#This script was originally used to estimate the cable delay differences 
#Between all PMTs using the hit time distributions from PMTs in an LED run.
#The general flow of the script is as follows:
#  1. Define the time windows in which LED pulses from a specific LED arrive
#     on each set of tubes.
#     #TODO: These will need to be adjusted, given a first-order cable delay
#             correction has been applied by Jonathan already to current data.
#  2. Load a ROOT file output by Michael's calibration tool from ToolAnalysis.
#     This file has hit time histograms for all Channels in a run.
#  3. For each time window in each tube, calculated the weighted mean and 
#     weighted standard deviation of the hit time distribution.  This 
#     estimates the mean LED pulse arrival time for each tube.
#

def EstimateCableDelays(myfile):
    #### 3 ####
    print(" ###### WELCOME TO ANNIE LED PEAK FINDER ###### ")

    #### 1 ####
    channel_list = np.arange(331,470,1)
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
    thehist_title = "hist_peaktime_CNUM" #FIXME: Make a configurable somehow
    #loop through each channel, show the histogram, and ask whether to use 
    #Default fitting params
    results = {"channel":[], "LED":[], "mu":[], "mu_unc":[],"calctype": 'None'}
    missing = {"channel":[], "LED":[]}
    for channel_num in HMWBLUXETEL_list:
        for j in range(6):
            thehist = thehist_title.replace("CNUM",str(channel_num))
            print("FINDING PEAKS FOR CHANNEL %i, LED %i"%(channel_num,j))
            if not myfile.GetListOfKeys().Contains(thehist):
                print("HISTOGRAM %s NOT FOUND.  SKIPPING"%(thehist))
                continue
            results["calctype"] = "WeightedMean"
            #Calculate weighted mean and stdev
            wmin = HMWBLUXETELLB[j][1]
            wmax = HMWBLUXETELUB[j][1]
            bin_centers,evts,evts_unc = hb.GetBins(myfile,thehist)
            window_inds = np.where((bin_centers<wmax) & (bin_centers>wmin))[0]
            LED_bins = bin_centers[window_inds]
            LED_evts = evts[window_inds]
            WMean = m.WeightedMean(LED_evts,LED_bins)
            WStd = m.WeightedStd(LED_evts,LED_bins)
            WSEM = np.sqrt(m.WeightedSEM(LED_evts,LED_bins))
            if math.isnan(WMean):
                print("NO DATA FOR PEAKS.  TUBE WAS LIKELY OFF")
                missing["channel"].append(channel_num)
                missing["LED"].append(LEDMAP[j])
                continue
            print("WEIGHTED MEAN: %f"%(WMean))
            print("WEIGHTED STD: %f"%(WStd))
            print("WEIGHTED SEM: %f"%(WSEM))
            results["channel"].append(channel_num)
            results["LED"].append(LEDMAP[j])
            results["mu"].append(WMean)
            results["mu_unc"].append(WSEM)
    
    for channel_num in WM_list:
        for k in range(6):
            thehist = thehist_title.replace("CNUM",str(channel_num))
            print("FITTING FOUR GAUSSIANS (C,m,s) TO PEAKTIME DISTRIBUTION")
            if not myfile.GetListOfKeys().Contains(thehist):
                print("HISTOGRAM %s NOT FOUND.  SKIPPING"%(thehist))
                continue
            #Calculate weighted mean and stdev
            wmin = WMLB[k][1]
            wmax = WMUB[k][1]
            bin_centers,evts,evts_unc = hb.GetBins(myfile,thehist)
            window_inds = np.where((bin_centers<wmax) & (bin_centers>wmin))[0]
            LED_bins = bin_centers[window_inds]
            LED_evts = evts[window_inds]
            WMean = m.WeightedMean(LED_evts,LED_bins)
            WStd = m.WeightedStd(LED_evts,LED_bins)
            WSEM = np.sqrt(m.WeightedSEM(LED_evts,LED_bins))
            if math.isnan(WMean):
                print("NO DATA FOR PEAKS.  TUBE WAS LIKELY OFF")
                missing["channel"].append(channel_num)
                missing["LED"].append(LEDMAP[j])
                continue
            print("WEIGHTED MEAN: %f"%(WMean))
            print("WEIGHTED STD: %f"%(WStd))
            print("WEIGHTED SEM: %f"%(WSEM))
            results["channel"].append(channel_num)
            results["LED"].append(LEDMAP[k])
            results["mu"].append(WMean)
            results["mu_unc"].append(WSEM)
    return results,missing

import numpy as np
import json
import ROOT
import sys
import matplotlib.pyplot as plt
import math

import lib.ArgParser as ap
import lib.Functions as fu
import lib.GainFinder as gf

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

def WeightedMean(w,x):
    return np.sum((w*x)/np.sum(w))

def WeightedStd(w,x):
    nonzerow = np.where(w > 0)[0]
    M = np.sum(w[nonzerow])
    WMean = WeightedMean(w,x)
    return np.sqrt(np.sum(w*((x-WMean)**2))/(((M-1)/M)*np.sum(w)))

def WeightedSEM(w,x):
    '''
    Approximation from Cochran, 1977
    '''
    WMean = WeightedMean(w,x)
    PMean = np.average(w)
    P_i = w/np.sum(w)
    nonzerow = np.where(w > 0)[0]
    M = np.sum(w[nonzerow])
    return(M/(M-1))*(1/(np.sum(P_i)**2))*(np.sum((P_i*x - PMean*WMean)**2) - 
            2*WMean*np.sum((P_i - PMean)*(P_i*x - PMean*WMean)) + 
            (WMean**2) * np.sum((P_i - PMean)**2))



USEFIT = False

if __name__=='__main__':
    print(" ###### WELCOME TO ANNIE LED PEAK FINDER ###### ")
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
    #HMWBLUXETELInitialParams = [np.array([1000,450,6]),np.array([1000,578,6]),
    #                 np.array([1000,750,6]),np.array([1000,867,5])]
    #HMWBLUXETELLB = [np.array([0,410,0]),np.array([0,550,0]),
    #                 np.array([0,720,0]),np.array([0,840,0])]
    #HMWBLUXETELUB = [np.array([1E4,490,15]),np.array([1E5,600,15]),
    #                 np.array([1E4,780,15]),np.array([1E5,900,15])]


    #WMInitialParams = [np.array([1000,780,6]),np.array([1000,900,6]),
    #                  np.array([1000,1070,6]),np.array([1000,1200,5])]
    #WMLB = [np.array([0,750,0]),np.array([0,870,0]),
    #                  np.array([0,1040,0]),np.array([0,1170,0])]
    #WMUB = [np.array([1E4,810,10]),np.array([1E4,930,10]),
    #                  np.array([1E4,1100,10]),np.array([1E4,1230,10])]
    if ap.APPEND is not None:
        print(" ####### USING THIS FILE TO FIT LED PEAK TIMES ####### ")
    with open(ap.APPEND,"r") as f:
        myfile = ROOT.TFile.Open(ap.APPEND)

    GainFinder = gf.GainFinder(myfile)
    GainFinder.setFitFunction(fu.landau)
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
            if USEFIT:
                results["calctype"] = "Fit"
                GainFinder.setInitialFitParams(HMWBLUXETELInitialParams[j])
                GainFinder.setBounds(HMWBLUXETELLB[j],HMWBLUXETELUB[j])
                popt,pcov,xdata,ydata,y_unc = GainFinder.FitPEPeaks(thehist, 
                        exclude_ped = False, subtract_ped = False,
                        fit_range = [HMWBLUXETELLB[j][1],HMWBLUXETELUB[j][1]])
                if type(popt) != np.ndarray:
                    print("CHANNEL %i PEAK %i FAILED TO FIT"%(channel_num,j))
                    missing["channel"].append(channel_num)
                    missing["LED"].append(LEDMAP[j])
                    continue
                pcovd = np.diag(pcov)
                #pl.PlotHistAndFit(xdata,ydata,fu.landau,xdata,popt,"Landau")
                print("BEST FIT PARAMS: " + str(popt))
                if np.sqrt(pcovd[1]) > 15:
                    print("CHANNEL %i PEAK %i HAS CRITICAL UNC. ON MEAN"%(channel_num,j))
                    missing["channel"].append(channel_num)
                    missing["LED"].append(LEDMAP[j])
                    continue
                print("%i,%f,%f,%f,%f"%(channel_num,popt[1],popt[2],np.sqrt(pcovd[1]),
                    np.sqrt(pcovd[2])))
                if (np.sqrt(pcovd[1]) < 15):
                    results["channel"].append(channel_num)
                    results["LED"].append(LEDMAP[j])
                    results["mu"].append(popt[1])
                    results["mu_unc"].append(np.sqrt(pcovd[1]))
            else:
                results["calctype"] = "WeightedMean"
                #Calculate weighted mean and stdev
                wmin = HMWBLUXETELLB[j][1]
                wmax = HMWBLUXETELUB[j][1]
                bin_centers,evts,evts_unc = GetBins(myfile,thehist)
                window_inds = np.where((bin_centers<wmax) & (bin_centers>wmin))[0]
                LED_bins = bin_centers[window_inds]
                LED_evts = evts[window_inds]
                WMean = WeightedMean(LED_evts,LED_bins)
                WSEM = np.sqrt(WeightedSEM(LED_evts,LED_bins))
                if math.isnan(WMean):
                    print("NO DATA FOR PEAKS.  TUBE WAS LIKELY OFF")
                    missing["channel"].append(channel_num)
                    missing["LED"].append(LEDMAP[j])
                    continue
                print("WEIGHTED MEAN: %f"%(WeightedMean(LED_evts,LED_bins)))
                print("WEIGHTED STD: %f"%(WeightedStd(LED_evts,LED_bins)))
                print("WEIGHTED SEM: %f"%(np.sqrt(WeightedSEM(LED_evts,LED_bins))))
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
            GainFinder.setInitialFitParams(WMInitialParams[j])
            GainFinder.setBounds(WMLB[j],WMUB[j])
            if USEFIT:
                popt,pcov,xdata,ydata,y_unc = GainFinder.FitPEPeaks(thehist, 
                        exclude_ped = False, subtract_ped = False,
                        fit_range = [WMLB[j][1],WMUB[j][1]])
                if type(popt) != np.ndarray:
                    print("CHANNEL %i PEAK %i FAILED TO FIT"%(channel_num,k))
                    missing["channel"].append(channel_num)
                    missing["LED"].append(LEDMAP[j])
                    continue
                pcovd = np.diag(pcov)
                print("BEST FIT PARAMS: " + str(popt))
                if np.sqrt(pcovd[1]) > 15:
                    print("CHANNEL %i PEAK %i HAS CRITICAL UNC. ON MEAN"%(channel_num,j))
                    missing["channel"].append(channel_num)
                    missing["LED"].append(LEDMAP[j])
                    continue
                print("%i,%f,%f,%f,%f"%(channel_num,popt[1],popt[2],np.sqrt(pcovd[1]),
                    np.sqrt(pcovd[2])))
                if (np.sqrt(pcovd[1]) < 15):
                    results["channel"].append(channel_num)
                    results["LED"].append(LEDMAP[j])
                    results["mu"].append(popt[1])
                    results["mu_unc"].append(np.sqrt(pcovd[1]))
                #pl.PlotHistAndFit(xdata,ydata,GainFinder.fitfunc,xdata,popt)
            else:
                #Calculate weighted mean and stdev
                wmin = WMLB[k][1]
                wmax = WMUB[k][1]
                bin_centers,evts,evts_unc = GetBins(myfile,thehist)
                window_inds = np.where((bin_centers<wmax) & (bin_centers>wmin))[0]
                LED_bins = bin_centers[window_inds]
                LED_evts = evts[window_inds]
                WMean = WeightedMean(LED_evts,LED_bins)
                WSEM = np.sqrt(WeightedSEM(LED_evts,LED_bins))
                if math.isnan(WMean):
                    print("NO DATA FOR PEAKS.  TUBE WAS LIKELY OFF")
                    missing["channel"].append(channel_num)
                    missing["LED"].append(LEDMAP[j])
                    continue
                print("WEIGHTED MEAN: %f"%(WeightedMean(LED_evts,LED_bins)))
                print("WEIGHTED STD: %f"%(WeightedStd(LED_evts,LED_bins)))
                print("WEIGHTED SEM: %f"%(np.sqrt(WeightedSEM(LED_evts,LED_bins))))
                results["channel"].append(channel_num)
                results["LED"].append(LEDMAP[k])
                results["mu"].append(WMean)
                results["mu_unc"].append(WSEM)
    with open("PeakFitResults.json","w") as f:
        json.dump(results,f,indent=4)
    with open("MissingPeaks.json","w") as f:
        json.dump(missing,f,indent=4)


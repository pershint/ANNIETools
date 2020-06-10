#This script has functions needed to make a Cumulative Distribution Plot from
#Different variables output in  PhaseIITreeMaker root file.

import glob

import sys
import uproot
import lib.ROOTProcessor as rp
import lib.EventSelection as es
import lib.HitsPlotter as hp
import lib.AmBePlots as abp
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as scp
import numpy as np

SIGNAL_DIR = "./Data/UncorrBkgData/"

expoPFlat= lambda x,C1,tau,mu,B: C1*np.exp(-(x-mu)/tau) + B

def EstimateDarkRates(Sdf,Sdf_trig):

    #Sdf_NoClusterWindow = es.NoBurstClusters(Sdf_CleanPrompt,0,5)
    Sdf_trig_NoClusterWindow = es.NoBurst_WholeFile(Sdf,Sdf_trig,0,5)
    print("SIZE OF DATAFRAME BEFORE BURST CUT: " + str(len(Sdf_trig)))
    print("SIZE OF DATAFRAME AFTER BURST CUT: " + str(len(Sdf_trig_NoClusterWindow)))

    All_PE = np.hstack(Sdf_trig_NoClusterWindow['hitPE'])
    All_T = np.hstack(Sdf_trig_NoClusterWindow['hitT'])
    All_ID = np.hstack(Sdf_trig_NoClusterWindow['hitDetID'])

    plt.hist(Sdf["clusterPE"],bins=200,range=(0,50))
    plt.title("cluster PE distribution \n (All acquisitions in tank)")
    plt.xlabel("Hit PE")
    plt.show()

    plt.hist(All_PE,bins=200,range=(0,50))
    plt.title("PE distribution for all PMT hits \n (AmBe background run 1718, housing in dark box)")
    plt.ylabel("Number of hits")
    plt.xlabel("Hit PE")
    plt.show()

    plt.hist(All_T,bins=350,range=(0,70000))
    plt.title("Hit time distribution for all PMT hits \n (AmBe background run 1718, housing in dark box)")
    plt.ylabel("Number of hits")
    plt.xlabel("Hit time (ns)")
    plt.show()

    #Still some prompt activity in window; let's only use the last 50 microseconds
    latewin = np.where((All_T>20000)&(All_T<70000))[0]
    late_PE = All_PE[latewin]
    late_T = All_T[latewin]
    late_ID = All_ID[latewin]

    IDSet = set(late_ID)

    Acquisition_Time = (50E-6 * len(Sdf_trig_NoClusterWindow))
    print("ACQUISITION TIME IN LATE WINDOW ESTIMATE IS: " + str(Acquisition_Time))

    hit_counts = []
    dark_rate = []
    print("ALL HIT IDS SEEN: " + str(IDSet))
    for theid in IDSet:
        IDhits = np.where(late_ID==theid)[0]
        IDnumhits = len(late_ID[IDhits])
        hit_counts.append(IDnumhits)
        dark_rate.append(IDnumhits/Acquisition_Time)
    hit_counts = np.array(hit_counts)
    dark_rates = np.array(dark_rate)

    plt.hist(hit_counts,bins=30,histtype='stepfilled',range=(0,10000),linewidth=6)
    #plt.hist(hit_counts,bins=30,alpha=0.75,histtype='step',range=(0,10000),linewidth=6)

    plt.title("Total number of hits for each PMT \n (All PMT hits with $t_{hit}>20 \, \mu s$)")
    plt.xlabel("Number of hits")
    plt.show()

    plt.hist(dark_rate,bins=30, histtype='stepfilled',range=(0,30000))
    #plt.hist(dark_rate,bins=30,alpha=0.75,histtype='step',range=(0,30000))
    plt.title("Total number of hits for each PMT \n (All PMT hits with $t_{hit}>20 \, \mu s$)")
    plt.xlabel("Dark count rate (Hz)")
    plt.ylabel("Number of PMTs")
    plt.show()

    wm_hit_counts = []
    wm_dark_rate = []
    WM_IDs = [382,393,404]
    print("ALL HIT IDS SEEN: " + str(WM_IDs))
    for theid in WM_IDs:
        IDhits = np.where(late_ID==theid)[0]
        IDnumhits = len(late_ID[IDhits])
        wm_hit_counts.append(IDnumhits)
        wm_dark_rate.append(IDnumhits/Acquisition_Time)
    
    plt.hist(wm_hit_counts,bins=30,histtype='stepfilled',range=(0,10000),linewidth=6)
    #plt.hist(hit_counts,bins=30,alpha=0.75,histtype='step',range=(0,10000),linewidth=6)

    plt.title("Total number of hits for WATCHMAN tubes \n (All PMT hits with $t_{hit}>20 \, \mu s$)")
    plt.xlabel("Number of hits")
    plt.show()

    plt.hist(wm_dark_rate,bins=30, histtype='stepfilled',range=(0,30000))
    #plt.hist(dark_rate,bins=30,alpha=0.75,histtype='step',range=(0,30000))
    plt.title("Total number of hits for WATCHMAN tubes \n (All PMT hits with $t_{hit}>20 \, \mu s$)")
    plt.xlabel("Dark count rate (Hz)")
    plt.ylabel("Number of PMTs")
    plt.show()

if __name__=='__main__':
    slist = glob.glob(SIGNAL_DIR+"*.ntuple.root")

    livetime_estimate = es.EstimateLivetime(slist)
    print("SIGNAL LIVETIME ESTIMATE IN SECONDS IS: " + str(livetime_estimate))

    mybranches = ['eventNumber','eventTimeTank','clusterTime','hitT','hitQ','hitPE','hitDetID','clusterPE']
    SProcessor = rp.ROOTProcessor(treename="phaseIITankClusterTree")
    for f1 in slist:
        SProcessor.addROOTFile(f1,branches_to_get=mybranches)
    Sdata = SProcessor.getProcessedData()
    Sdf = pd.DataFrame(Sdata)

    SProcessor = rp.ROOTProcessor(treename="phaseIITriggerTree")
    for f1 in slist:
        SProcessor.addROOTFile(f1,branches_to_get=mybranches)
    Sdata = SProcessor.getProcessedData()
    Sdf_trig = pd.DataFrame(Sdata)

    EstimateDarkRates(Sdf,Sdf_trig)



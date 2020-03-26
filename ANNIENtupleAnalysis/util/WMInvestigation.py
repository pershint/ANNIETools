#This script has functions needed to make a Cumulative Distribution Plot from
#Different variables output in  PhaseIITreeMaker root file.

import glob

import sys
import uproot
import lib.ROOTProcessor as rp
import lib.EventSelection as es
import lib.AmBePlots as abp
import lib.FileReader as fr
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as scp
import numpy as np

MCDELTAT = "./Data/MCProfiles/DeltaTHist.csv"
MCPE = "./Data/MCProfiles/PEHist.csv"
SIGNAL_DIR = "./Data/V3/CentralData/"
#BKG_DIR = "./Data/BkgCentralData/"
BKG_DIR = "./Data/V3/BkgCentralData/"

expoPFlat= lambda x,C1,tau,mu,B: C1*np.exp(-(x-mu)/tau) + B

def WMPlots(Sdf,Bdf,Sdf_trig,Bdf_trig):

    Sdf_SinglePulses = es.SingleSiPMPulses(Sdf)
    Bdf_SinglePulses = es.SingleSiPMPulses(Bdf)


    All_PE = np.hstack(Sdf_SinglePulses['hitPE'])
    All_T = np.hstack(Sdf_SinglePulses['hitT'])
    All_ID = np.hstack(Sdf_SinglePulses['hitDetID'])
    WM = np.where((All_ID == 382) | (All_ID == 393) | (All_ID == 404))[0]
    LateTime = np.where(All_T>12000)[0]
    WMDelayed = np.intersect1d(LateTime,WM)
    WM_PE = All_PE[WMDelayed]
    plt.hist(WM_PE,bins=30,range=(0,100))
    plt.title("PE distribution for three WATCHMAN tubes in hit clusters")
    plt.xlabel("Hit PE")
    plt.show()


    Sdf_Odd = Sdf_SinglePulses.loc[(Sdf_SinglePulses["clusterChargePointY"]>0.125)&(Sdf_SinglePulses["clusterChargePointY"]<0.2)].reset_index(drop=True)

    All_PE = np.hstack(Sdf_Odd['hitPE'])
    All_T = np.hstack(Sdf_Odd['hitT'])
    All_ID = np.hstack(Sdf_Odd['hitDetID'])
    WM = np.where((All_ID == 382) | (All_ID == 393) | (All_ID == 404))[0]
    LateTime = np.where(All_T>12000)[0]
    WMDelayed = np.intersect1d(LateTime,WM)
    WM_PE = All_PE[WMDelayed]
    plt.hist(WM_PE,bins=30,range=(0,100))
    plt.title("PE distribution for three WATCHMAN tubes in hit clusters \n (0.125 < PEPointY < 0.2 clusters only")
    plt.xlabel("Hit PE")
    plt.show()


    Sdf_CleanPrompt = es.NoPromptClusters(Sdf_SinglePulses,2000)
    Bdf_CleanPrompt = es.NoPromptClusters(Bdf_SinglePulses,2000)




    labels = {'title': 'Comparison of total PE to charge balance parameter (Source)', 
            'xlabel': 'Total PE', 'ylabel': 'Charge balance parameter'}
    ranges = {'xbins': 100, 'ybins':100, 'xrange':[0,150],'yrange':[0,1]}
    #abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    abp.Make2DHist(Sdf_CleanPrompt,'clusterPE','clusterChargeBalance',labels,ranges)
    abp.ShowPlot()

    labels = {'title': 'Comparison of total PE to charge balance parameter (No source, >12 $\mu$s)', 
            'xlabel': 'Total PE', 'ylabel': 'Charge balance parameter'}
    ranges = {'xbins': 50, 'ybins':50, 'xrange':[0,150],'yrange':[0,1]}
    #abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    Bdf_latewindow = Bdf.loc[Bdf['clusterTime']>12000]
    abp.Make2DHist(Bdf_latewindow,'clusterPE','clusterChargeBalance',labels,ranges)
    abp.ShowPlot()

    labels = {'title': 'Comparison of total PE to charge balance parameter', 
            'xlabel': 'Total PE', 'ylabel': 'Charge balance parameter','llabel':'No source',
            'color':'Reds'}
    ranges = {'xbins': 50, 'ybins':50, 'xrange':[0,150],'yrange':[0,1]}
    #abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)

    Bdf_window = Bdf_latewindow.loc[(Bdf_latewindow['clusterPE']<150) & \
            (Bdf_latewindow['clusterChargeBalance']>0) & (Bdf_latewindow['clusterChargeBalance']<1)]
    Sdf_window = Sdf_CleanPrompt.loc[(Sdf_CleanPrompt['clusterPE']<150) & \
            (Sdf_CleanPrompt['clusterChargeBalance']>0) & (Sdf_CleanPrompt['clusterChargeBalance']<1)]
    abp.MakeKDEPlot(Bdf_window,'clusterPE','clusterChargeBalance',labels,ranges)
    labels = {'title': 'Comparison of total PE to charge balance parameter', 
            'xlabel': 'Total PE', 'ylabel': 'Charge balance parameter','llabel':'No source',
            'color':'Blues'}
    abp.MakeKDEPlot(Sdf_window,'clusterPE','clusterChargeBalance',labels,ranges)
    abp.ShowPlot()


    #Apply late window and charge balance cuts for data cleaning
    Bdf_latewindow = Bdf.loc[Bdf['clusterTime']>12000]
    Sdf_CleanPromptCB = Sdf_CleanPrompt.loc[Sdf_CleanPrompt['clusterChargeBalance']<0.4].reset_index(drop=True)

    
    labels = {'title': 'Comparison of total PE to Charge Point Y-component (Source) \n (No CB cut)', 
            'xlabel': 'Total PE', 'ylabel': 'Charge Point Y'}
    ranges = {'xbins': 30, 'ybins':30, 'xrange':[0,60],'yrange':[-1,1]}
    #abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    abp.Make2DHist(Sdf_CleanPrompt,'clusterPE','clusterChargePointY',labels,ranges)
    abp.ShowPlot()

    labels = {'title': 'Comparison of total PE to Charge Point Y-component (Source) \n (Charge balance < 0.4)', 
            'xlabel': 'Total PE', 'ylabel': 'Charge Point Y'}
    ranges = {'xbins': 30, 'ybins':30, 'xrange':[0,60],'yrange':[-1,1]}
    #abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    abp.Make2DHist(Sdf_CleanPromptCB,'clusterPE','clusterChargePointY',labels,ranges)
    abp.ShowPlot()


    labels = {'title': 'Comparison of total PE to Charge Point Y-component (No source, >12 $\mu$s)', 
            'xlabel': 'Total PE', 'ylabel': 'Charge Point Y'}
    ranges = {'xbins': 30, 'ybins':30, 'xrange':[0,60],'yrange':[-1,1]}
    #abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    abp.Make2DHist(Bdf_latewindow,'clusterPE','clusterChargePointY',labels,ranges)
    abp.ShowPlot()

    Bdf_latewindow_hiCBCut = Bdf_latewindow.loc[Bdf_latewindow['clusterChargeBalance']>0.4]
    labels = {'title': 'Comparison of total PE to Charge Point Y-component (No source, >12 $\mu$s, Charge Balance > 0.4)', 
            'xlabel': 'Total PE', 'ylabel': 'Charge Point Y'}
    ranges = {'xbins': 30, 'ybins':30, 'xrange':[0,60],'yrange':[-1,1]}
    #abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    abp.Make2DHist(Bdf_latewindow_hiCBCut,'clusterPE','clusterChargePointY',labels,ranges)
    abp.ShowPlot()

    Bdf_latewindow_CBCut = Bdf_latewindow.loc[Bdf_latewindow['clusterChargeBalance']<0.4]
    labels = {'title': 'Comparison of total PE to Charge Point Y-component (No source, >12 $\mu$s, Charge Balance < 0.4)', 
            'xlabel': 'Total PE', 'ylabel': 'Charge Point Y'}
    ranges = {'xbins': 30, 'ybins':30, 'xrange':[0,60],'yrange':[-1,1]}
    #abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    abp.Make2DHist(Bdf_latewindow_CBCut,'clusterPE','clusterChargePointY',labels,ranges)
    abp.ShowPlot()



if __name__=='__main__':
    slist = glob.glob(SIGNAL_DIR+"*.ntuple.root")
    blist = glob.glob(BKG_DIR+"*.ntuple.root")

    livetime_estimate = es.EstimateLivetime(slist)
    print("SIGNAL LIVETIME ESTIMATE IN SECONDS IS: " + str(livetime_estimate))
    livetime_estimate = es.EstimateLivetime(blist)
    print("BKG LIVETIME ESTIMATE IN SECONDS IS: " + str(livetime_estimate))

    mybranches = ['eventNumber','eventTimeTank','clusterTime','SiPMhitQ','SiPMNum','SiPMhitT','hitT','hitQ','hitPE','hitDetID','SiPMhitAmplitude','clusterChargeBalance','clusterPE','clusterMaxPE','SiPM1NPulses','SiPM2NPulses','clusterChargePointY']
    SProcessor = rp.ROOTProcessor(treename="phaseIITankClusterTree")
    for f1 in slist:
        SProcessor.addROOTFile(f1,branches_to_get=mybranches)
    Sdata = SProcessor.getProcessedData()
    Sdf = pd.DataFrame(Sdata)

    BProcessor = rp.ROOTProcessor(treename="phaseIITankClusterTree")
    for f1 in blist:
        BProcessor.addROOTFile(f1,branches_to_get=mybranches)
    Bdata = BProcessor.getProcessedData()
    Bdf = pd.DataFrame(Bdata)

    SProcessor = rp.ROOTProcessor(treename="phaseIITriggerTree")
    for f1 in slist:
        SProcessor.addROOTFile(f1,branches_to_get=mybranches)
    Sdata = SProcessor.getProcessedData()
    Sdf_trig = pd.DataFrame(Sdata)

    BProcessor = rp.ROOTProcessor(treename="phaseIITriggerTree")
    for f1 in blist:
        BProcessor.addROOTFile(f1,branches_to_get=mybranches)
    Bdata = BProcessor.getProcessedData()
    Bdf_trig = pd.DataFrame(Bdata)
    WMPlots(Sdf,Bdf,Sdf_trig,Bdf_trig)



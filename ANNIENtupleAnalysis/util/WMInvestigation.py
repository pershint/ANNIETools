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

SIGNAL_DIR = "./Data/V3_5PE100ns/Pos0Data/"
#BKG_DIR = "./Data/BkgCentralData/"
BKG_DIR = "./Data/V3_5PE100ns/BkgPos0Data/"

expoPFlat= lambda x,C1,tau,mu,B: C1*np.exp(-(x-mu)/tau) + B

def WMPlots(Sdf,Bdf,Sdf_trig,Bdf_trig):

    Sdf_SinglePulses = es.SingleSiPMPulses(Sdf)
    Sdf_CleanPrompt = es.NoPromptClusters(Sdf_SinglePulses,2000)
    Sdf_CleanWindow = es.NoBurstClusters(Sdf_CleanPrompt,2000,150)
    Sdf_LateWindow = Sdf_CleanWindow.loc[(Sdf_CleanWindow["clusterTime"]>2000)].reset_index(drop=True)
    Sdf_trig_goodSiPM = Sdf_trig.loc[(Sdf_trig['SiPM1NPulses']==1) & (Sdf_trig['SiPM2NPulses']==1)].reset_index(drop=True)
    Sdf_trig_CleanPrompt = es.NoPromptClusters_WholeFile(Sdf_SinglePulses,Sdf_trig_goodSiPM,2000)
    Sdf_trig_CleanWindow = es.NoBurst_WholeFile(Sdf_CleanPrompt,Sdf_trig_CleanPrompt,2000,150)

    Bdf_SinglePulses = es.SingleSiPMPulses(Bdf)
    Bdf_CleanPrompt = es.NoPromptClusters(Bdf_SinglePulses,2000)
    Bdf_CleanWindow = es.NoBurstClusters(Bdf_CleanPrompt,2000,150)
    Bdf_LateWindow = Bdf_CleanWindow.loc[(Bdf_CleanWindow['clusterTime']>2000)].reset_index(drop=True)
    Bdf_trig_cleanSiPM = Bdf_trig.loc[(Bdf_trig['SiPM1NPulses']==1) & (Bdf_trig['SiPM2NPulses']==1)].reset_index(drop=True)
    Bdf_trig_cleanPrompt = es.NoPromptClusters_WholeFile(Bdf_SinglePulses,Bdf_trig_cleanSiPM,2000)
    Bdf_trig_BurstCut = es.NoBurst_WholeFile(Bdf_CleanPrompt,Bdf_trig_cleanPrompt,2000,150)


    All_PE = np.hstack(Sdf_LateWindow['hitPE'])
    All_T = np.hstack(Sdf_LateWindow['hitT'])
    All_ID = np.hstack(Sdf_LateWindow['hitDetID'])


    WM = np.where((All_ID == 382) | (All_ID == 393) | (All_ID == 404))[0]
    LateTime = np.where(All_T>12000)[0]
    WMDelayed = np.intersect1d(LateTime,WM)
    WM_PE = All_PE[WMDelayed]
    plt.hist(WM_PE,bins=30,range=(0,100))
    plt.title("PE distribution for three WATCHMAN tubes in hit clusters \n (AmBe source data, all preliminary cuts)")
    plt.xlabel("Hit PE")
    plt.show()

    #Sdf_Odd = Sdf.loc[(Sdf["clusterChargePointY"]>0.125)&(Sdf["clusterChargePointY"]<0.2)].reset_index(drop=True)
    Sdf_Odd = Sdf.loc[(Sdf["clusterChargeBalance"]>0.9)].reset_index(drop=True)
    HiCB_PE = np.hstack(Sdf_Odd.hitPE)
    HiCB_DetID = np.hstack(Sdf_Odd.hitDetID)
    plt.hist2d(HiCB_DetID,HiCB_PE, bins=(138,20),
            range=[(331,469),(0,20)],
            cmap = plt.cm.inferno)
    plt.colorbar()
    plt.title("PE distribution for all hits in clusters \n (Central source, $t_{c}>2 \, \mu s$, CB>0.9)")
    plt.xlabel("Tube ID")
    plt.ylabel("PE")
    plt.show()

    #Sdf_Odd = Sdf.loc[(Sdf["clusterChargePointY"]>0.125)&(Sdf["clusterChargePointY"]<0.2)].reset_index(drop=True)
    Sdf_Mid = Sdf.loc[(Sdf["clusterChargeBalance"]>0.4)&(Sdf["clusterChargeBalance"]<0.6)].reset_index(drop=True)
    MidCB_PE = np.hstack(Sdf_Mid.hitPE)
    MidCB_DetID = np.hstack(Sdf_Mid.hitDetID)
    plt.hist2d(MidCB_DetID,MidCB_PE, bins=(138,20),
            range=[(331,469),(0,20)],
            cmap = plt.cm.inferno)
    plt.colorbar()
    plt.title("PE distribution for all hits in clusters \n (Central source, $t_{c}>2 \, \mu s$, 0.4<CB<0.6)")
    plt.xlabel("Tube ID")
    plt.ylabel("PE")
    plt.show()

    Sdf_Lo = Sdf.loc[(Sdf["clusterChargeBalance"]<0.4)].reset_index(drop=True)
    LoCB_PE = np.hstack(Sdf_Lo.hitPE)
    LoCB_DetID = np.hstack(Sdf_Lo.hitDetID)
    plt.hist2d(LoCB_DetID,LoCB_PE, bins=(138,20),
            range=[(331,469),(0,20)],
            cmap = plt.cm.inferno)
    plt.colorbar()
    plt.title("PE distribution for all hits in clusters \n (Central source, $t_{c}>2 \, \mu s$, CB<0.4)")
    plt.xlabel("Tube ID")
    plt.ylabel("PE")
    plt.show()

    All_PE = np.hstack(Sdf_Odd['hitPE'])
    All_T = np.hstack(Sdf_Odd['hitT'])
    All_ID = np.hstack(Sdf_Odd['hitDetID'])
    WM = np.where((All_ID == 382) | (All_ID == 393) | (All_ID == 404))[0]
    LateTime = np.where(All_T>12000)[0]
    WMDelayed = np.intersect1d(LateTime,WM)
    WM_PE = All_PE[WMDelayed]
    plt.hist(WM_PE,bins=30,range=(0,100))
    plt.title("PE distribution for three WATCHMAN tubes in hit clusters \n (Source, 0.9 < CB < 1.0 clusters only")
    plt.xlabel("Hit PE")
    plt.show()



    Sdf_rawLate = Sdf.loc[(Sdf["clusterTime"]>2000)].reset_index(drop=True)
    labels = {'title': 'Comparison of total PE to charge balance parameter \n (Source, $t_{c}>2 \, \mu s$)', 
            'xlabel': 'Total PE', 'ylabel': 'Charge balance parameter'}
    ranges = {'xbins': 40, 'ybins':40, 'xrange':[0,80],'yrange':[0,1]}
    #abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    abp.Make2DHist(Sdf_rawLate,'clusterPE','clusterChargeBalance',labels,ranges)
    plt.show()

    labels = {'title': 'Charge balance parameters in time window \n (Source, $t_{c}>2 \, \mu s$)', 
            'xlabel': 'Cluster time (ns)', 'ylabel': 'Charge balance'}
    ranges = {'xbins': 40, 'ybins':40, 'xrange':[2000,67000],'yrange':[0,1]}
    #abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    abp.Make2DHist(Sdf_rawLate,'clusterTime','clusterChargeBalance',labels,ranges)
    plt.show()

    labels = {'title': 'Comparison of total PE to charge balance parameter \n (Central source w/ preliminary cuts, $t_{c}>2 \, \mu s$)', 
            'xlabel': 'Total PE', 'ylabel': 'Charge balance parameter'}
    ranges = {'xbins': 40, 'ybins':40, 'xrange':[0,80],'yrange':[0,1]}
    #abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    abp.Make2DHist(Sdf_LateWindow,'clusterPE','clusterChargeBalance',labels,ranges)
    plt.show()

    labels = {'title': 'Comparison of total PE to charge balance parameter \n (Central background, >2 $\mu$s)', 
            'xlabel': 'Total PE', 'ylabel': 'Charge balance parameter'}
    ranges = {'xbins': 50, 'ybins':50, 'xrange':[0,80],'yrange':[0,1]}
    #abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    Bdf_latewindow = Bdf.loc[Bdf['clusterTime']>12000]
    abp.Make2DHist(Bdf_latewindow,'clusterPE','clusterChargeBalance',labels,ranges)
    plt.show()

    labels = {'title': 'Comparison of total PE to charge balance parameter \n (Central background w\ preliminary cuts, >2 $\mu$s)', 
            'xlabel': 'Total PE', 'ylabel': 'Charge balance parameter'}
    ranges = {'xbins': 50, 'ybins':50, 'xrange':[0,80],'yrange':[0,1]}
    #abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    abp.Make2DHist(Bdf_LateWindow,'clusterPE','clusterChargeBalance',labels,ranges)
    plt.show()

    #Apply late window and charge balance cuts for data cleaning
    Bdf_latewindow = Bdf.loc[Bdf['clusterTime']>12000]
    Sdf_CleanPromptCB = Sdf_CleanPrompt.loc[Sdf_CleanPrompt['clusterChargeBalance']<0.4].reset_index(drop=True)

    
    #labels = {'title': 'Comparison of total PE to Charge Point Y-component (Source) \n (No CB cut)', 
    #        'xlabel': 'Total PE', 'ylabel': 'Charge Point Y'}
    #ranges = {'xbins': 30, 'ybins':30, 'xrange':[0,60],'yrange':[-1,1]}
    ##abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    #abp.Make2DHist(Sdf_CleanPrompt,'clusterPE','clusterChargePointY',labels,ranges)
    #plt.show()

    #labels = {'title': 'Comparison of total PE to Charge Point Y-component (Source) \n (Charge balance < 0.4)', 
    #        'xlabel': 'Total PE', 'ylabel': 'Charge Point Y'}
    #ranges = {'xbins': 30, 'ybins':30, 'xrange':[0,60],'yrange':[-1,1]}
    ##abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    #abp.Make2DHist(Sdf_CleanPromptCB,'clusterPE','clusterChargePointY',labels,ranges)
    #plt.show()


    #labels = {'title': 'Comparison of total PE to Charge Point Y-component (No source, >12 $\mu$s)', 
    #        'xlabel': 'Total PE', 'ylabel': 'Charge Point Y'}
    #ranges = {'xbins': 30, 'ybins':30, 'xrange':[0,60],'yrange':[-1,1]}
    ##abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    #abp.Make2DHist(Bdf_latewindow,'clusterPE','clusterChargePointY',labels,ranges)
    #plt.show()

    #Bdf_latewindow_hiCBCut = Bdf_latewindow.loc[Bdf_latewindow['clusterChargeBalance']>0.4]
    #labels = {'title': 'Comparison of total PE to Charge Point Y-component (No source, >12 $\mu$s, Charge Balance > 0.4)', 
    #        'xlabel': 'Total PE', 'ylabel': 'Charge Point Y'}
    #ranges = {'xbins': 30, 'ybins':30, 'xrange':[0,60],'yrange':[-1,1]}
    ##abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    #abp.Make2DHist(Bdf_latewindow_hiCBCut,'clusterPE','clusterChargePointY',labels,ranges)
    #plt.show()

    #Bdf_latewindow_CBCut = Bdf_latewindow.loc[Bdf_latewindow['clusterChargeBalance']<0.4]
    #labels = {'title': 'Comparison of total PE to Charge Point Y-component (No source, >12 $\mu$s, Charge Balance < 0.4)', 
    #        'xlabel': 'Total PE', 'ylabel': 'Charge Point Y'}
    #ranges = {'xbins': 30, 'ybins':30, 'xrange':[0,60],'yrange':[-1,1]}
    ##abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    #abp.Make2DHist(Bdf_latewindow_CBCut,'clusterPE','clusterChargePointY',labels,ranges)
    #plt.show()



if __name__=='__main__':
    slist = glob.glob(SIGNAL_DIR+"*.ntuple.root")
    blist = glob.glob(BKG_DIR+"*.ntuple.root")

    livetime_estimate = es.EstimateLivetime(slist)
    print("SIGNAL LIVETIME ESTIMATE IN SECONDS IS: " + str(livetime_estimate))
    livetime_estimate = es.EstimateLivetime(blist)
    print("BKG LIVETIME ESTIMATE IN SECONDS IS: " + str(livetime_estimate))

    mybranches = ['eventNumber','eventTimeTank','clusterTime','SiPMhitQ','SiPMNum','SiPMhitT','hitT','hitQ','hitPE','hitDetID','clusterChargeBalance','clusterPE','SiPM1NPulses','SiPM2NPulses']
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



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

MCFILE = "./Data/MCProfiles/Analyzer_AmBe_Housing_Center-0-0-0_100k_Nogammas.root"
HISTBASE = "h_HitsDelayed_withEdep_min"
HISTENDS = ["250keV","500keV","1MeV","2MeV","3MeV","4MeV"]

SIGNAL_DIR = "./Data/V3_5PE100ns/CentralData/"
#BKG_DIR = "./Data/BkgCentralData/"
BKG_DIR = "./Data/V3_5PE100ns/BkgCentralData/"

def PlotDemo(Sdf,Bdf,Sdf_trig,Bdf_trig):
    Sdf_SinglePulses = es.SingleSiPMPulses(Sdf)
    Sdf_CleanPrompt = es.NoPromptClusters(Sdf_SinglePulses,2000)
    Sdf_CleanWindow = es.NoBurstClusters(Sdf_CleanPrompt,2000,150)
    Sdf_CleanWindow_CBClean = Sdf_CleanWindow.loc[Sdf_CleanWindow['clusterChargeBalance']<0.4]
    Sdf_trig_goodSiPM = Sdf_trig.loc[(Sdf_trig['SiPM1NPulses']==1) & (Sdf_trig['SiPM2NPulses']==1)].reset_index(drop=True)
    Sdf_trig_CleanPrompt = es.NoPromptClusters_WholeFile(Sdf_SinglePulses,Sdf_trig_goodSiPM,2000)
    Sdf_trig_CleanWindow = es.NoBurst_WholeFile(Sdf_CleanPrompt,Sdf_trig_CleanPrompt,2000,150)

    #Line of cuts applied to background clusters
    Bdf_SinglePulses = es.SingleSiPMPulses(Bdf)
    Bdf_CleanPrompt = es.NoPromptClusters(Bdf_SinglePulses,2000)
    Bdf_CleanWindow = es.NoBurstClusters(Bdf_CleanPrompt,2000,150)
    Bdf_latewindow = Bdf_CleanWindow.loc[(Bdf_CleanWindow['clusterTime']>2000)].reset_index(drop=True)
    Bdf_latewindow = Bdf_latewindow.loc[(Bdf_latewindow['clusterChargeBalance']<0.4)].reset_index(drop=True)
    Bdf_trig_cleanSiPM = Bdf_trig.loc[(Bdf_trig['SiPM1NPulses']==1) & (Bdf_trig['SiPM2NPulses']==1)].reset_index(drop=True)
    Bdf_trig_cleanPrompt = es.NoPromptClusters_WholeFile(Bdf_SinglePulses,Bdf_trig_cleanSiPM,2000)
    Bdf_trig_BurstCut = es.NoBurst_WholeFile(Bdf_CleanPrompt,Bdf_trig_cleanPrompt,2000,150)

    #Special case
    Bdf_NoBurstLateWindow = es.NoBurstClusters(Bdf_SinglePulses,12000,150)
    Bdf_NBlatewindow = Bdf_NoBurstLateWindow.loc[(Bdf_NoBurstLateWindow['clusterTime']>12000)].reset_index(drop=True)
    Bdf_NBCBCut = Bdf_NBlatewindow.loc[Bdf_NBlatewindow['clusterChargeBalance']<0.4].reset_index(drop=True)


    #Now, we've gotta get the total PE observed around the SiPM pulses
    df = Sdf_trig_goodSiPM
    SiPMTimeThreshold = 100
    TotalPE = []
    for j in df.index.values:  #disgusting...
        if df["SiPM1NPulses"][j]!=1 or df["SiPM2NPulses"][j]!=1:
            continue
        elif abs(df["SiPMhitT"][j][0] - df["SiPMhitT"][j][1]) > SiPMTimeThreshold:
           continue
        SiPMMeanTime = (df["SiPMhitT"][j][0] + df["SiPMhitT"][j][1])/2.
        clusterHitInds = np.where(((SiPMMeanTime - np.array(df["hitT"][j]))<0)&((SiPMMeanTime - np.array(df["hitT"][j]))>-600))[0]
        clusterHits = np.array(df["hitPE"][j])[clusterHitInds]
        clusterPE = np.sum(clusterHits)
        TotalPE.append(clusterPE)
    TotalPE = np.array(TotalPE)
    TotalPE_neut = np.where(TotalPE<200)[0]
    TotalPE = TotalPE[TotalPE_neut]
    plt.hist(TotalPE,bins=200,label="Source",alpha=0.7)
    plt.xlabel("False start candidate PE")
    plt.title("Manual cluster-finding PE neighboring SiPM pulses \n "+ r"(One pulse each SiPM, $-600<\mu_{SiPM} - t_{PMT}<0$)")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    print("AFTER MANUAL FINDING STUFF")

    #Get prompt clusters within a 100 ns window of the mean SiPM single pulse time
    Sdf_SPPrompt = es.SiPMCorrelatedPrompts(Sdf_SinglePulses,100,1000,2000)
    print("NUMBER OF PROMPT CLUSTERS PASSING SINGLE SIPM PULSE CUTS: " + str(len(Sdf_SinglePulses.eventTimeTank)))
    #Sdf_SPPrompt = Sdf_SPPrompt.loc[Sdf_SPPrompt['clusterChargeBalance']<0.4].reset_index(drop=True)

    plt.hist(TotalPE,bins=200,range=(0,200),label="Source, all hits near SiPM",alpha=0.8,histtype='step',color='blue',linewidth=6)
    plt.hist(np.hstack(Sdf_SPPrompt['clusterPE']),bins=200,range=(0,200),alpha=0.75,histtype='step',linewidth=6,color='green',label='Source, TA clusters')
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Comparison of FS candidate cluster PE \n using manual clustering and ToolAnalysis clustering")
    plt.xlabel("Cluster PE")
    plt.show()

    #plt.hist(TotalPE,bins=200,range=(0,200),label="Source, all hits near SiPM",alpha=0.8,histtype='step',color='blue',linewidth=6)
    plt.hist(TotalPE,bins=200,range=(0,200),alpha=0.8,histtype='step',color='blue',linewidth=6)
    plt.hist(TotalPE,bins=200,range=(0,200),alpha=0.5,histtype='stepfilled',color='blue',linewidth=6)
    #plt.hist(np.hstack(Sdf_SPPrompt['clusterPE']),bins=200,range=(0,200),alpha=0.75,histtype='step',linewidth=6,color='green',label='Source, TA clusters')
    #leg = plt.legend(loc=1,fontsize=24)
    #leg.set_frame_on(True)
    #leg.draw_frame(True)
    #plt.title("Comparison of FS candidate cluster PE \n using manual clustering and ToolAnalysis clustering")
    plt.title("SiPM-correlated tank PE for position 0 AmBe source data \n (Sum of all PMT hits within 300 ns of SiPM pulses)")
    plt.ylabel("Number of acquisitions")
    plt.xlabel("Total PE")
    plt.show()


    Sdf_SPPrompt_trig = es.SiPMCorrelatedPrompts_WholeFile(Sdf_SPPrompt,Sdf_trig)
    print("NUMBER OF PROMPT CLUSTERS PASSING CORRELATED PROMPT CUTS: " + str(len(Sdf_SPPrompt.eventTimeTank)))
    print("NUMBER OF TRIGS WITH AT LEAST ONE CORRELATED PROMPT: " + str(len(Sdf_SPPrompt_trig.eventTimeTank)))
    MSData = abp.MakeClusterMultiplicityPlot(Sdf_SPPrompt,Sdf_SPPrompt_trig)
    #s_bins, s_edges = np.histogram(MSData,bins=20, range=(0,20))
    #plt.hist(MSData,bins=20, range=(0,20), label="Source",alpha=0.7)
    plt.hist(MSData,bins=100,label="Source",alpha=0.7)
    plt.xlabel("False start candidate cluster multiplicity")
    plt.title("Cluster multiplicity of false starts in source data \n (One pulse each SiPM, cluster within prompt window)")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    S1Delta,S2Delta = abp.SiPMClusterDifferences(Sdf_SPPrompt,100)
    plt.hist(S1Delta,bins=100,label="S1Time-clusterTime",alpha=0.7)
    plt.hist(S2Delta,bins=100,label="S2Time-clusterTime",alpha=0.7)
    plt.xlabel("SiPM time - Cluster time (ns)")
    plt.title("Difference between SiPM peak time and cluster time \n (One pulse each SiPM, cluster within prompt window)")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    Sdf_SPPromptLoPE = Sdf_SPPrompt.loc[Sdf_SPPrompt['clusterPE']<80].reset_index(drop=True)
    print("NUMBER OF PROMPT CLUSTERS PASSING CORRELATED PROMPT CUTS LT 80 PE: " + str(len(Sdf_SPPromptLoPE.eventTimeTank)))
    NumTrigs = len(set(Sdf_SPPromptLoPE.eventNumber))
    print("NUMBER OF TRIGS WITH A PASSING CORRELATED PROMPT CUTS CLUSTER LT 80 PE: " + str(NumTrigs))
    labels = {'title': 'Comparison of cluster PE to total SiPM Charge \n (Position 0, AmBe source installed)', 
            'xlabel': 'Cluster PE', 'ylabel': 'Total SiPM charge [nC]'}
    ranges = {'xbins': 30, 'ybins':40, 'xrange':[0,60],'yrange':[0,0.5],'promptTime':2000}
    #abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    abp.Make2DHist_PEVsQ(Sdf_SPPrompt,labels,ranges)
    plt.show()

    plt.hist(np.hstack(Sdf_SPPrompt['clusterPE']),bins=30,range=(0,80),alpha=0.5,histtype='stepfilled',linewidth=6)
    plt.hist(np.hstack(Sdf_SPPrompt['clusterPE']),bins=30,range=(0,80),alpha=0.75,histtype='step',linewidth=6,color='green')
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("False start candidate PE distribution (AmBe central source data)")
    plt.xlabel("Cluster PE")
    plt.show()


    #Sdf_SPPrompt_10N = Sdf_SPPrompt.loc[Sdf_SPPrompt['clusterHits']==10].reset_index(drop=True)
    #Sdf_SPPrompt_11N = Sdf_SPPrompt.loc[Sdf_SPPrompt['clusterHits']==11].reset_index(drop=True)
    #Sdf_SPPrompt_12N = Sdf_SPPrompt.loc[Sdf_SPPrompt['clusterHits']==12].reset_index(drop=True)
    #Sdf_SPPrompt_8N = Sdf_SPPrompt.loc[Sdf_SPPrompt['clusterHits']==8].reset_index(drop=True)
    #Sdf_SPPrompt_9N = Sdf_SPPrompt.loc[Sdf_SPPrompt['clusterHits']==9].reset_index(drop=True)
    ##plt.hist(np.hstack(Sdf_SPPrompt_10N['clusterPE']),bins=30,range=(0,80),alpha=0.5,histtype='stepfilled',linewidth=6)
    #plt.hist(np.hstack(Sdf_SPPrompt_8N['clusterPE']),bins=30,range=(0,80),label='8 hits',alpha=0.75,histtype='step',linewidth=6,color='green')
    #plt.hist(np.hstack(Sdf_SPPrompt_9N['clusterPE']),bins=30,range=(0,80),label='9 hits',alpha=0.75,histtype='step',linewidth=6,color='black')
    #plt.hist(np.hstack(Sdf_SPPrompt_10N['clusterPE']),bins=30,range=(0,80),label='10 hits',alpha=0.75,histtype='step',linewidth=6,color='blue')
    ##plt.hist(np.hstack(Sdf_SPPrompt_11N['clusterPE']),bins=30,range=(0,80),alpha=0.5,histtype='stepfilled',linewidth=6)
    #plt.hist(np.hstack(Sdf_SPPrompt_11N['clusterPE']),bins=30,range=(0,80),label='11 hits',alpha=0.75,histtype='step',linewidth=6,color='red')
    ##plt.hist(np.hstack(Sdf_SPPrompt_12N['clusterPE']),bins=30,range=(0,80),alpha=0.5,histtype='stepfilled',linewidth=6)
    #plt.hist(np.hstack(Sdf_SPPrompt_12N['clusterPE']),bins=30,range=(0,80),label='12 hits',alpha=0.75,histtype='step',linewidth=6,color='purple')
    ##plt.hist(np.hstack(Sdf_SPPrompt_8N['clusterPE']),bins=30,range=(0,80),alpha=0.5,histtype='stepfilled',linewidth=6)
    ##plt.hist(np.hstack(Sdf_SPPrompt_9N['clusterPE']),bins=30,range=(0,80),alpha=0.5,histtype='stepfilled',linewidth=6)
    #leg = plt.legend(loc=1,fontsize=24)
    #leg.set_frame_on(True)
    #leg.draw_frame(True)
    #plt.title("Cluster PE distribution for clusters of varying nhit")
    #plt.xlabel("Cluster PE")
    #plt.show()

if __name__=='__main__':
    slist = glob.glob(SIGNAL_DIR+"*.ntuple.root")
    blist = glob.glob(BKG_DIR+"*.ntuple.root")

    livetime_estimate = es.EstimateLivetime(slist)
    print("SIGNAL LIVETIME ESTIMATE IN SECONDS IS: " + str(livetime_estimate))
    livetime_estimate = es.EstimateLivetime(blist)
    print("BKG LIVETIME ESTIMATE IN SECONDS IS: " + str(livetime_estimate))

    mybranches = ['eventNumber','eventTimeTank','clusterTime','SiPMhitT','SiPMhitQ','SiPMhitAmplitude','clusterChargeBalance','clusterPE','SiPM1NPulses','SiPM2NPulses','SiPMNum','clusterHits','hitT','hitPE']
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
    PlotDemo(Sdf,Bdf,Sdf_trig,Bdf_trig)



#This script has functions needed to make a Cumulative Distribution Plot from
#Different variables output in  PhaseIITreeMaker root file.

import glob

import sys
import uproot
import lib.ROOTProcessor as rp
import lib.AmBePlots as abp
import pandas as pd
import matplotlib.pyplot as plt

SIGNAL_DIR = "./Data/CentralData/"
BKG_DIR = "./Data/BkgCentralData/"

def PlotDemo(Sdf,Bdf):
    abp.MakeClusterTimeDistribution(Sdf,"Source")
    abp.MakeClusterTimeDistribution(Bdf,"No source")
    leg = plt.legend(loc=4,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    labels = {'title': "Amplitude of SiPM1 hits", 'xlabel':'Pulse amplitude [V]','llabel':'SiPM1'}
    ranges = {'bins': 100, 'range':(0,0.1)}
    abp.MakeSiPMVariableDistribution(Sdf, "SiPMhitAmplitude",1,labels,ranges,True)
    labels = {'title': "Amplitude of SiPM2 hits", 'xlabel':'Pulse amplitude [V]','llabel':'SiPM2'}
    abp.MakeSiPMVariableDistribution(Sdf, "SiPMhitAmplitude",2,labels,ranges,True)
    leg = plt.legend(loc=4,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    labels = {'title': "Pulse times in SiPM1", 'xlabel':'Pulse peak time [ns]','llabel':'Source'}
    ranges = {'bins': 500, 'range':(0,65000)}
    abp.MakeSiPMVariableDistribution(Sdf, "SiPMhitT",1,labels,ranges,False)
    leg = plt.legend(loc=4,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()


    labels = {'title': 'Comparison of total PE to charge balance parameter (Source)', 
            'xlabel': 'Total PE', 'ylabel': 'Charge balance parameter'}
    ranges = {'xbins': 100, 'ybins':100, 'xrange':[0,150],'yrange':[0,1]}
    #abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    abp.Make2DHist(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    abp.ShowPlot()

    labels = {'title': 'Comparison of total PE to charge balance parameter (No source, >20 $\mu$s)', 
            'xlabel': 'Total PE', 'ylabel': 'Charge balance parameter'}
    ranges = {'xbins': 50, 'ybins':50, 'xrange':[0,150],'yrange':[0,1]}
    #abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    Bdf_latewindow = Bdf.loc[Bdf['clusterTime']>20000]
    abp.Make2DHist(Bdf_latewindow,'clusterPE','clusterChargeBalance',labels,ranges)
    abp.ShowPlot()

    labels = {'title': 'Comparison of total PE to charge balance parameter', 
            'xlabel': 'Total PE', 'ylabel': 'Charge balance parameter','llabel':'No source',
            'color':'Reds'}
    ranges = {'xbins': 50, 'ybins':50, 'xrange':[0,150],'yrange':[0,1]}
    #abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    Bdf_window = Bdf.loc[(Bdf['clusterTime']>20000) & (Bdf['clusterPE']<150)]
    Sdf_window = Sdf.loc[(Sdf['clusterPE']<150)]
    abp.MakeKDEPlot(Bdf_window,'clusterPE','clusterChargeBalance',labels,ranges)
    labels = {'title': 'Comparison of total PE to charge balance parameter', 
            'xlabel': 'Total PE', 'ylabel': 'Charge balance parameter','llabel':'No source',
            'color':'Blues'}
    abp.MakeKDEPlot(Sdf_window,'clusterPE','clusterChargeBalance',labels,ranges)
    abp.ShowPlot()



if __name__=='__main__':
    slist = glob.glob(SIGNAL_DIR+"*.ntuple.root")
    blist = glob.glob(BKG_DIR+"*.ntuple.root")

    livetime_estimate = abp.EstimateLivetime(slist)
    print("SIGNAL LIVETIME ESTIMATE IN SECONDS IS: " + str(livetime_estimate))
    livetime_estimate = abp.EstimateLivetime(blist)
    print("BKG LIVETIME ESTIMATE IN SECONDS IS: " + str(livetime_estimate))

    mybranches = ['eventNumber','clusterTime','SiPMhitQ','SiPMNum','SiPMhitT','hitT','hitQ','SiPMhitAmplitude','clusterChargeBalance','clusterPE','SiPM1NPulses','SiPM2NPulses','clusterChargePointZ']
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
    PlotDemo(Sdf,Bdf)



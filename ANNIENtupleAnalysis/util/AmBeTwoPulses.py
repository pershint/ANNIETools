#This script has functions needed to make a Cumulative Distribution Plot from
#Different variables output in  PhaseIITreeMaker root file.

import glob

import sys
import uproot
import lib.ROOTProcessor as rp
import lib.EventSelection as es
import lib.AmBePlots as abp
import lib.FileReader as fr
import lib.HistogramUtils as hu

import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as scp
import numpy as np

MCDELTAT = "./Data/MCProfiles/DeltaTHistAmBe.csv"
MCPE = "./Data/MCProfiles/PEHistAmBe.csv"

SIGNAL_DIR = "./Data/V3_5PE100ns/Pos0Data/"



def TwoPulsePlot(Sdf_trig):
    Sdf_DoublePulses = Sdf_trig.loc[(Sdf_trig["SiPM1NPulses"]==2) | (Sdf_trig["SiPM2NPulses"]==2)].reset_index(drop=True)
    print("NUM DOUBLE PULSES IN EACH SIPM: " + str(len(Sdf_DoublePulses)))
    plt.hist(np.hstack(Sdf_DoublePulses['SiPMhitT']),bins=30,range=(0,67000),alpha=0.75,histtype='step',linewidth=6,color='green')
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("SiPM pulse time distribution for events with two pulses in each SiPM")
    plt.xlabel("SiPM times (ns)")
    plt.show()

    #Get prompt clusters within a 100 ns window of the mean SiPM single pulse time
    LateClusterPEs = []
    LateClusterQs = []
    for j in Sdf_DoublePulses.index.values:  #disgusting...
        LateSiPMs = np.where(np.array(Sdf_DoublePulses['SiPMhitT'][j])>2000)[0]
        LateTimes = np.array(Sdf_DoublePulses["SiPMhitT"][j])[LateSiPMs]
        LateCharges = np.array(Sdf_DoublePulses["SiPMhitQ"][j])[LateSiPMs]
        for k,time in enumerate(LateTimes):
            LateClusterQs.append(LateCharges[k])
            print("SIPM TIME: " + str(time))
            Myhit_inds = np.where(abs(np.array(Sdf_DoublePulses['hitT'][j]) - LateTimes[k])<1000)[0]
            ClusterQ = np.sum(np.array(Sdf_DoublePulses['hitPE'][j])[Myhit_inds])
            print("CHARGE NEAR SIPM: " + str(ClusterQ))
            LateClusterPEs.append(ClusterQ)

    labels = {'title': 'Comparison of total nearby PE to late SiPM Charge \n (Position 0, AmBe source installed)', 
            'xlabel': 'Cluster PE', 'ylabel': 'Total SiPM charge [nC]'}
    ranges = {'xbins': 30, 'ybins':20, 'xrange':[0,40],'yrange':[0,0.5],'promptTime':2000}
    #abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    plt.hist2d(LateClusterPEs,LateClusterQs, bins=(ranges['xbins'],ranges['ybins']),
            range=[ranges['xrange'],ranges['yrange']],
            cmap = plt.cm.inferno)
    plt.colorbar()
    plt.title(labels['title'])
    plt.xlabel(labels['xlabel'])
    plt.ylabel(labels['ylabel'])
    plt.show()

if __name__=='__main__':
    slist = glob.glob(SIGNAL_DIR+"*.ntuple.root")

    mybranches = ['eventNumber','eventTimeTank','clusterTime','SiPMhitT','SiPMhitQ','SiPMhitAmplitude','clusterChargeBalance','clusterPE','SiPM1NPulses','SiPM2NPulses','SiPMNum','clusterHits','hitPE','hitT']

    SProcessor = rp.ROOTProcessor(treename="phaseIITriggerTree")
    for f1 in slist:
        SProcessor.addROOTFile(f1,branches_to_get=mybranches)
    Sdata = SProcessor.getProcessedData()
    Sdf_trig = pd.DataFrame(Sdata)
    
    TwoPulsePlot(Sdf_trig)



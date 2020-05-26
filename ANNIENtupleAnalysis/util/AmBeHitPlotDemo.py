#This script has functions needed to make a Cumulative Distribution Plot from
#Different variables output in  PhaseIITreeMaker root file.

import glob

import sys
import uproot
import lib.ROOTProcessor as rp
import lib.EventSelection as es
import lib.HitsPlotter as hp
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as scp
import numpy as np

SIGNAL_DIR = "./Data/V3_5PE100ns/CentralData/"
#BKG_DIR = "./Data/BkgCentralData/"
BKG_DIR = "./Data/V3_5PE100ns/BkgCentralData/"

if __name__=='__main__':
    #flist = glob.glob(BKG_DIR+"*.ntuple.root")
    flist = glob.glob(SIGNAL_DIR+"*.ntuple.root")

    mybranches = ['eventNumber','eventTimeTank','clusterTime','clusterChargeBalance','SiPMNum','SiPMhitT','hitX','hitY','hitZ','hitT','hitQ','hitPE','hitDetID','SiPM1NPulses','SiPM2NPulses','clusterPE']
    SProcessor = rp.ROOTProcessor(treename="phaseIITankClusterTree")
    for f1 in flist:
        SProcessor.addROOTFile(f1,branches_to_get=mybranches)
    Sdata = SProcessor.getProcessedData()
    Sdf = pd.DataFrame(Sdata)

    SProcessor = rp.ROOTProcessor(treename="phaseIITriggerTree")
    for f1 in flist:
        SProcessor.addROOTFile(f1,branches_to_get=mybranches)
    Sdata = SProcessor.getProcessedData()
    Sdf_trig = pd.DataFrame(Sdata)

    Sdf_SinglePulses = es.SingleSiPMPulses(Sdf)
    Sdf_CleanPrompt = es.NoPromptClusters(Sdf_SinglePulses,2000)
    Sdf_CleanWindow = es.NoBurstClusters(Sdf_CleanPrompt,2000,150)
    Sdf_LateWindow = Sdf_CleanWindow.loc[(Sdf_CleanWindow["clusterTime"]>2000)].reset_index(drop=True)
    Sdf_trig_goodSiPM = Sdf_trig.loc[(Sdf_trig['SiPM1NPulses']==1) & (Sdf_trig['SiPM2NPulses']==1)].reset_index(drop=True)
    Sdf_trig_CleanPrompt = es.NoPromptClusters_WholeFile(Sdf_SinglePulses,Sdf_trig_goodSiPM,2000)
    Sdf_trig_CleanWindow = es.NoBurst_WholeFile(Sdf_CleanPrompt,Sdf_trig_CleanPrompt,2000,150)
    Sdf_Mid = Sdf_SinglePulses.loc[(Sdf_SinglePulses["clusterChargeBalance"]>0.4)&(Sdf_SinglePulses["clusterChargeBalance"]<0.6)].reset_index(drop=True)
    Sdf_Lo = Sdf_SinglePulses.loc[(Sdf_SinglePulses["clusterChargeBalance"]<0.4)].reset_index(drop=True)

    enums = np.arange(10,20,1)
    for j in enums:
        hp.YVSTheta(Sdf_Lo,j,'hitT','Hit Times (ns)')
        hp.YVSTheta(Sdf_Lo,j,'hitPE','Hit PE count')
        hp.YVSTheta_Nhit(Sdf_Lo,j)

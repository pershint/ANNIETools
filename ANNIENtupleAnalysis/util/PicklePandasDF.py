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

#SIGNAL_DIR = "./Data/CentralData2020/Source/"
#BKG_DIR = "./Data/CentralData2020/Background/"
SIGNAL_DIR = "./Data/V3_5PE100ns/Pos0Data/"
BKG_DIR = "./Data/V3_5PE100ns/BkgPos0Data/"

expoPFlat= lambda x,C1,tau,mu,B: C1*np.exp(-(x-mu)/tau) + B

if __name__=='__main__':
    slist = glob.glob(SIGNAL_DIR+"*.ntuple.root")
    blist = glob.glob(BKG_DIR+"*.ntuple.root")

    mybranches = ['eventNumber','eventTimeTank','clusterTime','SiPMhitQ','SiPMhitAmplitude','SiPMNum','SiPMhitT','hitT','hitQ','hitPE','hitDetID','clusterChargeBalance','clusterPE','SiPM1NPulses','SiPM2NPulses']
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
    #There was an issue with duplicate entries being filled into the tree.  Resolve these.
    for i in Sdf_trig.index.values:
        all_charges = Sdf_trig['SiPMhitQ'][i]
        seen_charges = []
        unique_indices = []
        for j,q in enumerate(all_charges):
            if q not in seen_charges:
                unique_indices.append(j)
                seen_charges.append(q)
        #Sdf_trig['SiPMhitQ'][i] = list(set(Sdf_trig['SiPMhitQ'][i]))
        Sdf_trig['SiPMhitQ'][i] = list(np.array(Sdf_trig['SiPMhitQ'][i])[unique_indices])
        Sdf_trig['SiPMhitT'][i] = list(np.array(Sdf_trig['SiPMhitT'][i])[unique_indices])
        Sdf_trig['SiPMNum'][i] = list(np.array(Sdf_trig['SiPMNum'][i])[unique_indices])
        Sdf_trig['SiPMhitAmplitude'][i] = list(np.array(Sdf_trig['SiPMhitAmplitude'][i])[unique_indices])

    BProcessor = rp.ROOTProcessor(treename="phaseIITriggerTree")
    for f1 in blist:
        BProcessor.addROOTFile(f1,branches_to_get=mybranches)
    Bdata = BProcessor.getProcessedData()
    Bdf_trig = pd.DataFrame(Bdata)
    for i in Bdf_trig.index.values:
        all_charges = Bdf_trig['SiPMhitQ'][i]
        seen_charges = []
        unique_indices = []
        for j,q in enumerate(all_charges):
            if q not in seen_charges:
                unique_indices.append(j)
                seen_charges.append(q)
        #Bdf_trig['SiPMhitQ'][i] = list(set(Bdf_trig['SiPMhitQ'][i]))
        Bdf_trig['SiPMhitQ'][i] = list(np.array(Bdf_trig['SiPMhitQ'][i])[unique_indices])
        Bdf_trig['SiPMhitT'][i] = list(np.array(Bdf_trig['SiPMhitT'][i])[unique_indices])
        Bdf_trig['SiPMNum'][i] = list(np.array(Bdf_trig['SiPMNum'][i])[unique_indices])
        Bdf_trig['SiPMhitAmplitude'][i] = list(np.array(Bdf_trig['SiPMhitAmplitude'][i])[unique_indices])

    #Pickle all dataframes
    Sdf.to_pickle("./SdfPos0.pkl")
    Sdf_trig.to_pickle("./Sdf_trigPos0.pkl")
    Bdf.to_pickle("./BdfPos0.pkl")
    Bdf_trig.to_pickle("./Bdf_trigPos0.pkl")


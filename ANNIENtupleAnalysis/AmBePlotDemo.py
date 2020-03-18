#This script has functions needed to make a Cumulative Distribution Plot from
#Different variables output in  PhaseIITreeMaker root file.

import glob

import sys
import uproot
import lib.ROOTProcessor as rp
import lib.AmBePlots as abp
import pandas as pd

if __name__=='__main__':
    if str(sys.argv[1])=="--help":
        print(("USAGE: python AmBePlots.py [directory_with_files]"))
        sys.exit(0)
    f1list = glob.glob(str(sys.argv[1])+"*.ntuple.root")

    livetime_estimate = abp.EstimateLivetime(f1list)
    print("LIVETIME ESTIMATE IN SECONDS IS: " + str(livetime_estimate))

    mybranches = ['eventNumber','clusterTime','SiPMhitQ','SiPMNum','SiPMhitT','hitT','hitQ','SiPMhitAmplitude','clusterChargeBalance','clusterChargePointX','SiPM1NPulses','SiPM2NPulses','clusterChargePointZ']
    f1Processor = rp.ROOTProcessor(treename="phaseIITankClusterTree")
    for f1 in f1list:
        f1Processor.addROOTFile(f1,branches_to_get=mybranches)
    f1data = f1Processor.getProcessedData()
    f1data_pd = pd.DataFrame(f1data)


    abp.MakeClusterTimeDistribution(f1data_pd)
    labels = {'title': "Amplitude of SiPM1 hits", 'xlabel':'Pulse amplitude [V]'}
    ranges = {'bins': 100, 'range':(0,0.1)}
    abp.MakeSiPMVariableDistribution(f1data_pd, "SiPMhitAmplitude",1,labels,ranges,False)
    abp.MakeSiPMVariableDistribution(f1data_pd, "SiPMhitAmplitude",1,labels,ranges,True)

    labels = {'title': "Pulse times in SiPM1", 'xlabel':'Pulse peak time [ns]'}
    ranges = {'bins': 500, 'range':(0,65000)}
    abp.MakeSiPMVariableDistribution(f1data_pd, "SiPMhitT",1,labels,ranges,False)

    labels = {'title': "Hit times in PMTs", 'xlabel':'Pulse peak time [ns]'}
    ranges = {'bins': 500, 'range':(0,65000)}
    abp.MakePMTVariableDistribution(f1data_pd, "hitT",labels,ranges,False)


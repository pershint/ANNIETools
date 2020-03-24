#This script has functions needed to make a Cumulative Distribution Plot from
#Different variables output in  PhaseIITreeMaker root file.

import glob

import sys
import uproot
import lib.ROOTProcessor as rp
import lib.EventSelection as es
import lib.ProfileLikelihoodBuilder as plb
import lib.AmBePlots as abp
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as scp
import numpy as np
import scipy.misc as scm

SIGNAL_DIRS = ["./Data/BeamData/smallBeam/"]
SIGNAL_LABELS = ['Beam']
BKG_DIR = "./Data/BkgCentralData/"

expoPFlat= lambda x,C1,tau,mu,B: C1*np.exp(-(x-mu)/tau) + B
mypoisson = lambda x,mu: (mu**x)*np.exp(-mu)/scm.factorial(x)

def GetDataFrame(mytreename,mybranches,filelist):
    RProcessor = rp.ROOTProcessor(treename=mytreename)
    for f1 in filelist:
        RProcessor.addROOTFile(f1,branches_to_get=mybranches)
    data = RProcessor.getProcessedData()
    df = pd.DataFrame(data)
    return df

def BeamPlotDemo(PositionDict, Bdf, Bdf_trig):
    Sdf = PositionDict["Beam"][0]
    Sdf_trig = PositionDict["Beam"][1]
    Sdf_mrd = PositionDict["Beam"][2]
    #First, estimate the mean neutrons generated per acquisition due to bkgs
    Bdf_SinglePulses = es.SingleSiPMPulses(Bdf)
    Bdf_latewindow = Bdf_SinglePulses.loc[(Bdf_SinglePulses['clusterTime']>20000)].reset_index(drop=True)
    
    #Let's list some plots we want to make here:
    #  - Get delayed clusters that have eventTimeTanks matching the prompt/delayed
    #    cluster times.  Apply
    #  - clusterPE>10 and clusterChargeBalance<0.4 and plot the time distribution


    Sdf_prompt = Sdf.loc[Sdf['clusterTime']<2000].reset_index(drop=True)
    plt.hist(Sdf_prompt['clusterTime'],bins=100,range=(0,2000))
    plt.title("Prompt window Tank cluster times")
    plt.xlabel("Cluster time [ns]")
    plt.show()
    print("TOTAL PROMPT CLUSTERS: " + str(len(Sdf_prompt)))

    Sdf_maxPE = es.MaxPEClusters(Sdf_prompt)
    plt.hist(Sdf_maxPE['clusterTime'],bins=100,range=(0,2000))
    plt.title("Prompt window Tank cluster times (Highest PE cluster in each event)")
    plt.xlabel("Cluster time [ns]")
    plt.show()
    print("TOTAL MAXPE PROMPT CLUSTERS: " + str(len(Sdf_maxPE)))



    Sdf_mrd_maxhit = es.MaxHitClusters(Sdf_mrd)

    #Now, get the clusterTime pairs corresponding to matching triggers
    TankIndices, MRDIndices = es.MatchingEventTimes(Sdf_maxPE,Sdf_mrd_maxhit)
    TankTimes = Sdf_maxPE["clusterTime"].values[TankIndices]
    MRDTimes = Sdf_mrd_maxhit["clusterTime"].values[MRDIndices]
    plt.scatter(TankTimes,MRDTimes,marker='o',s=13)
    plt.title("Tank and MRD cluster times in acquisitions \n (Highest PE and highest paddle hit clusters)")
    plt.xlabel("Tank Cluster time [ns]")
    plt.ylabel("MRD Cluster time [ns]")
    plt.show()

    plt.hist(MRDTimes - TankTimes, bins = 30)
    plt.title("Difference in MRD and Tank cluster times in acquisitions \n (Highest PE and highest paddle hit clusters)")
    plt.xlabel("MRD cluster time - Tank cluster time")
    plt.show()

    clusterIndices_match = np.where(((MRDTimes - TankTimes)<800) & ((MRDTimes - TankTimes) > 200))[0]
    print(clusterIndices_match)
    MRDIndices_match = MRDIndices[clusterIndices_match]
    plt.hist(Sdf_mrd_maxhit['clusterTime'].values,bins=80,range=(0,4000),label="All MRD clusters")
    plt.hist(Sdf_mrd_maxhit['clusterTime'].values[MRDIndices_match],bins=80,range=(0,4000),label="Tank Cluster Match")
    plt.title("Prompt window MRD cluster times \n (Highest nhit cluster in event)")
    plt.xlabel("Cluster time [ns]")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()


if __name__=='__main__':

    mybkgbranches = ['eventNumber','eventTimeTank','clusterTime','hitT','hitQ','hitPE','clusterChargeBalance','clusterPE','clusterMaxPE','clusterChargePointY','SiPM1NPulses','SiPM2NPulses']
    mybkgtrigbranches = ['eventNumber','eventTimeTank','eventTimeMRD','vetoHit','SiPM1NPulses','SiPM2NPulses']
    mybranches = ['eventNumber','eventTimeTank','clusterTime','hitT','hitQ','hitPE','clusterChargeBalance','clusterPE','clusterMaxPE','clusterChargePointY']
    mytrigbranches = ['eventNumber','eventTimeTank','eventTimeMRD','vetoHit']

    myMRDbranches = ['eventNumber','eventTimeTank','eventTimeMRD','clusterTime','clusterHits','vetoHit',
            'numClusterTracks','MRDTrackAngle','MRDPenetrationDepth','MRDEntryPointRadius','MRDEnergyLoss','MRDEnergyLossError','MRDTrackLength']

    blist = glob.glob(BKG_DIR+"*.ntuple.root")
    Bdf = GetDataFrame("phaseIITankClusterTree",mybkgbranches,blist)
    Bdf_trig = GetDataFrame("phaseIITriggerTree",mybranches,blist)

    PositionDict = {}
    for j,direc in enumerate(SIGNAL_DIRS):
        direcfiles = glob.glob(direc+"*.ntuple.root")

        livetime_estimate = abp.EstimateLivetime(direcfiles)
        print("SIGNAL LIVETIME ESTIMATE IN SECONDS IS: " + str(livetime_estimate))
        PositionDict[SIGNAL_LABELS[j]] = []
        PositionDict[SIGNAL_LABELS[j]].append(GetDataFrame("phaseIITankClusterTree",mybranches,direcfiles))
        PositionDict[SIGNAL_LABELS[j]].append(GetDataFrame("phaseIITriggerTree",mybranches,direcfiles))
        PositionDict[SIGNAL_LABELS[j]].append(GetDataFrame("phaseIIMRDClusterTree",myMRDbranches,direcfiles))

    print("THE GOOD STUFF")
    BeamPlotDemo(PositionDict,Bdf,Bdf_trig)
    #EstimateNeutronEfficiencyAllPosns(PositionDict,Bdf,Bdf_trig)



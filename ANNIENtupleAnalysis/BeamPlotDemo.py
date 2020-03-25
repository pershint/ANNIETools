#This script has functions needed to make a Cumulative Distribution Plot from
#Different variables output in  PhaseIITreeMaker root file.

import glob

import sys
import uproot
import lib.ROOTProcessor as rp
import lib.EventSelection as es
import lib.ProfileLikelihoodBuilder as plb
import lib.BeamPlots as bp
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as scp
import numpy as np
import scipy.misc as scm

SIGNAL_DIRS = ["./Data/BeamData/"]
SIGNAL_LABELS = ['Beam']
BKG_DIR = "./Data/BkgCentralData/"

PEPERMEV = 12.

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

    #Get largest cluster in each acquisition in prompt window
    Sdf_maxPE = es.MaxPEClusters(Sdf_prompt)
    Sdf_mrd_maxhit = es.MaxHitClusters(Sdf_mrd)

    #Now, get the clusterTime pairs that are in the same triggers 
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

    #Get indices for MRD/Tank cluster times within the coincident window
    clusterIndices_match = np.where(((MRDTimes - TankTimes)<800) & ((MRDTimes - TankTimes) > 200))[0]
    MRDIndices_match = MRDIndices[clusterIndices_match]
    TankIndices_match = TankIndices[clusterIndices_match]

    plt.hist(Sdf_mrd_maxhit['clusterTime'].values,bins=80,range=(0,4000),label="All MRD clusters")
    plt.hist(Sdf_mrd_maxhit['clusterTime'].values[MRDIndices_match],bins=80,range=(0,4000),label="Tank Cluster Match")
    plt.title("Prompt window MRD cluster times \n (Highest nhit cluster in event)")
    plt.xlabel("Cluster time [ns]")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    plt.hist(Sdf_maxPE['clusterTime'].values,bins=80,range=(0,2000),label="All Tank clusters")
    plt.hist(Sdf_maxPE['clusterTime'].values[TankIndices_match],bins=80,range=(0,2000),label="MRD Cluster Match")
    plt.title("Prompt window Tank cluster times \n (Highest PE cluster in event)")
    plt.xlabel("Cluster time [ns]")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    #Now, Get all clusters past 12 us with the Event Times from TankIndices
    TankEventTimes_match = Sdf_maxPE["eventTimeTank"].values[TankIndices_match]
    Sdf_ClustersInMaxPE = es.FilterByEventTime(Sdf,TankEventTimes_match) #All clusters in events with a PMT/MRD match
    Sdf_ValidDelayedClusters = Sdf_ClustersInMaxPE.loc[Sdf_ClustersInMaxPE['clusterTime']>12000].reset_index(drop=True)

    plt.hist(Sdf.loc[Sdf["clusterTime"]>12000,"clusterTime"],bins=20,range=(12000,65000),label='No PMT/MRD pairing in prompt',alpha=0.8)
    plt.hist(Sdf_ValidDelayedClusters["clusterTime"], bins=20, range=(12000,67000),label='PMT/MRD pair required in prompt',alpha=0.8)
    plt.title("Delayed cluster times \n [Matching tank and MRD cluster in prompt window]")
    plt.xlabel("Cluster time [ns]")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    #Let's try to make the energy calibration plot
    plt.hist(Sdf_maxPE["clusterPE"], bins=40, range=(0,5000))
    plt.title("Prompt cluster PE \n (Highest PE cluster in prompt window)")
    plt.xlabel("Cluster PE")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()



    Sdf_MatchingPrompts = Sdf_maxPE.loc[TankIndices_match].reset_index(drop=True)
    Sdf_MatchingPromptsMRD = Sdf_mrd_maxhit.loc[MRDIndices_match].reset_index(drop=True)
    print("LEN OF TANK MATCHES: " + str(len(Sdf_MatchingPrompts)))
    print("LEN OF MRD MATCHES: " + str(len(Sdf_MatchingPromptsMRD)))
    HasVetoHit = np.where(Sdf_MatchingPromptsMRD["vetoHit"].values==1)[0]
    OneTrack = np.where(Sdf_MatchingPromptsMRD["numClusterTracks"].values==1)[0]
    ThroughGoingCandidates = np.intersect1d(HasVetoHit,OneTrack)
    Sdf_ThroughGoingCandidates = Sdf_MatchingPrompts.loc[ThroughGoingCandidates].reset_index(drop=True)
    plt.hist(Sdf_ThroughGoingCandidates["clusterPE"], bins=40, range=(0,5000))
    plt.title("Prompt cluster PE \n (Matching MRD cluster + one track + veto hit)")
    plt.xlabel("Cluster PE")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    #Now, apply track event selection
    TGValidTrackInds = es.SingleTrackSelection(Sdf_MatchingPromptsMRD.loc[ThroughGoingCandidates].reset_index(drop=True),100,1.0,10)
    print("VALID TRACK INDICES: " + str(TGValidTrackInds))
    Sdf_TGValidTracks = Sdf_ThroughGoingCandidates.loc[TGValidTrackInds].reset_index(drop=True)
    plt.hist(Sdf_TGValidTracks["clusterPE"], bins=40, range=(0,5000))
    plt.title("Prompt cluster PE \n (Matching MRD cluster + one track + veto hit + track cuts)")
    plt.xlabel("Cluster PE")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    #Now, apply aggressive track event selection
    TGValidTrackInds = es.SingleTrackSelection(Sdf_MatchingPromptsMRD.loc[ThroughGoingCandidates].reset_index(drop=True),60,0.4,60)
    Sdf_TGValidTracks = Sdf_ThroughGoingCandidates.loc[TGValidTrackInds].reset_index(drop=True)
    Sdf_TGValidTracks = Sdf_TGValidTracks.loc[Sdf_TGValidTracks['clusterHits']>70].reset_index(drop=True)
    plt.hist(Sdf_TGValidTracks["clusterPE"], bins=40, range=(0,5000))
    plt.title("Prompt cluster PE, aggressive cuts \n (Matching MRD cluster + one track + veto hit + track cuts)")
    plt.xlabel("Cluster PE")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    #Let's estimate the visible energy for coincident Tank/Cluster events
    NoVetoHit = np.where(Sdf_MatchingPromptsMRD["vetoHit"].values==0)[0]
    OneTrack = np.where(Sdf_MatchingPromptsMRD["numClusterTracks"].values==1)[0]
    NuCandidates = np.intersect1d(NoVetoHit,OneTrack)
    Sdf_NuCandidates = Sdf_MatchingPrompts.loc[NuCandidates].reset_index(drop=True)
    Sdf_NuCandidatesMRD = Sdf_MatchingPromptsMRD.loc[NuCandidates].reset_index(drop=True)
    NUC_PE = Sdf_NuCandidates['clusterPE'].values
    NUC_EVENTTIMETANKS = Sdf_NuCandidates['eventTimeTank'].values
    NUC_TANKMEV = NUC_PE / PEPERMEV
    NUC_MRDENERGY = es.SingleTrackEnergies(Sdf_NuCandidatesMRD)
    VISIBLE_ENERGY = (NUC_TANKMEV + NUC_MRDENERGY)/1000
    plt.hist(VISIBLE_ENERGY,bins=30,range=(0,4))
    plt.xlabel("Visible energy estimate [GeV]")
    plt.title("Visible energy for single track event in tank and MRD")
    plt.show()

    #Last plots... Average delayed cluster multiplicity as a function of visible energy
    #Delayed cluster multiplicity for these candidates in a histogram
    NuEventClusters = es.FilterByEventTime(Sdf,Sdf_NuCandidates['eventTimeTank'].values)
    NuEventDelayedClusters = NuEventClusters.loc[NuEventClusters['clusterTime']>12000].reset_index(drop=True)
    energy_min = 0
    energy_max = 3
    EnergyBins, ClusterMultiplicity,ClusterMultiplicity_unc = bp.EstimateEnergyPerClusterRelation(VISIBLE_ENERGY,
            NUC_EVENTTIMETANKS, NuEventDelayedClusters,energy_min,energy_max,10)
    plt.errorbar(EnergyBins, ClusterMultiplicity,yerr=ClusterMultiplicity_unc,linestyle='None',marker='o',markersize=10)
    plt.xlabel("Visible energy [GeV]")
    plt.ylabel("Neutron candidates per event")
    plt.title("Mean number of neutron candidates seen per interaction")
    plt.show()

if __name__=='__main__':

    mybkgbranches = ['eventNumber','eventTimeTank','clusterTime','hitT','hitQ','hitPE','clusterChargeBalance','clusterPE','clusterMaxPE','clusterChargePointY','SiPM1NPulses','SiPM2NPulses','clusterHits']
    mybkgtrigbranches = ['eventNumber','eventTimeTank','eventTimeMRD','vetoHit','SiPM1NPulses','SiPM2NPulses']
    mybranches = ['eventNumber','eventTimeTank','clusterTime','hitT','hitQ','hitPE','clusterChargeBalance','clusterPE','clusterMaxPE','clusterChargePointY','clusterHits']
    mytrigbranches = ['eventNumber','eventTimeTank','eventTimeMRD','vetoHit']

    myMRDbranches = ['eventNumber','eventTimeTank','eventTimeMRD','clusterTime','clusterHits','vetoHit',
            'numClusterTracks','MRDTrackAngle','MRDPenetrationDepth','MRDEntryPointRadius','MRDEnergyLoss','MRDEnergyLossError','MRDTrackLength']

    blist = glob.glob(BKG_DIR+"*.ntuple.root")
    Bdf = GetDataFrame("phaseIITankClusterTree",mybkgbranches,blist)
    Bdf_trig = GetDataFrame("phaseIITriggerTree",mybranches,blist)

    PositionDict = {}
    for j,direc in enumerate(SIGNAL_DIRS):
        direcfiles = glob.glob(direc+"*.ntuple.root")

        livetime_estimate = es.EstimateLivetime(direcfiles)
        print("SIGNAL LIVETIME ESTIMATE IN SECONDS IS: " + str(livetime_estimate))
        PositionDict[SIGNAL_LABELS[j]] = []
        PositionDict[SIGNAL_LABELS[j]].append(GetDataFrame("phaseIITankClusterTree",mybranches,direcfiles))
        PositionDict[SIGNAL_LABELS[j]].append(GetDataFrame("phaseIITriggerTree",mybranches,direcfiles))
        PositionDict[SIGNAL_LABELS[j]].append(GetDataFrame("phaseIIMRDClusterTree",myMRDbranches,direcfiles))

    print("THE GOOD STUFF")
    BeamPlotDemo(PositionDict,Bdf,Bdf_trig)
    #EstimateNeutronEfficiencyAllPosns(PositionDict,Bdf,Bdf_trig)



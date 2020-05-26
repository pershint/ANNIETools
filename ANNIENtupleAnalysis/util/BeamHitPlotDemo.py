#This script uses the event visualization in the HitsPlotter
#library to plot out some example events in beam data.

import glob

import sys
import uproot
import lib.ROOTProcessor as rp
import lib.EventSelection as es
import lib.HitsPlotter as hp
import lib.BeamPlots as bp
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as scp
import numpy as np

SIGNAL_DIRS = ["./Data/V3_5PE100ns/BeamData/"]
SIGNAL_LABELS = ['Beam']

def GetDataFrame(mytreename,mybranches,filelist): 
    RProcessor = rp.ROOTProcessor(treename=mytreename)
    for f1 in filelist:
        RProcessor.addROOTFile(f1,branches_to_get=mybranches)
    data = RProcessor.getProcessedData()
    df = pd.DataFrame(data)
    return df

if __name__=='__main__':

    mybranches = ['eventNumber','eventTimeTank','clusterTime','hitDetID','hitT','hitQ','hitPE','hitX','hitY','hitZ','clusterPE','clusterHits']
    mytrigbranches = ['eventNumber','eventTimeTank','eventTimeMRD','vetoHit']

    myMRDbranches = ['eventNumber','eventTimeTank','eventTimeMRD','clusterTime','clusterHits','vetoHit',
            'numClusterTracks','MRDTrackAngle','MRDPenetrationDepth','MRDEntryPointRadius','MRDEnergyLoss','MRDEnergyLossError','MRDTrackLength']

    PositionDict = {}
    for j,direc in enumerate(SIGNAL_DIRS):
        direcfiles = glob.glob(direc+"*.ntuple.root")

        livetime_estimate = es.EstimateLivetime(direcfiles)
        print("SIGNAL LIVETIME ESTIMATE IN SECONDS IS: " + str(livetime_estimate))
        PositionDict[SIGNAL_LABELS[j]] = []
        PositionDict[SIGNAL_LABELS[j]].append(GetDataFrame("phaseIITankClusterTree",mybranches,direcfiles))
        PositionDict[SIGNAL_LABELS[j]].append(GetDataFrame("phaseIITriggerTree",mytrigbranches,direcfiles))
        PositionDict[SIGNAL_LABELS[j]].append(GetDataFrame("phaseIIMRDClusterTree",myMRDbranches,direcfiles))

    Sdf = PositionDict["Beam"][0]
    Sdf_trig = PositionDict["Beam"][1]
    Sdf_mrd = PositionDict["Beam"][2]

    Sdf = Sdf.loc[Sdf["eventTimeTank"]>-9].reset_index(drop=True)
    Sdf_trig = Sdf_trig.loc[Sdf_trig["eventTimeTank"]>-9].reset_index(drop=True)
    Sdf_mrd = Sdf_mrd.loc[Sdf_mrd["eventTimeTank"]>-9].reset_index(drop=True)
    

    #Down-select to neutrino candidates and through-going candidates
    Sdf_prompt = Sdf.loc[Sdf['clusterTime']<2000].reset_index(drop=True)
    Sdf_maxPE = es.MaxPEClusters(Sdf_prompt)
    Sdf_mrd_maxhit = es.MaxHitClusters(Sdf_mrd)
    TankIndices, MRDIndices = es.MatchingEventTimes(Sdf_maxPE,Sdf_mrd_maxhit)
    TankTimes = Sdf_maxPE["clusterTime"].values[TankIndices]
    MRDTimes = Sdf_mrd_maxhit["clusterTime"].values[MRDIndices]
    clusterIndices_match = np.where(((MRDTimes - TankTimes)<800) & ((MRDTimes - TankTimes) > 600))[0]
    MRDIndices_match = MRDIndices[clusterIndices_match]
    TankIndices_match = TankIndices[clusterIndices_match]
    MatchedMRDTimes = Sdf_mrd_maxhit['clusterTime'].values[MRDIndices_match]
    MatchedTankTimes = Sdf_maxPE['clusterTime'].values[TankIndices_match]
    Sdf_MatchingPrompts = Sdf_maxPE.loc[TankIndices_match].reset_index(drop=True)
    Sdf_MatchingPromptsMRD = Sdf_mrd_maxhit.loc[MRDIndices_match].reset_index(drop=True)
    #print("LEN OF TANK MATCHES: " + str(len(Sdf_MatchingPrompts)))
    #print("LEN OF MRD MATCHES: " + str(len(Sdf_MatchingPromptsMRD)))
    HasVetoHit = np.where(Sdf_MatchingPromptsMRD["vetoHit"].values==1)[0]
    NoVetoHit = np.where(Sdf_MatchingPromptsMRD["vetoHit"].values==0)[0]
    OneTrack = np.where(Sdf_MatchingPromptsMRD["numClusterTracks"].values==1)[0]
    ThroughGoingCandidates = np.intersect1d(HasVetoHit,OneTrack)
    Sdf_ThroughGoingCandidates = Sdf_MatchingPrompts.loc[ThroughGoingCandidates].reset_index(drop=True)
    NuCandidates = np.intersect1d(NoVetoHit,OneTrack)
    Sdf_NuCandidates = Sdf_MatchingPrompts.loc[NuCandidates].reset_index(drop=True)

    NuMaxBackPE = bp.GetBackMaxes(Sdf_NuCandidates,'hitPE')
    NuTotalPE = Sdf_NuCandidates['clusterPE'].values
    ThruMaxBackPE = bp.GetBackMaxes(Sdf_ThroughGoingCandidates,'hitPE')
    ThruTotalPE = Sdf_ThroughGoingCandidates['clusterPE'].values

    print(NuMaxBackPE)
    plt.hist(NuMaxBackPE,bins=30,range=(0,300),alpha=0.5,histtype='step',linewidth=6,label=r'$\nu$ candidates',color='blue')
    plt.hist(ThruMaxBackPE,bins=30,range=(0,300),alpha=0.75,histtype='step',linewidth=6,color='green',label='Dirt $\mu$ candidates')
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Largest PE observed in upsteam PMTs ($z<0$ tubes)")
    plt.xlabel("Largest hit PE")
    plt.ylabel("Arb. units")
    plt.show()

    plt.hist2d(NuMaxBackPE,NuTotalPE, bins=(30,100),
            range=[[0,300],[0,5000]],
            cmap = plt.cm.inferno)
    plt.colorbar()
    plt.title("Largest back half PE compared to total cluster PE \n (Neutrino candidates, beam data)")
    plt.xlabel("Largest hit PE (z<0)")
    plt.ylabel("Tank cluster PE")
    plt.show()
    
    plt.hist2d(ThruMaxBackPE,ThruTotalPE, bins=(30,100),
            range=[[0,300],[0,5000]],
            cmap = plt.cm.inferno)
    plt.colorbar()
    plt.title("Largest back half PE compared to total cluster PE \n (Through-going candidates, beam data)")
    plt.xlabel("Largest hit PE (z<0)")
    plt.ylabel("Tank cluster PE")
    plt.show()

    Sdf_NuCandidates_hitZ = np.hstack(Sdf_NuCandidates['hitZ'].values)
    Sdf_ThroughGoingCandidates_hitZ = np.hstack(Sdf_ThroughGoingCandidates['hitZ'].values)
    Sdf_NuCandidates_hitPE = np.hstack(Sdf_NuCandidates['hitPE'].values)
    Sdf_ThroughGoingCandidates_hitPE = np.hstack(Sdf_ThroughGoingCandidates['hitPE'].values)
    NuBackPE = Sdf_NuCandidates_hitPE[np.where(Sdf_NuCandidates_hitZ<0)[0]]
    ThroughGoingBackPE = Sdf_ThroughGoingCandidates_hitPE[np.where(Sdf_ThroughGoingCandidates_hitZ<0)[0]]
    plt.hist(NuBackPE,bins=30,range=(0,300),alpha=0.5,histtype='step',linewidth=6,label=r'$\nu$ candidates',color='blue')
    plt.hist(ThroughGoingBackPE,bins=30,range=(0,300),alpha=0.75,histtype='step',linewidth=6,color='green',label='Dirt $\mu$ candidates')
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("PE distribution for hits in upsteam PMTs ($z<0$ tubes)")
    plt.xlabel("Hit PE")
    plt.ylabel("Arb. units")
    plt.show()

    plt.hist(np.hstack(Sdf_NuCandidates['clusterPE']),bins=30,range=(0,5000),alpha=0.5,histtype='step',linewidth=6,label=r'$\nu$ candidates',color='blue')
    plt.hist(np.hstack(Sdf_ThroughGoingCandidates['clusterPE']),bins=30,range=(0,5000),alpha=0.75,histtype='step',linewidth=6,color='green',label=r'$\nu$ candidates + veto hit')
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Cluster PE distribution for neutrino candidate events \n with/without a FMV paddle hit")
    plt.xlabel("Cluster PE")
    plt.ylabel("Arb. units")
    plt.show()



    enums = np.arange(150,250,1)
    for j in enums:
        mahboy = Sdf_NuCandidates
        #hp.YVSTheta(mahboy,j,'hitT','Hit Times (ns)',sum_duplicates=True)
        #hp.YVSTheta(mahboy,j,'hitPE','Hit PE count')
        hp.YVSTheta(mahboy,j,'hitPE','Hit PE count',sum_duplicates=True)
        #hp.YVSTheta(mahboy,j,'hitPE','Hit PE count',hitrange=[0,30],sum_duplicates=True)
        #hp.XVSZ_barrel(mahboy,j,'hitT','Hit Times (ns)')
        #hp.XVSZ_barrel(mahboy,j,'hitPE','Hit PE count')

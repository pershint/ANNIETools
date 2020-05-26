#This script has functions needed to make a Cumulative Distribution Plot from
#Different variables output in  PhaseIITreeMaker root file.

import glob

import sys
import uproot
import lib.ROOTProcessor as rp
import lib.EventSelection as es
import lib.ProfileLikelihoodBuilder as plb
import lib.AmBePlots as abp
import lib.BeamPlots as bp
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as scp
import numpy as np
import scipy.misc as scm

SIGNAL_DIRS = ["./Data/V3_5PE100ns/BeamData/"]
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

def BeamPlotDemo(PositionDict,MCdf):
    Sdf = PositionDict["Beam"][0]
    Sdf_trig = PositionDict["Beam"][1]
    Sdf_mrd = PositionDict["Beam"][2]

    Sdf = Sdf.loc[Sdf["eventTimeTank"]>-9].reset_index(drop=True)
    Sdf_trig = Sdf_trig.loc[Sdf_trig["eventTimeTank"]>-9].reset_index(drop=True)
    Sdf_mrd = Sdf_mrd.loc[Sdf_mrd["eventTimeTank"]>-9].reset_index(drop=True)
    
    Sdf_TankVeto = es.HasVetoHit_TankClusters(Sdf,Sdf_trig)
    print("NUM TANK CLUSTERS WITH VETO HIT: " + str(len(Sdf_TankVeto)))

    print("NUM TRIGS: " + str(len(Sdf_trig)))
    HasVetoHit = np.where(Sdf_mrd["vetoHit"].values==1)[0]
    print("NUM MRD CLUSTERS WITH VETO HIT: " + str(len(HasVetoHit)))
    #Let's list some plots we want to make here:
    #  - Get delayed clusters that have eventTimeTanks matching the prompt/delayed
    #    cluster times.  Apply
    #  - clusterPE>10 and clusterChargeBalance<0.4 and plot the time distribution


    Sdf_prompt = Sdf.loc[Sdf['clusterTime']<2000].reset_index(drop=True)
    plt.hist(Sdf_prompt['clusterTime'],bins=100,range=(0,2000))
    plt.title("Prompt window Tank cluster times")
    plt.xlabel("Cluster time [ns]")
    plt.show()
    print("TOTAL PROMPT TANK CLUSTERS: " + str(len(Sdf_prompt)))
    print("TOTAL PROMPT MRD CLUSTERS: " + str(len(Sdf_mrd)))
    
    plt.hist(Sdf_mrd['clusterTime'].values,bins=80,range=(0,4000),label="All MRD clusters")
    plt.title("Prompt window MRD cluster times")
    plt.xlabel("Cluster time [ns]")
    plt.show()

    #Get largest cluster in each acquisition in prompt window
    Sdf_maxPE = es.MaxPEClusters(Sdf_prompt)
    print("TOTAL HIGHEST PE PROMPT CLUSTERS: " + str(len(Sdf_maxPE)))
    Sdf_mrd_maxhit = es.MaxHitClusters(Sdf_mrd)
    print("TOTAL MOST PADDLE MRD CLUSTERS: " + str(len(Sdf_mrd_maxhit)))

    #Now, get the index number for clusterTime pairs in the same triggers 
    TankIndices, MRDIndices = es.MatchingEventTimes(Sdf_maxPE,Sdf_mrd_maxhit)
    TankTimes = Sdf_maxPE["clusterTime"].values[TankIndices]
    MRDTimes = Sdf_mrd_maxhit["clusterTime"].values[MRDIndices]
    Pairs_HaveVeto = Sdf_mrd_maxhit.loc[(Sdf_mrd_maxhit["vetoHit"].values[MRDIndices]==1)]
    print("NUM OF MRD CLUSTERS IN TRIG WITH A TANK CLUSTER: " + str(len(MRDTimes)))
    print("NUM OF MRD CLUSTERS WITH VETO IN SUBSET: " + str(len(Pairs_HaveVeto)))
    plt.scatter(TankTimes,MRDTimes,marker='o',s=15,color='blue',alpha=0.7)
    plt.title("Tank and MRD cluster times in prompt window \n (Largest PE tank clusters, largest paddle count MRD clusters)")
    plt.xlabel("Tank Cluster time [ns]")
    plt.ylabel("MRD Cluster time [ns]")
    plt.show()

    plt.hist(MRDTimes - TankTimes, bins = 160, color='blue', alpha=0.7)
    plt.axvline(x=700,color='black',linewidth=6)
    plt.axvline(x=800,color='black',linewidth=6)
    plt.title("Difference in MRD and Tank cluster times in acquisitions \n (Largest PE tank clusters, largest paddle count MRD clusters)")
    plt.xlabel("MRD cluster time - Tank cluster time [ns]")
    plt.show()

    #Get indices for MRD/Tank cluster times within the coincident window
    clusterIndices_match = np.where(((MRDTimes - TankTimes)<800) & ((MRDTimes - TankTimes) > 600))[0]
    MRDIndices_match = MRDIndices[clusterIndices_match]
    TankIndices_match = TankIndices[clusterIndices_match]

    MatchedMRDTimes = Sdf_mrd_maxhit['clusterTime'].values[MRDIndices_match]
    MatchedTankTimes = Sdf_maxPE['clusterTime'].values[TankIndices_match]
    print("NUMBER OF MATCHED TANKS: " + str(len(MatchedTankTimes)))
    print("NUMBER OF MATCHED MRDS: " + str(len(MatchedMRDTimes)))
    plt.hist(MatchedMRDTimes - MatchedTankTimes, bins = 80,color='blue')
    plt.axvline(x=700,color='black',linewidth=6)
    plt.axvline(x=800,color='black',linewidth=6)
    plt.title("Time distribution for matched MRD and Tank times")
    plt.xlabel("MRD cluster time - Tank cluster time [ns]")
    plt.show()


    plt.hist(Sdf_mrd['clusterTime'].values,bins=80,range=(0,4000),label="All MRD clusters")
    plt.hist(Sdf_mrd_maxhit['clusterTime'].values,bins=80,range=(0,4000),label="MRD clusters with most hits")
    #plt.hist(Sdf_mrd_maxhit['clusterTime'].values[MRDIndices],bins=80,range=(0,4000),label="+ Tank Cluster pair")
    plt.hist(Sdf_mrd_maxhit['clusterTime'].values[MRDIndices_match],bins=80,range=(0,4000),label="+ Tank cluster match")
    plt.title("Prompt window MRD cluster times \n event selection impact")
    plt.xlabel("Cluster time [ns]")
    plt.ylabel("Number of clusters")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()


    plt.hist(Sdf_prompt['clusterTime'].values,bins=80,range=(0,2000),label="All Tank clusters")
    plt.hist(Sdf_maxPE['clusterTime'].values,bins=80,range=(0,2000),label="Tank clusters with highest PE")
    #plt.hist(Sdf_maxPE['clusterTime'].values[TankIndices],bins=80,range=(0,2000),label="+ MRD Cluster Match")
    plt.hist(Sdf_maxPE['clusterTime'].values[TankIndices_match],bins=80,range=(0,2000),label="+ MRD cluster match")
    plt.title("Prompt window Tank cluster times \n event selection impact")
    plt.ylabel("Number of clusters")
    plt.xlabel("Cluster time [ns]")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    #Now, Get all clusters past 12 us with the Event Times from TankIndices
    TankEventTimes_match = Sdf_maxPE["eventTimeTank"].values[TankIndices_match]
    Sdf_ClustersInPromptCandidates = es.FilterByEventTime(Sdf,TankEventTimes_match) #All clusters in events with a PMT/MRD match
    print("ALL CLUSTER COUNT IN EVENTS WITH PMT/MRD ACTIVITY: " + str(len(Sdf_ClustersInPromptCandidates)))
    plt.hist(Sdf_ClustersInPromptCandidates['clusterTime'].values,bins=40,range=(0,67000),label="All clusters")
    plt.title("All tank cluster times \n (Acquisitions with valid prompt event selection)")
    plt.ylabel("Number of clusters")
    plt.xlabel("Cluster time [ns]")
    plt.show()

    print("CLUSTER COUNT IN EVENTS BEFORE 2 US: " + str(len(Sdf_ClustersInPromptCandidates.loc[Sdf_ClustersInPromptCandidates["clusterTime"]<2000].values)))
    Sdf_ValidDelayedClusters = Sdf_ClustersInPromptCandidates.loc[Sdf_ClustersInPromptCandidates['clusterTime']>12000].reset_index(drop=True)
    Sdf_ValidDelayedClustersCB = Sdf_ClustersInPromptCandidates.loc[Sdf_ClustersInPromptCandidates['clusterChargeBalance']<0.4].reset_index(drop=True)
    print("CLUSTER COUNT IN EVENTS WITH PMT/MRD ACTIVITY PAST 12 US: " + str(len(Sdf_ValidDelayedClusters)))

    plt.hist(Sdf.loc[Sdf["clusterTime"]>12000,"clusterTime"],bins=20,range=(12000,65000),label='No PMT/MRD pairing in prompt',alpha=0.8)
    plt.hist(Sdf_ValidDelayedClusters["clusterTime"], bins=20, range=(12000,65000),label='PMT/MRD pair required in prompt',alpha=0.8)
    plt.hist(Sdf_ValidDelayedClustersCB["clusterTime"], bins=20, range=(12000,65000),label=' + CB < 0.4',alpha=0.8)
    plt.title("Delayed cluster times in beam runs")
    plt.ylabel("Number of clusters")
    plt.xlabel("Cluster time [ns]")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    #Let's try to make the energy calibration plot
    #plt.hist(Sdf_maxPE["clusterPE"], bins=40, range=(0,5000))
    #plt.title("Prompt cluster PE \n (Highest PE cluster in prompt window)")
    #plt.xlabel("Cluster PE")
    #leg = plt.legend(loc=1,fontsize=24)
    #leg.set_frame_on(True)
    #leg.draw_frame(True)
    #plt.show()


    Sdf_MatchingPrompts = Sdf_maxPE.loc[TankIndices_match].reset_index(drop=True)
    Sdf_MatchingPromptsMRD = Sdf_mrd_maxhit.loc[MRDIndices_match].reset_index(drop=True)
    #print("LEN OF TANK MATCHES: " + str(len(Sdf_MatchingPrompts)))
    #print("LEN OF MRD MATCHES: " + str(len(Sdf_MatchingPromptsMRD)))
    HasVetoHit = np.where(Sdf_MatchingPromptsMRD["vetoHit"].values==1)[0]
    print("NUMBER OF INTERACTIONS WITH A VETO HIT: " + str(len(HasVetoHit)))
    OneTrack = np.where(Sdf_MatchingPromptsMRD["numClusterTracks"].values==1)[0]
    print("NUMBER OF INTERACTIONS WITH ONE TRACK: " + str(len(OneTrack)))
    ThroughGoingCandidates = np.intersect1d(HasVetoHit,OneTrack)
    Sdf_ThroughGoingCandidates = Sdf_MatchingPrompts.loc[ThroughGoingCandidates].reset_index(drop=True)
    #plt.hist(Sdf_ThroughGoingCandidates["clusterPE"], bins=40, range=(0,5000))
    #plt.title("Prompt cluster PE \n (Matching MRD cluster + one track + veto hit)")
    #plt.xlabel("Cluster PE")
    #leg = plt.legend(loc=1,fontsize=24)
    #leg.set_frame_on(True)
    #leg.draw_frame(True)
    #plt.show()

    ##Now, apply track event selection
    #TGValidTrackInds = es.SingleTrackSelection(Sdf_MatchingPromptsMRD.loc[ThroughGoingCandidates].reset_index(drop=True),100,1.0,10)
    #print("VALID TRACK INDICES: " + str(TGValidTrackInds))
    #Sdf_TGValidTracks = Sdf_ThroughGoingCandidates.loc[TGValidTrackInds].reset_index(drop=True)
    #plt.hist(Sdf_TGValidTracks["clusterPE"], bins=40, range=(0,5000))
    #plt.title("Prompt cluster PE \n (Matching MRD cluster + one track + veto hit + track cuts)")
    #plt.xlabel("Cluster PE")
    #leg = plt.legend(loc=1,fontsize=24)
    #leg.set_frame_on(True)
    #leg.draw_frame(True)
    #plt.show()

    ##Now, apply aggressive track event selection
    #TGValidTrackInds = es.SingleTrackSelection(Sdf_MatchingPromptsMRD.loc[ThroughGoingCandidates].reset_index(drop=True),60,0.4,60)
    #Sdf_TGValidTracks = Sdf_ThroughGoingCandidates.loc[TGValidTrackInds].reset_index(drop=True)
    #Sdf_TGValidTracks = Sdf_TGValidTracks.loc[Sdf_TGValidTracks['clusterHits']>70].reset_index(drop=True)
    #plt.hist(Sdf_TGValidTracks["clusterPE"], bins=40, range=(0,5000))
    #plt.title("Prompt cluster PE, aggressive cuts \n (Matching MRD cluster + one track + veto hit + track cuts)")
    #plt.xlabel("Cluster PE")
    #leg = plt.legend(loc=1,fontsize=24)
    #leg.set_frame_on(True)
    #leg.draw_frame(True)
    #plt.show()

    #Let's estimate the visible energy for coincident Tank/Cluster events
    NoVetoHit = np.where(Sdf_MatchingPromptsMRD["vetoHit"].values==0)[0]
    print("NUMBER OF INTERACTIONS WITH NO VETO HIT: " + str(len(NoVetoHit)))
    OneTrack = np.where(Sdf_MatchingPromptsMRD["numClusterTracks"].values==1)[0]
    NuCandidates = np.intersect1d(NoVetoHit,OneTrack)
    Sdf_NuCandidates = Sdf_MatchingPrompts.loc[NuCandidates].reset_index(drop=True)
    print("NUMBER OF NU INTERACTION CANDIDATES: " + str(len(Sdf_NuCandidates)))
    Sdf_NuCandidatesMRD = Sdf_MatchingPromptsMRD.loc[NuCandidates].reset_index(drop=True)
    NUC_PE = Sdf_NuCandidates['clusterPE'].values
    NUC_EVENTTIMETANKS = Sdf_NuCandidates['eventTimeTank'].values
    NUC_TANKMEV = NUC_PE / PEPERMEV
    NUC_MRDENERGY = es.SingleTrackEnergies(Sdf_NuCandidatesMRD)
    VISIBLE_ENERGY = (NUC_TANKMEV + NUC_MRDENERGY)/1000
    plt.hist(VISIBLE_ENERGY,bins=40,range=(0,4))
    plt.xlabel("Visible energy estimate [GeV]")
    plt.title("Visible energy for single track event in tank and MRD")
    plt.show()


    #Try to do a normalized comparison.
    mc_energy = MCdf["trueMuonEnergy"].values/1000.
    mc_energy = MCdf.loc[(MCdf["Pi0Count"]==0) & (MCdf["PiPlusCount"]==0) & 
        (MCdf["PiMinusCount"]==0),"trueMuonEnergy"].values/1000.
    mc_bins, mc_binedges = np.histogram(mc_energy,bins=15,range=(0,2))
    mc_binlefts =mc_binedges[0:len(mc_binedges)-1]
    binwidth = (mc_binlefts[1]-mc_binlefts[0])
    mc_binrights = mc_binlefts + binwidth
    mc_bincenters = mc_binlefts + (binwidth/2.)
    mc_bins_unc = np.sqrt(mc_bins)
    mc_bins_normed = mc_bins/np.sum(mc_bins)
    mc_bins_unc_normed = mc_bins_unc/np.sum(mc_bins)


    data_bins, data_binedges = np.histogram(VISIBLE_ENERGY,bins=15,range=(0,2))
    data_binlefts =data_binedges[0:len(data_binedges)-1]
    binwidth = (data_binlefts[1]-data_binlefts[0])
    data_bincenters = data_binlefts + (binwidth/2.)
    data_bins_unc = np.sqrt(data_bins)
    data_bins_normed = data_bins/np.sum(data_bins)
    data_bins_unc_normed = data_bins_unc/np.sum(data_bins)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax = abp.NiceBins(ax,mc_binlefts,mc_binrights,mc_bins_normed,'dark blue',"$E_{\mu}$ MC truth")
    ax.errorbar(data_bincenters,data_bins_normed,xerr=binwidth/2., yerr=data_bins_unc_normed,
            color='black',linestyle='None',markersize=6,label='ANNIE beam data')
    #plt.hist(mc_energy,density=True,bins=20,range=(0,2),label='$E_{\mu}$ MC Truth')
    #plt.hist(VISIBLE_ENERGY,normed=True,bins=20,range=(0,2), label='Beam data')
    plt.xlabel("Visible energy estimate [GeV]")
    plt.ylabel("Arb. units")
    plt.title("Visible energy of neutrino interaction candidates \n compared to MC truth information")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    zerobins = np.where(data_bins_unc==0)[0]
    data_bins_unc_normed[zerobins] = 1.15/np.sum(data_bins)
    print("CHISQUARE: " + str(np.sum(((data_bins_normed-mc_bins_normed)/np.sqrt(data_bins_unc_normed**2 + mc_bins_unc_normed**2))**2)))
    print("NDOF: " + str(len(data_bins_normed)))


    plt.hist(Sdf_maxPE['clusterTime'].values[TankIndices_match],bins=80,range=(0,2000),label="PMT clusters w/ matched MRD")
    plt.hist(Sdf_NuCandidates['clusterTime'],bins=80,range=(0,2000),label="+ single track and no veto")
    plt.title("Prompt window Tank cluster times \n (Highest PE cluster in event)")
    plt.xlabel("Cluster time [ns]")
    plt.ylabel("Number of clusters")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()



    #Last plots... Average delayed cluster multiplicity as a function of visible energy
    #Delayed cluster multiplicity for these candidates in a histogram
    NuEventClusters = es.FilterByEventTime(Sdf,Sdf_NuCandidates['eventTimeTank'].values)
    NuEventClusters_trig = es.FilterByEventTime(Sdf_trig,Sdf_NuCandidates['eventTimeTank'].values)
    NuEventDelayedClusters = NuEventClusters.loc[NuEventClusters['clusterTime']>12000].reset_index(drop=True)
    NuEventDelayedClusters = NuEventDelayedClusters.loc[NuEventDelayedClusters['clusterChargeBalance']<0.4].reset_index(drop=True)

    MData = abp.MakeClusterMultiplicityPlot(NuEventDelayedClusters,NuEventClusters_trig)
    plt.hist(MData,bins=6, range=(0,6), label="Beam data", alpha=0.5,histtype='stepfilled',linewidth=6,color='green')
    plt.hist(MData,bins=6, range=(0,6), label="Beam data", alpha=0.9,histtype='step',linewidth=6,color='green')
    Michael_NM = [64200, 26663, 7045, 4229, 3121, 2288, 1609, 1024, 258, 4]
    Michael_NMCounts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Michael_NMX = np.array(Michael_NM)+0.5
    #plt.errorbar(Michael_NMX, Michael_NM, xerr=0.5,color='black',label='GENIE MC')
    plt.xlabel("Neutron candidate multiplicity")
    plt.ylabel("Number of neutrino candidates")
    plt.title("Neutron candidate multiplicity for neutrino candidate events \n (CB<0.4, [12,67] $\mu s$ interval)")
    #leg = plt.legend(loc=1,fontsize=24)
    #leg.set_frame_on(True)
    #leg.draw_frame(True)
    plt.show()

    energy_min = 0
    energy_max = 1.4
    EnergyBins, ClusterMultiplicity,ClusterMultiplicity_unc = bp.EstimateEnergyPerClusterRelation(VISIBLE_ENERGY,
            NUC_EVENTTIMETANKS, NuEventDelayedClusters,energy_min,energy_max,8)
    errwidth = (EnergyBins[1] - EnergyBins[0])/2
    plt.errorbar(EnergyBins+errwidth, ClusterMultiplicity,xerr=errwidth,yerr=ClusterMultiplicity_unc,linestyle='None',marker='o',markersize=10,color='green',alpha=0.8,linewidth=5)
    plt.xlabel("Visible energy [GeV]")
    plt.ylabel("Neutron candidates per event")
    plt.title("Mean number of neutron candidates per neutrino candidate")
    plt.show()

    Sdf_NuCandidates_Aggressive = Sdf_NuCandidates.loc[(Sdf_NuCandidates['clusterPE']>500)&(Sdf_NuCandidates['clusterPE']<3500)]
    theinds = Sdf_NuCandidates_Aggressive.index.values
    Sdf_NuCandidatesMRD_Aggressive = Sdf_NuCandidatesMRD.loc[theinds]
    NUC_PE_AGG = Sdf_NuCandidates_Aggressive['clusterPE']
    NUC_EVENTTIMETANKS_AGG = Sdf_NuCandidates_Aggressive['eventTimeTank'].values
    NUC_TANKMEV_AGG = NUC_PE_AGG / PEPERMEV
    NUC_MRDENERGY_AGG = es.SingleTrackEnergies(Sdf_NuCandidatesMRD_Aggressive)
    VISIBLE_ENERGY_AGG = (NUC_TANKMEV_AGG + NUC_MRDENERGY_AGG)/1000
    NuEventClusters_Agg = es.FilterByEventTime(Sdf,Sdf_NuCandidates_Aggressive['eventTimeTank'].values)
    NuEventDelayedClusters_Agg = NuEventClusters_Agg.loc[NuEventClusters_Agg['clusterTime']>12000].reset_index(drop=True)
    NuEventDelayedClusters_Agg = NuEventDelayedClusters_Agg.loc[NuEventDelayedClusters_Agg['clusterChargeBalance']<0.4].reset_index(drop=True)

    energy_min = 0
    energy_max = 1.4
    EnergyBins, ClusterMultiplicity,ClusterMultiplicity_unc = bp.EstimateEnergyPerClusterRelation(VISIBLE_ENERGY_AGG,
            NUC_EVENTTIMETANKS_AGG, NuEventDelayedClusters_Agg,energy_min,energy_max,8)
    errwidth = (EnergyBins[1] - EnergyBins[0])/2
    plt.errorbar(EnergyBins+errwidth, ClusterMultiplicity,xerr=errwidth,yerr=ClusterMultiplicity_unc,linestyle='None',marker='o',markersize=10,color='green',alpha=0.8,linewidth=5)
    plt.xlabel("Visible energy [GeV]")
    plt.ylabel("Neutron candidates per event")
    plt.title("Mean number of neutron candidates per neutrino candidate \n (500 < Tank cluster PE < 3500)")
    plt.show()

    pe_min = 0
    pe_max = 6000
    PEBins, ClusterMultiplicity,ClusterMultiplicity_unc = bp.EstimatePEPerClusterRelation(NUC_PE,
            NUC_EVENTTIMETANKS, NuEventDelayedClusters,pe_min,pe_max,8)
    errwidth = (PEBins[1] - PEBins[0])/2
    plt.errorbar(PEBins+errwidth, ClusterMultiplicity,xerr=errwidth,yerr=ClusterMultiplicity_unc,linestyle='None',marker='o',markersize=10,color='green',alpha=0.8,linewidth=5)
    plt.xlabel("Visible tank energy [p.e.]")
    plt.ylabel("Neutron candidates per event")
    plt.title("Mean number of neutron candidates per neutrino candidate")
    plt.show()

if __name__=='__main__':

    mybranches = ['eventNumber','eventTimeTank','clusterTime','hitT','hitQ','hitPE','clusterChargeBalance','clusterPE','clusterMaxPE','clusterHits']
    mytrigbranches = ['eventNumber','eventTimeTank','eventTimeMRD','vetoHit']

    myMRDbranches = ['eventNumber','eventTimeTank','eventTimeMRD','clusterTime','clusterHits','vetoHit',
            'numClusterTracks','MRDTrackAngle','MRDPenetrationDepth','MRDEntryPointRadius','MRDEnergyLoss','MRDEnergyLossError','MRDTrackLength']

    MCbranches = ['trueMuonEnergy','Pi0Count','PiPlusCount','PiMinusCount']
    mclist = ["./Data/V3_5PE100ns/MCProfiles/PMTVolumeReco_Full_06262019.ntuple.root"]

    #mybkgbranches = ['eventNumber','eventTimeTank','clusterTime','hitT','hitQ','hitPE','clusterChargeBalance','clusterPE','clusterMaxPE','clusterChargePointZ','SiPM1NPulses','SiPM2NPulses','clusterHits']
    #mybkgtrigbranches = ['eventNumber','eventTimeTank','eventTimeMRD','vetoHit','SiPM1NPulses','SiPM2NPulses']
    #blist = glob.glob(BKG_DIR+"*.ntuple.root")
    #Bdf = GetDataFrame("phaseIITankClusterTree",mybkgbranches,blist)
    #Bdf_trig = GetDataFrame("phaseIITriggerTree",mybranches,blist)

    MCdf = GetDataFrame("phaseII",MCbranches,mclist)
    PositionDict = {}
    for j,direc in enumerate(SIGNAL_DIRS):
        direcfiles = glob.glob(direc+"*.ntuple.root")

        livetime_estimate = es.EstimateLivetime(direcfiles)
        print("SIGNAL LIVETIME ESTIMATE IN SECONDS IS: " + str(livetime_estimate))
        PositionDict[SIGNAL_LABELS[j]] = []
        PositionDict[SIGNAL_LABELS[j]].append(GetDataFrame("phaseIITankClusterTree",mybranches,direcfiles))
        PositionDict[SIGNAL_LABELS[j]].append(GetDataFrame("phaseIITriggerTree",mytrigbranches,direcfiles))
        PositionDict[SIGNAL_LABELS[j]].append(GetDataFrame("phaseIIMRDClusterTree",myMRDbranches,direcfiles))

    print("THE GOOD STUFF")
    BeamPlotDemo(PositionDict,MCdf)
    #EstimateNeutronEfficiencyAllPosns(PositionDict,Bdf,Bdf_trig)



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

SIGNAL_DIRS = ["./Data/V1/CentralData/","./Data/V1/Pos2Data/","./Data/V1/Pos3Data/",
"./Data/V1/Pos3P1mData/"]
SIGNAL_LABELS = ["Position 0","Position 1", "Position 2", "Position 3"]
BKG_DIR = "./Data/V1/BkgCentralData/"

expoPFlat= lambda x,C1,tau,mu,B: C1*np.exp(-(x-mu)/tau) + B
mypoisson = lambda x,mu: (mu**x)*np.exp(-mu)/scm.factorial(x)

def GetDataFrame(mytreename,mybranches,filelist):
    RProcessor = rp.ROOTProcessor(treename=mytreename)
    for f1 in filelist:
        RProcessor.addROOTFile(f1,branches_to_get=mybranches)
    data = RProcessor.getProcessedData()
    df = pd.DataFrame(data)
    return df

def EstimateNeutronEfficiency(Sdf, Bdf, Sdf_trig, Bdf_trig):

    #First, estimate the mean neutrons generated per acquisition due to bkgs
    Bdf_SinglePulses = es.SingleSiPMPulses(Bdf)
    Bdf_latewindow = Bdf_SinglePulses.loc[(Bdf_SinglePulses['clusterTime']>20000)].reset_index(drop=True)
    Bdf_trig_goodSiPM = Bdf_trig.loc[(Bdf_trig['SiPM1NPulses']==1) & (Bdf_trig['SiPM2NPulses']==1)].reset_index(drop=True)
    BkgScaleFactor = (67000-2000)/(67000-20000)  #Scale mean neutrons per window up by this
    MBData = abp.MakeClusterMultiplicityPlot(Bdf_latewindow,Bdf_trig_goodSiPM)
    print("MBData:" + str(MBData))
    Bbins,Bbin_edges = np.histogram(MBData,range=(0,6),bins=6)
    print("BINS AND EDGES")
    print(Bbins)
    print(Bbin_edges)
    Bbins_lefts = Bbin_edges[0:len(Bbin_edges)-1] #Combine clusters of 19 and 20 at end... negligible effect
    Bbins_normed = Bbins/float(np.sum(Bbins))
    Bbins_normed_unc = np.sqrt(Bbins)/float(np.sum(Bbins))
    zero_bins = np.where(Bbins_normed_unc==0)[0]
    Bbins_normed_unc[zero_bins] = 1/float(np.sum(Bbins))
    print("BBins_normed: " + str(Bbins_normed))
    init_params = [1]
    popt, pcov = scp.curve_fit(mypoisson, Bbins_lefts,Bbins_normed,p0=init_params, maxfev=6000,sigma=Bbins_normed_unc)
    print('BEST FIT MEAN: ' + str(popt[0]))
    myy = mypoisson(Bbins_lefts,popt[0])
    plt.errorbar(x=Bbins_lefts,y=Bbins_normed,yerr=Bbins_normed_unc,linestyle='None',marker='o',label='No source ($t>20 \, \mu s$)')
    plt.plot(Bbins_lefts,myy,marker='None',linewidth=6,label=r'Best poiss. fit $\mu= %s \pm %s$'%(str(np.round(popt[0],2)),str(np.round(np.sqrt(pcov[0]),2))),color='black')
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()
    
    #Get the normalized signal distribution
    Sdf_SinglePulses = es.SingleSiPMPulses(Sdf)
    Sdf_trig_SinglePulses = Sdf_trig.loc[(Sdf_trig['SiPM1NPulses']==1) & (Sdf_trig['SiPM2NPulses']==1)].reset_index(drop=True)
    Sdf_CleanPrompt = es.NoPromptClusters(Sdf_SinglePulses,2000)
    Sdf_trig_CleanPrompt = es.NoPromptClusters_WholeFile(Sdf_SinglePulses,Sdf_trig_SinglePulses,2000)
    MSData = abp.MakeClusterMultiplicityPlot(Sdf_CleanPrompt,Sdf_trig_CleanPrompt)
    Sbins,Sbin_edges = np.histogram(MSData,range=(0,6),bins=6)
    print("BINS AND EDGES")
    print(Sbins)
    print(Sbin_edges)
    Sbins_lefts = Sbin_edges[0:len(Bbin_edges)-1] #Combine clusters of 19 and 20 at end... negligible effect
    Sbins_normed = Sbins/float(np.sum(Sbins))
    Sbins_normed_unc = np.sqrt(Sbins)/float(np.sum(Sbins))
    plt.errorbar(x=Sbins_lefts,y=Sbins_normed,yerr=Sbins_normed_unc,linestyle='None',marker='o',label='Source ($t>2 \, \mu s$)')
    plt.errorbar(x=Bbins_lefts,y=Bbins_normed,yerr=Bbins_normed_unc,linestyle='None',marker='o',label='No source ($t>20 \, \mu s$)')
    plt.plot(Bbins_lefts,myy,marker='None',linewidth=6,label=r'Best poiss. fit $\mu= %s \pm %s$'%(str(np.round(popt[0],2)),str(np.round(np.sqrt(pcov[0]),2))),color='black')
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    #Cool, now the fun: We build a likelihood profile.
    PLBuilder = plb.ProfileLikelihoodBuilder()
    PLBuilder.SetBkgMean(BkgScaleFactor*popt[0])
    PLBuilder.SetBkgMeanUnc(np.sqrt(pcov[0][0]))
    NeutronProbProfile = np.arange(0,1,0.005)
    ChiSquare = PLBuilder.BuildLikelihoodProfile(NeutronProbProfile,Sbins_normed,Sbins_normed_unc,700000,Bbins_normed)
    ChiSquare_normed = ChiSquare/np.min(ChiSquare)
    plt.plot(NeutronProbProfile,ChiSquare_normed,marker='None',linewidth=4)
    plt.title("Weighted chi-square test parameter as a function of neutron capture efficiency")
    plt.xlabel("Neutron capture observation efficiency")
    plt.ylabel("$\chi^{2}$/$\chi^{2}_{min}$")
    plt.show()

def EstimateNeutronEfficiencyAllPosns(PositionDict,Bdf,Bdf_trig):

    #First, estimate the mean neutrons generated per acquisition due to bkgs
    Bdf_SinglePulses = es.SingleSiPMPulses(Bdf)
    Bdf_latewindow = Bdf_SinglePulses.loc[(Bdf_SinglePulses['clusterTime']>20000)].reset_index(drop=True)
    Bdf_trig_goodSiPM = Bdf_trig.loc[(Bdf_trig['SiPM1NPulses']==1) & (Bdf_trig['SiPM2NPulses']==1)].reset_index(drop=True)
    BkgScaleFactor = (67000-2000)/(67000-20000)  #Scale mean neutrons per window up by this
    MBData = abp.MakeClusterMultiplicityPlot(Bdf_latewindow,Bdf_trig_goodSiPM)
    print("MBData:" + str(MBData))
    Bbins,Bbin_edges = np.histogram(MBData,range=(0,6),bins=6)
    print("BINS AND EDGES")
    print(Bbins)
    print(Bbin_edges)
    Bbins_lefts = Bbin_edges[0:len(Bbin_edges)-1] #Combine clusters of 19 and 20 at end... negligible effect
    Bbins_normed = Bbins/float(np.sum(Bbins))
    Bbins_normed_unc = np.sqrt(Bbins)/float(np.sum(Bbins))
    zero_bins = np.where(Bbins_normed_unc==0)[0]
    Bbins_normed_unc[zero_bins] = 1/float(np.sum(Bbins))
    print("BBins_normed: " + str(Bbins_normed))
    init_params = [1]
    popt, pcov = scp.curve_fit(mypoisson, Bbins_lefts,Bbins_normed,p0=init_params, maxfev=6000,sigma=Bbins_normed_unc)
    print('BEST FIT MEAN: ' + str(popt[0]))
    myy = mypoisson(Bbins_lefts,popt[0])
    plt.errorbar(x=Bbins_lefts,y=Bbins_normed,yerr=Bbins_normed_unc,linestyle='None',marker='o',label='No source ($t>20 \, \mu s$)')
    plt.plot(Bbins_lefts,myy,marker='None',linewidth=6,label=r'Best poiss. fit $\mu= %s \pm %s$'%(str(np.round(popt[0],2)),str(np.round(np.sqrt(pcov[0]),2))),color='black')
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()
    
    #Get the normalized signal distribution
    for Position in PositionDict:
        Sdf = PositionDict[Position][0]
        Sdf_trig = PositionDict[Position][1]
        Sdf_SinglePulses = es.SingleSiPMPulses(Sdf)
        Sdf_trig_SinglePulses = Sdf_trig.loc[(Sdf_trig['SiPM1NPulses']==1) & (Sdf_trig['SiPM2NPulses']==1)].reset_index(drop=True)
        Sdf_CleanPrompt = es.NoPromptClusters(Sdf_SinglePulses,2000)
        Sdf_trig_CleanPrompt = es.NoPromptClusters_WholeFile(Sdf_SinglePulses,Sdf_trig_SinglePulses,2000)
        MSData = abp.MakeClusterMultiplicityPlot(Sdf_CleanPrompt,Sdf_trig_CleanPrompt)
        Sbins,Bbin_edges = np.histogram(MSData,range=(0,6),bins=6)
        print("BINS AND EDGES")
        Sbins_lefts = Bbin_edges[0:len(Bbin_edges)-1] #Combine clusters of 19 and 20 at end... negligible effect
        Sbins_normed = Sbins/float(np.sum(Sbins))
        Sbins_normed_unc = np.sqrt(Sbins)/float(np.sum(Sbins))
        #zero_bins = np.where(Sbins_normed_unc==0)[0]
        #Sbins_normed_unc[zero_bins] = 1/float(np.sum(Sbins))
        #Cool, now the fun: We build a likelihood profile.
        PLBuilder = plb.ProfileLikelihoodBuilder()
        PLBuilder.SetBkgMean(BkgScaleFactor*popt[0])
        PLBuilder.SetBkgMeanUnc(np.sqrt(pcov[0][0]))
        NeutronProbProfile = np.arange(0.2,0.8,0.005)
        #TODO: Instead of Shooting the background uncertainties, shoot values from the distribution
        ChiSquare = PLBuilder.BuildLikelihoodProfile(NeutronProbProfile,Sbins_normed,Sbins_normed_unc,600000,Bbins_normed)
        ChiSquare_normed = ChiSquare/np.min(ChiSquare)
        plt.plot(NeutronProbProfile,ChiSquare_normed,marker='None',linewidth=5,label=Position,alpha=0.75)
    plt.title("Profile likelihood of of neutron capture efficiency \n at different source deployment positions")
    plt.xlabel("Neutron capture efficiency")
    plt.ylabel("$\chi^{2}$/$\chi^{2}_{min}$")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()


if __name__=='__main__':
    blist = glob.glob(BKG_DIR+"*.ntuple.root")

    livetime_estimate = abp.EstimateLivetime(blist)
    print("BKG LIVETIME ESTIMATE IN SECONDS IS: " + str(livetime_estimate))

    mybranches = ['eventNumber','eventTimeTank','clusterTime','SiPMhitQ','SiPMNum','SiPMhitT','hitT','hitQ','hitPE','SiPMhitAmplitude','clusterChargeBalance','clusterPE','clusterMaxPE','SiPM1NPulses','SiPM2NPulses','clusterChargePointY']

    PositionDict = {}
    for j,direc in enumerate(SIGNAL_DIRS):
        direcfiles = glob.glob(direc+"*.ntuple.root")

        livetime_estimate = abp.EstimateLivetime(direcfiles)
        print("SIGNAL LIVETIME ESTIMATE IN SECONDS IS: " + str(livetime_estimate))
        PositionDict[SIGNAL_LABELS[j]] = []
        PositionDict[SIGNAL_LABELS[j]].append(GetDataFrame("phaseIITankClusterTree",mybranches,direcfiles))
        PositionDict[SIGNAL_LABELS[j]].append(GetDataFrame("phaseIITriggerTree",mybranches,direcfiles))

    Bdf = GetDataFrame("phaseIITankClusterTree",mybranches,blist)
    Bdf_trig = GetDataFrame("phaseIITriggerTree",mybranches,blist)

    #EstimateNeutronEfficiency(PositionDict["Position 0"][0],Bdf,PositionDict["Position 0"][1],Bdf_trig)
    EstimateNeutronEfficiencyAllPosns(PositionDict,Bdf,Bdf_trig)



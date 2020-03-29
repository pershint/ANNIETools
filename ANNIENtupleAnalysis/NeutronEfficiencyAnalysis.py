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
import seaborn as sns

#SIGNAL_DIRS = ["./Data/V3/CentralData/","./Data/V3/Pos2Data/","./Data/V3/Pos3Data/",
#"./Data/V3/Pos3P1mData/"]
#SIGNAL_LABELS = ["Position 0","Position 1", "Position 2", "Position 3"]
SIGNAL_DIRS = ["./Data/V3/CentralData/"]
SIGNAL_LABELS = ["Position 0"]
BKG_DIR = "./Data/V3/BkgCentralData/"

expoPFlat= lambda x,C1,tau,mu,B: C1*np.exp(-(x-mu)/tau) + B
mypoisson = lambda x,mu: (mu**x)*np.exp(-mu)/scm.factorial(x)
mypoissons = lambda x,R1,mu1,R2,mu2: R1*(mu1**x)*np.exp(-mu2)/scm.factorial(x) + R2*(mu2**x)*np.exp(-mu2)/scm.factorial(x)

BKG_WINDOW_START = 12000
SIGNAL_WINDOW_START = 2000

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
    Bdf_BurstCut = es.NoBurstClusters(Bdf_SinglePulses,BKG_WINDOW_START,150)
    Bdf_latewindow = Bdf_BurstCut.loc[(Bdf_BurstCut['clusterTime']>BKG_WINDOW_START)].reset_index(drop=True)
    Bdf_latewindow = Bdf_latewindow.loc[(Bdf_latewindow['clusterChargeBalance']<0.4)].reset_index(drop=True)
    Bdf_trig_goodSiPM = Bdf_trig.loc[(Bdf_trig['SiPM1NPulses']==1) & (Bdf_trig['SiPM2NPulses']==1)].reset_index(drop=True)
    Bdf_trig_BurstCut = es.NoBurst_WholeFile(Bdf_SinglePulses,Bdf_trig_goodSiPM,BKG_WINDOW_START,150)
    MBData = abp.MakeClusterMultiplicityPlot(Bdf_latewindow,Bdf_trig_BurstCut)
    plt.hist(MBData,bins=20, range=(0,20), alpha=0.2,histtype='stepfilled',linewidth=6)
    plt.hist(MBData,bins=20, range=(0,20), histtype='step',linewidth=6)
    plt.xlabel("Delayed cluster multiplicity")
    plt.title("Cluster multiplicity for delayed window of central background run \n (SiPM + Burst cut, [12,65] $\mu s$ window")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    print("MBData:" + str(MBData))
    Bbins,Bbin_edges = np.histogram(MBData,range=(0,5),bins=5)
    print("BINS AND EDGES")
    print(Bbins)
    print(Bbin_edges)
    Bbins_lefts = Bbin_edges[0:len(Bbin_edges)-1] #Combine clusters of 19 and 20 at end... negligible effect
    Bbins_normed = Bbins/float(np.sum(Bbins))
    Bbins_normed_unc = np.sqrt(Bbins)/float(np.sum(Bbins))
    zero_bins = np.where(Bbins_normed_unc==0)[0]
    Bbins_normed_unc[zero_bins] = 1.15/float(np.sum(Bbins))
    print("BBins_normed: " + str(Bbins_normed))
    init_params = [1]
    popt, pcov = scp.curve_fit(mypoisson, Bbins_lefts,Bbins_normed,p0=init_params, maxfev=6000,sigma=Bbins_normed_unc)
    #init_params = [5000,0.04,100,1]
    #popt, pcov = scp.curve_fit(mypoissons, Bbins_lefts,Bbins_normed,p0=init_params, maxfev=6000,sigma=Bbins_normed_unc)
    print('BEST FIT POPTS: ' + str(popt))
    myy = mypoisson(Bbins_lefts,popt[0])
    myy_upper = mypoisson(Bbins_lefts,popt[0]+np.sqrt(pcov[0][0]))
    #myy = mypoissons(Bbins_lefts,popt[0],popt[1],popt[2],popt[3])
    plt.errorbar(x=Bbins_lefts,y=Bbins_normed,yerr=Bbins_normed_unc,linestyle='None',marker='o',label='No source ($t>20 \, \mu s$)')
    plt.plot(Bbins_lefts,myy,marker='None',linewidth=6,label=r'Best poiss. fit $\mu= %s \pm %s$'%(str(np.round(popt[0],2)),str(np.round(np.sqrt(pcov[0]),2))),color='black')
    plt.plot(Bbins_lefts,myy_upper,marker='None',linewidth=6,label=r'Best poiss. fit upper bound',color='gray')
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()
    
    #Get the normalized signal distribution
    Sdf_SinglePulses = es.SingleSiPMPulses(Sdf)
    Sdf_trig_SinglePulses = Sdf_trig.loc[(Sdf_trig['SiPM1NPulses']==1) & (Sdf_trig['SiPM2NPulses']==1)].reset_index(drop=True)
    Sdf_CleanPrompt = es.NoPromptClusters(Sdf_SinglePulses,SIGNAL_WINDOW_START)
    Sdf_CleanWindow = es.NoBurstClusters(Sdf_CleanPrompt,SIGNAL_WINDOW_START,150)
    Sdf_CleanWindow_noCB = Sdf_CleanWindow.loc[Sdf_CleanWindow["clusterChargeBalance"]<0.4].reset_index(drop=True)
    Sdf_trig_CleanPrompt = es.NoPromptClusters_WholeFile(Sdf_SinglePulses,Sdf_trig_SinglePulses,2000)
    Sdf_trig_CleanWindow = es.NoBurst_WholeFile(Sdf_CleanPrompt,Sdf_trig_CleanPrompt,2000,150)
    MSData = abp.MakeClusterMultiplicityPlot(Sdf_CleanWindow_noCB,Sdf_trig_CleanWindow)
    plt.hist(MSData,bins=20, range=(0,20), alpha=0.2,histtype='stepfilled',linewidth=6)
    plt.hist(MSData,bins=20, range=(0,20), histtype='step',linewidth=6)
    plt.xlabel("Delayed cluster multiplicity")
    plt.title("Cluster multiplicity for delayed window of central source run \n (One pulse each SiPM, no prompt cluster)")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    Sbins,Sbin_edges = np.histogram(MSData,range=(0,5),bins=5)
    print("BINS AND EDGES")
    print(Sbins)
    print(Sbin_edges)
    Sbins_lefts = Sbin_edges[0:len(Bbin_edges)-1] #Combine clusters of 19 and 20 at end... negligible effect
    Sbins_normed = Sbins/float(np.sum(Sbins))
    Sbins_normed_unc = np.sqrt(Sbins)/float(np.sum(Sbins))
    zero_bins = np.where(Sbins_normed_unc==0)[0]
    Sbins_normed_unc[zero_bins] = 1.15/float(np.sum(Sbins)) #FIXME: This isn't quite right...
    plt.errorbar(x=Sbins_lefts,y=Sbins_normed,yerr=Sbins_normed_unc,linestyle='None',marker='o',label='Source ($t>2 \, \mu s$)',markersize=12)
    plt.errorbar(x=Bbins_lefts,y=Bbins_normed,yerr=Bbins_normed_unc,linestyle='None',marker='o',label='No source ($t>20 \, \mu s$)',markersize=12)
    plt.plot(Bbins_lefts,myy,marker='None',linewidth=6,label=r'Best poiss. fit $\mu= %s \pm %s$'%(str(np.round(popt[0],2)),str(np.round(np.sqrt(pcov[0]),2))),color='black')
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    #Cool, now the fun: We build a likelihood profile.
    PLBuilder = plb.ProfileLikelihoodBuilder()
    BkgScaleFactor = (67000-SIGNAL_WINDOW_START)/(67000-BKG_WINDOW_START)  #Scale mean neutrons per window up by this
    PLBuilder.SetBkgMean(BkgScaleFactor*popt[0])
    PLBuilder.SetBkgMeanUnc(np.sqrt(pcov[0][0]))
    NeutronProbProfile = np.arange(0.5,0.7,0.001)
    ChiSquare,lowestChiSqProfile = PLBuilder.BuildLikelihoodProfile(NeutronProbProfile,Sbins_normed,Sbins_normed_unc,700000,Bbins_normed)
    print("MINIMUM CHI SQUARE: " + str(np.min(ChiSquare)))
    ChiSquare_normed = ChiSquare/np.min(ChiSquare)
    plt.errorbar(x=Sbins_lefts,y=Sbins_normed,yerr=Sbins_normed_unc,linestyle='None',marker='o',label='Source ($t>2 \, \mu s$)',markersize=12)
    plt.plot(Sbins_lefts, lowestChiSqProfile,linestyle='None',marker='o',label='Best fit model profile',markersize=12)
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Best fit MC profile relative to central source data")
    plt.xlabel("Delayed cluster multiplicity")
    plt.show()

    plt.plot(NeutronProbProfile,ChiSquare_normed,marker='None',linewidth=4)
    plt.title("Weighted chi-square test parameter as a function of neutron capture efficiency")
    plt.xlabel("Neutron capture observation efficiency")
    plt.ylabel("$\chi^{2}$/$\chi^{2}_{min}$")
    plt.show()

    PLBuilder2D = plb.ProfileLikelihoodBuilder2D()
    neutron_efficiencies = np.arange(0.55,0.65,0.002)
    background_mean = np.arange(0.04,0.07,0.002)
    PLBuilder2D.SetEffProfile(neutron_efficiencies)
    PLBuilder2D.SetBkgMeanProfile(background_mean)
    x_var, y_var, ChiSquare,lowestChiSqProfile = PLBuilder2D.BuildLikelihoodProfile(Sbins_normed,Sbins_normed_unc,700000)
    print("MINIMUM CHI SQUARE: " + str(np.min(ChiSquare)))
    plt.errorbar(x=Sbins_lefts,y=Sbins_normed,yerr=Sbins_normed_unc,linestyle='None',marker='o',label='Source ($t>2 \, \mu s$)',markersize=12)
    plt.plot(Sbins_lefts, lowestChiSqProfile,linestyle='None',marker='o',label='Best fit model profile',markersize=12)
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Best fit MC profile relative to central source data")
    plt.xlabel("Delayed cluster multiplicity")
    plt.show()

    chisq_map = pd.DataFrame({"NeutronEff":np.round(x_var,3), "BkgPoisMean":np.round(y_var,3),"ChiSq":ChiSquare/np.min(ChiSquare)})
    cmap = chisq_map.pivot(index="NeutronEff",columns="BkgPoisMean",values="ChiSq")
    ax = sns.heatmap(cmap,vmin=1,vmax=7)
    plt.title("$\chi^{2}$/$\chi_{min}^{2}$ for profile likelihood parameters")
    plt.show()

    LowestInd = np.where(ChiSquare==np.min(ChiSquare))[0]
    best_eff = x_var[LowestInd]
    best_mean = y_var[LowestInd]
    best_eff_chisquareinds = np.where(x_var==best_eff)[0]
    best_eff_chisquares = ChiSquare[best_eff_chisquareinds]
    best_eff_bkgmeans = y_var[best_eff_chisquareinds]
    plt.plot(best_eff_bkgmeans,best_eff_chisquares/np.min(ChiSquare),marker='None',linewidth=4)
    plt.title("Goodness of fit varying background mean for best-fit \n neutron capture efficiency")
    plt.xlabel("Background mean [clusters/trigger]")
    plt.ylabel("$\chi^{2}$/$\chi^{2}_{min}$")
    plt.show()

    best_mean_chisquareinds = np.where(y_var==best_mean)[0]
    best_mean_chisquares = ChiSquare[best_mean_chisquareinds]
    best_mean_efficiencypro = x_var[best_mean_chisquareinds]
    plt.plot(best_mean_efficiencypro,best_mean_chisquares/np.min(ChiSquare),marker='None',linewidth=4)
    plt.title("Goodness of fit varying capture efficiency for best-fit \n background mean")
    plt.xlabel("Neutron capture efficiency")
    plt.ylabel("$\chi^{2}$/$\chi^{2}_{min}$")
    plt.show()



def EstimateNeutronEfficiencyAllPosns(PositionDict,Bdf,Bdf_trig):

    #First, estimate the mean neutrons generated per acquisition due to bkgs
    #Bdf_SinglePulses = es.SingleSiPMPulses(Bdf)
    #Bdf_latewindow = Bdf_SinglePulses.loc[(Bdf_SinglePulses['clusterTime']>12000)].reset_index(drop=True)
    #Bdf_latewindow = Bdf_latewindow.loc[(Bdf_latewindow['clusterChargeBalance']<0.4)].reset_index(drop=True)
    #Bdf_trig_goodSiPM = Bdf_trig.loc[(Bdf_trig['SiPM1NPulses']==1) & (Bdf_trig['SiPM2NPulses']==1)].reset_index(drop=True)
    #MBData = abp.MakeClusterMultiplicityPlot(Bdf_latewindow,Bdf_trig_goodSiPM)
    Bdf_SinglePulses = es.SingleSiPMPulses(Bdf)
    Bdf_BurstCut = es.NoBurstClusters(Bdf_SinglePulses,BKG_WINDOW_START,150)
    Bdf_latewindow = Bdf_BurstCut.loc[(Bdf_BurstCut['clusterTime']>BKG_WINDOW_START)].reset_index(drop=True)
    Bdf_latewindow = Bdf_latewindow.loc[(Bdf_latewindow['clusterChargeBalance']<0.4)].reset_index(drop=True)
    Bdf_trig_goodSiPM = Bdf_trig.loc[(Bdf_trig['SiPM1NPulses']==1) & (Bdf_trig['SiPM2NPulses']==1)].reset_index(drop=True)
    Bdf_trig_BurstCut = es.NoBurst_WholeFile(Bdf_SinglePulses,Bdf_trig_goodSiPM,BKG_WINDOW_START,150)
    MBData = abp.MakeClusterMultiplicityPlot(Bdf_latewindow,Bdf_trig_BurstCut)

    print("MBData:" + str(MBData))
    Bbins,Bbin_edges = np.histogram(MBData,range=(0,5),bins=5)
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
        #Sdf_SinglePulses = es.SingleSiPMPulses(Sdf)
        #Sdf_trig_SinglePulses = Sdf_trig.loc[(Sdf_trig['SiPM1NPulses']==1) & (Sdf_trig['SiPM2NPulses']==1)].reset_index(drop=True)
        #Sdf_CleanPrompt = es.NoPromptClusters(Sdf_SinglePulses,2000)
        #Sdf_CleanPrompt = Sdf_CleanPrompt.loc[Sdf_CleanPrompt["clusterChargeBalance"]<0.4].reset_index(drop=True)
        ##Sdf_CleanPrompt_latewindow = Sdf_CleanPrompt.loc[(Sdf_CleanPrompt['clusterTime']>12000)].reset_index(drop=True)
        #Sdf_trig_CleanPrompt = es.NoPromptClusters_WholeFile(Sdf_SinglePulses,Sdf_trig_SinglePulses,2000)
        #MSData = abp.MakeClusterMultiplicityPlot(Sdf_CleanPrompt,Sdf_trig_CleanPrompt)
        Sdf_SinglePulses = es.SingleSiPMPulses(Sdf)
        Sdf_trig_SinglePulses = Sdf_trig.loc[(Sdf_trig['SiPM1NPulses']==1) & (Sdf_trig['SiPM2NPulses']==1)].reset_index(drop=True)
        Sdf_CleanPrompt = es.NoPromptClusters(Sdf_SinglePulses,SIGNAL_WINDOW_START)
        Sdf_CleanWindow = es.NoBurstClusters(Sdf_CleanPrompt,SIGNAL_WINDOW_START,150)
        Sdf_CleanWindow_noCB = Sdf_CleanWindow.loc[Sdf_CleanWindow["clusterChargeBalance"]<0.4].reset_index(drop=True)
        Sdf_trig_CleanPrompt = es.NoPromptClusters_WholeFile(Sdf_SinglePulses,Sdf_trig_SinglePulses,2000)
        Sdf_trig_CleanWindow = es.NoBurst_WholeFile(Sdf_CleanPrompt,Sdf_trig_CleanPrompt,2000,150)
        MSData = abp.MakeClusterMultiplicityPlot(Sdf_CleanWindow_noCB,Sdf_trig_CleanWindow)
        #MSData = abp.MakeClusterMultiplicityPlot(Sdf_CleanPrompt_latewindow,Sdf_trig_CleanPrompt) #Get multiplicity of late window
        Sbins,Bbin_edges = np.histogram(MSData,range=(0,4),bins=4)
        print("BINS AND EDGES")
        Sbins_lefts = Bbin_edges[0:len(Bbin_edges)-1] #Combine clusters of 19 and 20 at end... negligible effect
        Sbins_normed = Sbins/float(np.sum(Sbins))
        Sbins_normed_unc = np.sqrt(Sbins)/float(np.sum(Sbins))
        zero_bins = np.where(Sbins_normed_unc==0)[0]
        Sbins_normed_unc[zero_bins] = 1/float(np.sum(Sbins)) #FIXME: Not quite right...
        #Cool, now the fun: We build a likelihood profile.
        PLBuilder = plb.ProfileLikelihoodBuilder()
        BkgScaleFactor = (67000-SIGNAL_WINDOW_START)/(67000-BKG_WINDOW_START)  #Scale mean neutrons per window up by this
        PLBuilder.SetBkgMean(BkgScaleFactor*popt[0])
        PLBuilder.SetBkgMeanUnc(np.sqrt(pcov[0][0]))
        NeutronProbProfile = np.arange(0.1,0.8,0.005)
        ChiSquare,lowestChiSqProfile = PLBuilder.BuildLikelihoodProfile(NeutronProbProfile,Sbins_normed,Sbins_normed_unc,700000,Bbins_normed)
        print("MINIMUM CHI SQUARE: " + str(np.min(ChiSquare)))
        ChiSquare_normed = ChiSquare/np.min(ChiSquare)
        plt.plot(NeutronProbProfile,ChiSquare_normed,marker='None',linewidth=5,label=Position,alpha=0.75)
    plt.title("Profile likelihood of of neutron capture efficiency \n in range [2,67] $\mu s$")
    plt.xlabel("Neutron capture efficiency")
    plt.ylabel("$\chi^{2}$/$\chi^{2}_{min}$")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()


if __name__=='__main__':
    blist = glob.glob(BKG_DIR+"*.ntuple.root")

    livetime_estimate = es.EstimateLivetime(blist)
    print("BKG LIVETIME ESTIMATE IN SECONDS IS: " + str(livetime_estimate))

    mybranches = ['eventNumber','eventTimeTank','clusterTime','SiPMhitQ','SiPMNum','SiPMhitT','hitT','hitQ','hitPE','SiPMhitAmplitude','clusterChargeBalance','clusterPE','clusterMaxPE','SiPM1NPulses','SiPM2NPulses','clusterChargePointY','clusterChargePointZ']

    PositionDict = {}
    for j,direc in enumerate(SIGNAL_DIRS):
        direcfiles = glob.glob(direc+"*.ntuple.root")

        livetime_estimate = es.EstimateLivetime(direcfiles)
        print("SIGNAL LIVETIME ESTIMATE IN SECONDS IS: " + str(livetime_estimate))
        PositionDict[SIGNAL_LABELS[j]] = []
        PositionDict[SIGNAL_LABELS[j]].append(GetDataFrame("phaseIITankClusterTree",mybranches,direcfiles))
        PositionDict[SIGNAL_LABELS[j]].append(GetDataFrame("phaseIITriggerTree",mybranches,direcfiles))

    Bdf = GetDataFrame("phaseIITankClusterTree",mybranches,blist)
    Bdf_trig = GetDataFrame("phaseIITriggerTree",mybranches,blist)

    EstimateNeutronEfficiency(PositionDict["Position 0"][0],Bdf,PositionDict["Position 0"][1],Bdf_trig)
    #EstimateNeutronEfficiencyAllPosns(PositionDict,Bdf,Bdf_trig)



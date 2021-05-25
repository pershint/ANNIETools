#This script has functions needed to make a Cumulative Distribution Plot from
#Different variables output in  PhaseIITreeMaker root file.

import glob
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
import uproot
import lib.ROOTProcessor as rp
import lib.EventSelection as es
import lib.ProfileLikelihoodBuilder as plb
import lib.AmBePlots as abp
import scipy.optimize as scp
import numpy as np
import scipy.special as scm

SIGNAL_DIR = "./Data/V1/CentralData/"
BKG_DIR = "./Data/V1/BkgCentralData/"
POSITION_TO_ANALYZE = "Position 0"

#POS0, NO UNCORR BKG: 64% Nu eff, 25$ gamma eff, 3200 Hz rate
#POS1, NO UNCORR BKG: 57% Nu eff, 20$ gamma eff, 3200 Hz rate
#POS2, NO UNCORR BKG: 46% Nu eff, 20$ gamma eff, 3600 Hz rate
#POS3, NO UNCORR BKG: 36% neu eff, 0% gamma eff, 5600 Hz rate
#NRateRanges = {'Position 0': np.arange(3000,3800,200), 'Position 1': np.arange(3000,3600,200),
#        'Position 2': np.arange(3600,4400,200), 'Position 3': np.arange(5400,6200,200)}

#Some good starting efficiency ranges; these ranges were seen in the first analysis
NEffRanges = {'Position 0': np.arange(0.60,0.68,0.01), 'Position 1': np.arange(0.52,0.61,0.01),
        'Position 2': np.arange(0.42, 0.50, 0.01), 'Position 3': np.arange(0.32,0.40, 0.01)}
GEffRanges = {'Position 0': np.arange(0.4,0.9,0.05), 'Position 1': np.arange(0.3,0.8,0.05),
        'Position 2': np.arange(0.25,0.75,0.05), 'Position 3': np.arange(0.2,0.7,0.05)}
BkgUncorrRanges = {'Position 0': np.arange(0.055,0.0725,0.0025), 'Position 1': np.arange(0.055,0.085,0.005),
        'Position 2': np.arange(0.055,0.085,0.005), 'Position 3': np.arange(0.055,0.085,0.005)}
BkgRanges = {'Position 0': np.arange(0.03,0.070,0.0025), 'Position 1': np.arange(0.04,0.080,0.005),
        'Position 2': np.arange(0.04, 0.080,0.005), 'Position 3': np.arange(0.04,0.080,0.005)}


expoPFlat= lambda x,C1,tau,mu,B: C1*np.exp(-(x-mu)/tau) + B
mypoisson = lambda x,mu: (mu**x)*np.exp(-mu)/scm.factorial(x)
mypoissons = lambda x,R1,mu1,R2,mu2: R1*(mu1**x)*np.exp(-mu2)/scm.factorial(x) + R2*(mu2**x)*np.exp(-mu2)/scm.factorial(x)

BKG_WINDOW_START = 2000
SIGNAL_WINDOW_START = 2000
NUMTHROWS = 1E6

def GetDataFrame(mytreename,mybranches,filelist):
    RProcessor = rp.ROOTProcessor(treename=mytreename)
    for f1 in filelist:
        RProcessor.addROOTFile(f1,branches_to_get=mybranches)
    data = RProcessor.getProcessedData()
    df = pd.DataFrame(data)
    return df

def EstimateNeutronEfficiency(Sdf, Sdf_trig):

    #Apply event selection cuts to source data
    Sdf_SinglePulses = es.SingleSiPMPulses(Sdf)
    Sdf_CleanPrompt = es.NoPromptClusters(Sdf_SinglePulses,SIGNAL_WINDOW_START)
    Sdf_CleanWindow = es.NoBurstClusters(Sdf_CleanPrompt,SIGNAL_WINDOW_START,150)
    Sdf_CleanWindow_noCB = Sdf_CleanWindow.loc[Sdf_CleanWindow["clusterChargeBalance"]<0.4].reset_index(drop=True)
    Sdf_trig_SinglePulses = Sdf_trig.loc[(Sdf_trig['SiPM1NPulses']==1) & (Sdf_trig['SiPM2NPulses']==1)].reset_index(drop=True)
    Sdf_trig_CleanPrompt = es.NoPromptClusters_WholeFile(Sdf_SinglePulses,Sdf_trig_SinglePulses,2000)
    Sdf_trig_CleanWindow = es.NoBurst_WholeFile(Sdf_CleanPrompt,Sdf_trig_CleanPrompt,2000,150)
    MSData = abp.MakeClusterMultiplicityPlot(Sdf_CleanWindow_noCB,Sdf_trig_CleanWindow)
    plt.hist(MSData,bins=20, range=(0,20), alpha=0.2,histtype='stepfilled',linewidth=6)
    plt.hist(MSData,bins=20, range=(0,20), histtype='step',linewidth=6)
    plt.xlabel("Neutron candidate multiplicity")
    plt.ylabel("Number of acquisitions")
    plt.title("Neutron candidate multiplicity, AmBe central source run \n (All preliminary cuts, [2,67] $\mu s$ window)")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    Sbins,Sbin_edges = np.histogram(MSData,range=(0,5),bins=5)
    print("SIGNAL BINS AND EDGES")
    print(Sbins)
    print(Sbin_edges)
    Sbins_lefts = Sbin_edges[0:len(Sbin_edges)-1] #Combine clusters of 19 and 20 at end... negligible effect
    Sbins_normed = Sbins/float(np.sum(Sbins))
    Sbins_normed_unc = np.sqrt(Sbins)/float(np.sum(Sbins))
    zero_bins = np.where(Sbins_normed_unc==0)[0]
    Sbins_normed_unc[zero_bins] = 1.15/float(np.sum(Sbins)) 


    plt.errorbar(x=Sbins_lefts,y=Sbins_normed,yerr=Sbins_normed_unc,linestyle='None',marker='o',label='Source ($t>2 \, \mu s$)',markersize=12)
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    #Plots for uncorrelated bkg. fit
    PLBuilder2D = plb.ProfileLikelihoodBuilder2D()
    neutron_efficiencies = NEffRanges[POSITION_TO_ANALYZE]
    background_mean = BkgUncorrRanges[POSITION_TO_ANALYZE]
    PLBuilder2D.SetEffProfile(neutron_efficiencies)
    PLBuilder2D.SetBkgMeanProfile(background_mean)
    x_var, y_var, ChiSquare,lowestChiSqProfileUncorr = PLBuilder2D.BuildLikelihoodProfile(Sbins_normed,Sbins_normed_unc,NUMTHROWS)
    print("MINIMUM CHI SQUARE: " + str(np.min(ChiSquare)))
    bestfit_index = np.where(ChiSquare==np.min(ChiSquare))[0]
    print("BEST FIT UNCORR BKG: " + str(y_var[bestfit_index]))
    print("BEST FIT NU EFF: " + str(x_var[bestfit_index]))
    plt.errorbar(x=Sbins_lefts,y=Sbins_normed,yerr=Sbins_normed_unc,linestyle='None',marker='o',label='Source ($t>2 \, \mu s$)',markersize=12)
    plt.plot(Sbins_lefts, lowestChiSqProfileUncorr,linestyle='None',marker='o',label='Best fit model profile',markersize=12)
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Best fit MC profile relative to central source data")
    plt.xlabel("Delayed cluster multiplicity")
    plt.show()

    #Find best-fit assuming uncorrelated background model
    PLBuilder3D = plb.ProfileLikelihoodBuilder3D()
    neutron_efficiencies = NEffRanges[POSITION_TO_ANALYZE]
    gamma_efficiencies = GEffRanges[POSITION_TO_ANALYZE]
    PLBuilder3D.SetNeuEffProfile(neutron_efficiencies)
    PLBuilder3D.SetGammaEffProfile(gamma_efficiencies)
    #background_mean = np.array([0.060])
    #PLBuilder3D.SetBkgMeanProfile(background_mean)
    PLBuilder3D.SetSourceRate(200.0)
    PLBuilder3D.SetBkgMeanProfile(BkgRanges[POSITION_TO_ANALYZE])
    x_var, y_var, z_var, ChiSquare,lowestChiSqProfile = PLBuilder3D.BuildLikelihoodProfile(Sbins_normed,Sbins_normed_unc,NUMTHROWS)
    print("MINIMUM CHI SQUARE: " + str(np.min(ChiSquare)))
    min_index = np.where(ChiSquare==np.min(ChiSquare))[0]
    print("BEST FIT BKG RATE:: " + str(z_var[min_index]))
    plt.errorbar(x=Sbins_lefts,y=Sbins_normed,yerr=Sbins_normed_unc,linestyle='None',marker='o',label='Source ($t>2 \, \mu s$)',markersize=12)
    plt.plot(Sbins_lefts, lowestChiSqProfile,linestyle='None',marker='o',label='Best fit model profile',markersize=12)
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Best fit MC profile relative to central source data")
    plt.xlabel("Delayed cluster multiplicity")
    plt.show()

    #Look at 2D chi-squared map
    best_bkg_indices = np.where(z_var==z_var[min_index])[0]
    x_var = x_var[best_bkg_indices]
    y_var = y_var[best_bkg_indices]
    ChiSquare = ChiSquare[best_bkg_indices]
    chisq_map = pd.DataFrame({"Neutron detection efficiency":np.round(x_var,3), "Gamma detection efficiency":np.round(y_var,3),"ChiSq":ChiSquare/np.min(ChiSquare)})
    cmap = chisq_map.pivot(index="Neutron detection efficiency",columns="Gamma detection efficiency",values="ChiSq")
    ax = sns.heatmap(cmap,vmin=1,vmax=10)
    plt.title("$\chi^{2}$/$\chi_{min}^{2}$ for profile likelihood parameters")
    plt.show()

    chisq_map = pd.DataFrame({"Neutron detection efficiency":np.round(x_var,3), "Gamma detection efficiency":np.round(y_var,3),"ChiSq":ChiSquare - np.min(ChiSquare)})
    cmap = chisq_map.pivot(index="Neutron detection efficiency",columns="Gamma detection efficiency",values="ChiSq")
    ax = sns.heatmap(cmap,vmin=0,vmax=80)
    plt.title("$\chi^{2} - \chi_{min}^{2}$ for profile likelihood parameters")
    plt.show()

    #chisq_map = pd.DataFrame({"Neutron detection efficiency":np.round(x_var,3), "Uncorr. Bkg. Rate":np.round(z_var,3),"ChiSq":ChiSquare/np.min(ChiSquare)})
    #cmap = chisq_map.pivot(index="Neutron detection efficiency",columns="Uncorr. Bkg. Rate",values="ChiSq")
    #ax = sns.heatmap(cmap,vmin=1,vmax=10)
    #plt.title("$\chi^{2}$/$\chi_{min}^{2}$ for profile likelihood parameters")
    #plt.show()

    #chisq_map = pd.DataFrame({"Neutron detection efficiency":np.round(x_var,3), "Uncorr. Bkg. Rate":np.round(z_var,3),"ChiSq":ChiSquare - np.min(ChiSquare)})
    #cmap = chisq_map.pivot(index="Neutron detection efficiency",columns="Uncorr. Bkg. Rate",values="ChiSq")
    #ax = sns.heatmap(cmap,vmin=0,vmax=80)
    #plt.title("$\chi^{2} - \chi_{min}^{2}$ for profile likelihood parameters")
    #plt.show()
    
    LowestInd = np.where(ChiSquare==np.min(ChiSquare))[0]
    best_eff = x_var[LowestInd]
    best_mean = y_var[LowestInd]
    best_eff_chisquareinds = np.where(x_var==best_eff)[0]
    best_eff_chisquares = ChiSquare[best_eff_chisquareinds]
    best_gammaeff = y_var[best_eff_chisquareinds]
    plt.plot(best_gammaeff,best_eff_chisquares/np.min(ChiSquare),marker='None',linewidth=6, label = 'Uncorr. bkg. model',color='blue')
    plt.title("Normalized Chi-square test parameter as $\epsilon_{\gamma}$ varies \n (best-fit detection efficiency $\epsilon_{n}$ fixed)")
    plt.xlabel("Gamma detection efficiency $\epsilon_{\gamma}$")
    plt.ylabel("$\chi^{2}$/$\chi^{2}_{min}$")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()
    plt.plot(best_gammaeff,best_eff_chisquares-np.min(ChiSquare),marker='None',linewidth=6, label = 'Uncorr. bkg. model',color='blue')
    plt.title("Normalized Chi-square test parameter as $\epsilon_{\gamma}$ varies \n (best-fit detection efficiency $\epsilon_{n}$ fixed)")
    plt.xlabel("Gamma detection efficiency $\epsilon_{\gamma}$")
    plt.ylabel("$\chi^{2} - \chi^{2}_{min}$")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()


    best_mean_chisquareinds = np.where(y_var==best_mean)[0]
    best_mean_chisquares = ChiSquare[best_mean_chisquareinds]
    best_mean_efficiencypro = x_var[best_mean_chisquareinds]
    plt.plot(best_mean_efficiencypro,best_mean_chisquares/np.min(ChiSquare),marker='None',linewidth=6, label = 'Uncorr. bkg. model',color='blue')
    plt.title("Normalized Chi-square test parameter as $\epsilon_{n}$ varies \n (best-fit Gamma detection efficiency $\epsilon_{\gamma}$ fixed)")
    plt.xlabel("Neutron detection efficiency $\epsilon_{n}$")
    plt.ylabel("$\chi^{2}$/$\chi^{2}_{min}$")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    plt.plot(best_mean_efficiencypro,best_mean_chisquares - np.min(ChiSquare),marker='None',linewidth=6, label = 'Uncorr. bkg. model',color='blue')
    plt.title("Normalized Chi-square test parameter as $\epsilon_{n}$ varies \n (best-fit Gamma detection efficiency $\epsilon_{\gamma}$ fixed)")
    plt.xlabel("Neutron detection efficiency $\epsilon_{n}$")
    plt.ylabel("$\chi^{2} - \chi^{2}_{min}$")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()


    #Compare the two model fits to the signal data
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    bin_rights = Sbins_lefts + (Sbins_lefts[1] - Sbins_lefts[0])
    ax = abp.NiceBins(ax,Sbins_lefts,bin_rights,lowestChiSqProfile,'dark blue',"V2. bkg. best fit")
    ax = abp.NiceBins(ax,Sbins_lefts,bin_rights,lowestChiSqProfileUncorr,'dark red',"Uncorr bkg. best fit")
    ax.errorbar(x=Sbins_lefts+0.5,y=Sbins_normed,yerr=Sbins_normed_unc,linestyle='None',marker='o',label='AmBe data',markersize=12,color='black')
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Best fit multiplicity distributions to central source data")
    plt.xlabel("Neutron candidate multiplicity")
    plt.show()


if __name__=='__main__':
    slist = glob.glob(SIGNAL_DIR+"*.ntuple.root")
    blist = glob.glob(BKG_DIR+"*.ntuple.root")

    livetime_estimate = es.EstimateLivetime(blist)
    print("BKG LIVETIME ESTIMATE IN SECONDS IS: " + str(livetime_estimate))
    livetime_estimate = es.EstimateLivetime(slist)
    print("SIGNAL LIVETIME ESTIMATE IN SECONDS IS: " + str(livetime_estimate))
    
    mybranches = ['eventNumber','eventTimeTank','clusterTime','clusterChargeBalance','SiPM1NPulses','SiPM2NPulses','clusterPE']
    Sdf = GetDataFrame("phaseIITankClusterTree",mybranches,slist)
    Sdf_trig = GetDataFrame("phaseIITriggerTree",mybranches,slist)

    EstimateNeutronEfficiency(Sdf,Sdf_trig)



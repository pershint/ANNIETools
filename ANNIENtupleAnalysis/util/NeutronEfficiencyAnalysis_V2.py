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

SIGNAL_DIR = "./Data/V3_5PE100ns/Pos0Data/"
BKG_DIR = "./Data/V3_5PE100ns/BkgPos0Data/"
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
GEffRanges = {'Position 0': np.arange(0.5,1.0,0.05), 'Position 1': np.arange(0.3,0.8,0.05),
        'Position 2': np.arange(0.25,0.75,0.05), 'Position 3': np.arange(0.2,0.7,0.05)}
BkgMeanRanges = {'Position 0': np.arange(0.045,0.085,0.0025), 'Position 1': np.arange(0.04,0.080,0.005),
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
    Sdf_ChargeCut = es.SiPMChargeCut(Sdf_CleanWindow_noCB,0.1,0.5)
    Sdf_trig_SinglePulses = Sdf_trig.loc[(Sdf_trig['SiPM1NPulses']==1) & (Sdf_trig['SiPM2NPulses']==1)].reset_index(drop=True)
    Sdf_trig_CleanPrompt = es.NoPromptClusters_WholeFile(Sdf_SinglePulses,Sdf_trig_SinglePulses,2000)
    Sdf_trig_CleanWindow = es.NoBurst_WholeFile(Sdf_CleanPrompt,Sdf_trig_CleanPrompt,2000,150)
    Sdf_trig_ChargeCut = es.SiPMChargeCut(Sdf_trig_CleanWindow,0.1,0.5)
    MSData = abp.MakeClusterMultiplicityPlot(Sdf_ChargeCut,Sdf_trig_ChargeCut)
    #plt.hist(MSData,bins=20, range=(0,20), alpha=0.2,histtype='stepfilled',linewidth=6)
    #plt.hist(MSData,bins=20, range=(0,20), histtype='step',linewidth=6)
    #plt.xlabel("Neutron candidate multiplicity")
    #plt.ylabel("Number of acquisitions")
    #plt.title("Neutron candidate multiplicity, AmBe central source run \n (All preliminary cuts, [2,67] $\mu s$ window)")
    #leg = plt.legend(loc=1,fontsize=24)
    #leg.set_frame_on(True)
    #leg.draw_frame(True)
    #plt.show()

    Sbins,Sbin_edges = np.histogram(MSData,range=(0,5),bins=5)
    print("SIGNAL BINS AND EDGES")
    print(Sbins)
    print(Sbin_edges)
    Sbins_lefts = Sbin_edges[0:len(Sbin_edges)-1] #Combine clusters of 19 and 20 at end... negligible effect
    Sbins_normed = Sbins/float(np.sum(Sbins))
    Sbins_normed_unc = np.sqrt(Sbins)/float(np.sum(Sbins))
    zero_bins = np.where(Sbins_normed_unc==0)[0]
    Sbins_normed_unc[zero_bins] = 1.15/float(np.sum(Sbins)) 


    #plt.errorbar(x=Sbins_lefts,y=Sbins_normed,yerr=Sbins_normed_unc,linestyle='None',marker='o',label='Source ($t>2 \, \mu s$)',markersize=12)
    #leg = plt.legend(loc=1,fontsize=24)
    #leg.set_frame_on(True)
    #leg.draw_frame(True)
    #plt.show()

    #Find best-fit assuming uncorrelated background model
    PLBuilder3D = plb.ProfileLikelihoodBuilder3D()
    neutron_efficiencies = NEffRanges[POSITION_TO_ANALYZE]
    gamma_efficiencies = GEffRanges[POSITION_TO_ANALYZE]
    PLBuilder3D.SetNeuEffProfile(neutron_efficiencies)
    PLBuilder3D.SetGammaEffProfile(gamma_efficiencies)
    #background_mean = np.array([0.060])
    #PLBuilder3D.SetBkgMeanProfile(background_mean)
    PLBuilder3D.SetSourceRate(200.0)
    PLBuilder3D.SetBkgMeanProfile(BkgMeanRanges[POSITION_TO_ANALYZE])
    #PROFILES AND ALL THEIR ASSOCIATED CHI-SQUARED VALUES HERE
    x_var, y_var, z_var, ChiSquare,lowestChiSqProfile = PLBuilder3D.BuildLikelihoodProfile(Sbins_normed,Sbins_normed_unc,NUMTHROWS)


    print("MINIMUM CHI SQUARE: " + str(np.min(ChiSquare)))
    min_index = np.where(ChiSquare==np.min(ChiSquare))[0]
    print("BEST FIT NEUTRON EFFICIENCY: " + str(x_var[min_index]))
    print("BEST FIT GAMMA EFFICIENCY: " + str(y_var[min_index]))
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
    x_var_hm1 = x_var[best_bkg_indices]
    y_var_hm1 = y_var[best_bkg_indices]
    ChiSquare_hm1 = ChiSquare[best_bkg_indices]
    chisq_map = pd.DataFrame({"Neutron detection efficiency":np.round(x_var_hm1,3), "Gamma detection efficiency":np.round(y_var_hm1,3),"ChiSq":ChiSquare_hm1 - np.min(ChiSquare_hm1)})
    cmap = chisq_map.pivot(index="Neutron detection efficiency",columns="Gamma detection efficiency",values="ChiSq")
    ax = sns.heatmap(cmap,vmin=0,vmax=80)
    plt.title("$\chi^{2} - \chi_{min}^{2}$ for profile likelihood parameters")
    plt.show()

    best_bkg_indices = np.where(y_var==y_var[min_index])[0]
    x_var_hm2 = x_var[best_bkg_indices]
    z_var_hm2 = z_var[best_bkg_indices]
    ChiSquare_hm2 = ChiSquare[best_bkg_indices]
    chisq_map = pd.DataFrame({"Neutron detection efficiency":np.round(x_var_hm2,3), "Uncorr. Bkg. Rate":np.round(z_var_hm2,3),"ChiSq":ChiSquare_hm2 - np.min(ChiSquare_hm2)})
    cmap = chisq_map.pivot(index="Neutron detection efficiency",columns="Uncorr. Bkg. Rate",values="ChiSq")
    ax = sns.heatmap(cmap,vmin=0,vmax=80)
    plt.title("$\chi^{2} - \chi_{min}^{2}$ for profile likelihood parameters")
    plt.show()
   
    #Show the gamma detection efficiency profile
    LowestInd = np.where(ChiSquare==np.min(ChiSquare))[0]
    best_eff = x_var[LowestInd]
    best_uncorrbkg_rate = z_var[LowestInd]
    best_eff_chisquareinds = np.where(x_var==best_eff)[0]
    best_ucorrbkg_chisquareinds = np.where(z_var==best_uncorrbkg_rate)[0]
    best_chisquareinds = np.intersect1d(best_eff_chisquareinds,best_ucorrbkg_chisquareinds)
    best_eff_chisquares = ChiSquare[best_chisquareinds]
    best_gammaeff = y_var[best_chisquareinds]
    plt.plot(best_gammaeff,best_eff_chisquares/np.min(ChiSquare),marker='None',linewidth=6, label = 'V2 model',color='blue')
    plt.title("Normalized Chi-square test parameter as $\epsilon_{\gamma}$ varies \n (best-fit $\epsilon_{n},\lambda_{n}$ fixed)")
    plt.xlabel("Gamma detection efficiency $\epsilon_{\gamma}$")
    plt.ylabel("$\chi^{2}$/$\chi^{2}_{min}$")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()
    plt.plot(best_gammaeff,best_eff_chisquares-np.min(ChiSquare),marker='None',linewidth=6, label = 'V2 model',color='blue')
    plt.title("Normalized Chi-square test parameter as $\epsilon_{\gamma}$ varies \n (best-fit $\epsilon_{n},\lambda_{n}$ fixed)")
    plt.xlabel("Gamma detection efficiency $\epsilon_{\gamma}$")
    plt.ylabel("$\chi^{2} - \chi^{2}_{min}$")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    LowestInd = np.where(ChiSquare==np.min(ChiSquare))[0]
    best_eff = x_var[LowestInd]
    best_geff = y_var[LowestInd]
    best_eff_chisquareinds = np.where(x_var==best_eff)[0]
    best_geff_chisquareinds = np.where(y_var==best_geff)[0]
    best_chisquareinds = np.intersect1d(best_eff_chisquareinds,best_geff_chisquareinds)
    best_uncorrbkg_chisquares = ChiSquare[best_chisquareinds]
    best_uncorrbkg = z_var[best_chisquareinds]

    plt.plot(best_uncorrbkg,best_uncorrbkg_chisquares/np.min(ChiSquare),marker='None',linewidth=6, label = 'V2 model',color='blue')
    plt.title("Normalized Chi-square test parameter as $\lambda_{b}$ varies \n (best-fit $\epsilon_{n},\epsilon_{\gamma}$ fixed)")
    plt.xlabel("Uncorrelated background rate $\lambda_{b}$")
    plt.ylabel("$\chi^{2}$/$\chi^{2}_{min}$")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()
    plt.plot(best_uncorrbkg,best_uncorrbkg_chisquares-np.min(ChiSquare),marker='None',linewidth=6, label = 'V2 model',color='blue')
    plt.title("Normalized Chi-square test parameter as $\lambda_{b}$ varies \n (best-fit $\epsilon_{n},\epsilon_{gamma}$ fixed)")
    plt.xlabel("Uncorrelated background rate $\lambda_{b}$")
    plt.ylabel("$\chi^{2} - \chi^{2}_{min}$")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()


    LowestInd = np.where(ChiSquare==np.min(ChiSquare))[0]
    best_geff = y_var[LowestInd]
    best_uncorrbkg_rate = z_var[LowestInd]
    best_geff_chisquareinds = np.where(y_var==best_geff)[0]
    best_ucorrbkg_chisquareinds = np.where(z_var==best_uncorrbkg_rate)[0]
    best_chisquareinds = np.intersect1d(best_geff_chisquareinds,best_ucorrbkg_chisquareinds)
    best_eff_chisquares = ChiSquare[best_chisquareinds]
    best_eff = x_var[best_chisquareinds]

    plt.plot(best_eff,best_eff_chisquares/np.min(ChiSquare),marker='None',linewidth=6, label = 'V2 Fit',color='blue')
    plt.title("Normalized Chi-square test parameter as $\epsilon_{n}$ varies \n (best-fit Gamma detection efficiency $\epsilon_{\gamma}$ fixed)")
    plt.xlabel("Neutron detection efficiency $\epsilon_{n}$")
    plt.ylabel("$\chi^{2}$/$\chi^{2}_{min}$")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    plt.plot(best_eff,best_eff_chisquares - np.min(ChiSquare),marker='None',linewidth=6, label = 'V2 Fit',color='blue')
    plt.title("Normalized Chi-square test parameter as $\epsilon_{n}$ varies \n (best-fit $\epsilon_{\gamma},\lambda_{b}$ fixed)")
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
    ax.errorbar(x=Sbins_lefts+0.5,y=Sbins_normed,yerr=Sbins_normed_unc,linestyle='None',marker='o',label='AmBe data',markersize=12,color='black')
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Best fit multiplicity distribution to central source data")
    plt.xlabel("Neutron candidate multiplicity")
    plt.show()


if __name__=='__main__':
    slist = glob.glob(SIGNAL_DIR+"*.ntuple.root")
    blist = glob.glob(BKG_DIR+"*.ntuple.root")

    livetime_estimate = es.EstimateLivetime(blist)
    print("BKG LIVETIME ESTIMATE IN SECONDS IS: " + str(livetime_estimate))
    livetime_estimate = es.EstimateLivetime(slist)
    print("SIGNAL LIVETIME ESTIMATE IN SECONDS IS: " + str(livetime_estimate))
    
    mybranches = ['eventNumber','eventTimeTank','clusterTime','clusterChargeBalance','SiPM1NPulses','SiPM2NPulses','clusterPE','SiPMhitQ']
    Sdf = GetDataFrame("phaseIITankClusterTree",mybranches,slist)
    Sdf_trig = GetDataFrame("phaseIITriggerTree",mybranches,slist)
    #There's some issue in the trigger tree processing... pulses were double-counted
    #Fix this by removing duplicates (probability of two SiPM pulses actually having
    #The exact same charge is zero, so this correction shouldn't affect actual pulses)
    for i in Sdf_trig.index.values:
        Sdf_trig['SiPMhitQ'][i] = list(set(Sdf_trig['SiPMhitQ'][i]))

    EstimateNeutronEfficiency(Sdf,Sdf_trig)



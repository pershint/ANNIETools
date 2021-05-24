#This script has functions needed to make a Cumulative Distribution Plot from
#Different variables output in  PhaseIITreeMaker root file.

import glob

import sys
import uproot
import lib.ROOTProcessor as rp
import lib.EventSelection as es
import lib.AmBePlots as abp
import lib.FileReader as fr
import lib.HistogramUtils as hu

import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as scp
import numpy as np

MCDELTAT = "./Data/V3_5PE100ns/MCProfiles/DeltaTHistAmBe.csv"
MCPE = "./Data/V3_5PE100ns/MCProfiles/PEHistAmBe.csv"

SIGNAL_DIR = "./Data/V3_5PE100ns/Pos0Data/"
#BKG_DIR = "./Data/BkgCentralData/"
BKG_DIR = "./Data/V3_5PE100ns/BkgPos0Data/"

expoPFlat= lambda x,C1,tau,mu,B: C1*np.exp(-(x-mu)/tau) + B

def EventSelectionLosses(df,df_trig):
    print("TOTAL NUMBER OF EVENT TIME TANKS SET: " + str(len(set(df_trig['eventTimeTank']))))
    print("TOTAL NUMBER OF EVENT TIME TANKS, LIST: " + str(len(df_trig['eventTimeTank'])))
    print("TOTAL NUMBER OF ENTRIES: " + str(len(df_trig)))
    
    df_cleanSiPM = es.SingleSiPMPulses(df_trig)
    print("TOTAL NUMBER OF EVENTS W/ ONE PULSE IN EACH SIPM: " + str(len(set(df_cleanSiPM['eventTimeTank']))))
    df_SinglePulses = es.SingleSiPMPulses(df) #Clusters with a single SiPM pulse

    df_cleanSiPMDT = es.SingleSiPMPulsesDeltaT(df_trig,200) 
    print("TOTAL NUMBER OF EVENTS W/ ONE PULSE IN EACH SIPM, PEAKS WITHIN 200 NS: " + str(len(set(df_cleanSiPMDT['eventTimeTank']))))
    
    df_trig_cleanSiPM = df_trig.loc[(df_trig['SiPM1NPulses']==1) & (df_trig['SiPM2NPulses']==1)].reset_index(drop=True)
    df_cleanPrompt = es.NoPromptClusters(df_SinglePulses,2000)
    df_trig_cleanPrompt = es.NoPromptClusters_WholeFile(df_SinglePulses,df_trig_cleanSiPM,2000)
    print("TOTAL NUMBER OF EVENTS W/ ONE PULSE IN EACH SIPM AND NO PROMPT CLUSTER: " + str(len(set(df_trig_cleanPrompt['eventTimeTank']))))

    #Late burst cut
    df_trig_cleanWindow = es.NoBurst_WholeFile(df_cleanPrompt,df_trig_cleanPrompt,2000,150)
    print("TOTAL NUMBER OF EVENTS W/ CLEAN PROMPT AND NO BURST ABOVE 150 PE AND 2 MICROSECONDS: " + str(len(set(df_trig_cleanWindow['eventTimeTank']))))

def PlotDemo(Sdf,Bdf,Sdf_trig,Bdf_trig):
    print("EVENT SELECTION LOSSES FOR CENTRAL SOURCE RUN")
    EventSelectionLosses(Sdf,Sdf_trig)

    print("EVENT SELECTION LOSSES FOR BKG CENTRAL SOURCE RUN")
    EventSelectionLosses(Bdf,Bdf_trig)

    Sdf_SinglePulses = es.SingleSiPMPulses(Sdf)
    Sdf_CleanPrompt = es.NoPromptClusters(Sdf_SinglePulses,2000)
    Sdf_CleanWindow = es.NoBurstClusters(Sdf_CleanPrompt,2000,150)
    Sdf_CleanWindow_CBClean = Sdf_CleanWindow.loc[Sdf_CleanWindow['clusterChargeBalance']<0.4]
    Sdf_trig_goodSiPM = Sdf_trig.loc[(Sdf_trig['SiPM1NPulses']==1) & (Sdf_trig['SiPM2NPulses']==1)].reset_index(drop=True)
    Sdf_trig_CleanPrompt = es.NoPromptClusters_WholeFile(Sdf_SinglePulses,Sdf_trig_goodSiPM,2000)
    Sdf_trig_CleanWindow = es.NoBurst_WholeFile(Sdf_CleanPrompt,Sdf_trig_CleanPrompt,2000,150)

    #Line of cuts applied to background clusters
    Bdf_SinglePulses = es.SingleSiPMPulses(Bdf)
    Bdf_CleanPrompt = es.NoPromptClusters(Bdf_SinglePulses,2000)
    Bdf_CleanWindow = es.NoBurstClusters(Bdf_CleanPrompt,2000,150)
    Bdf_latewindow = Bdf_CleanWindow.loc[(Bdf_CleanWindow['clusterTime']>2000)].reset_index(drop=True)
    Bdf_latewindow = Bdf_latewindow.loc[(Bdf_latewindow['clusterChargeBalance']<0.4)].reset_index(drop=True)
    Bdf_trig_cleanSiPM = Bdf_trig.loc[(Bdf_trig['SiPM1NPulses']==1) & (Bdf_trig['SiPM2NPulses']==1)].reset_index(drop=True)
    Bdf_trig_cleanPrompt = es.NoPromptClusters_WholeFile(Bdf_SinglePulses,Bdf_trig_cleanSiPM,2000)
    Bdf_trig_BurstCut = es.NoBurst_WholeFile(Bdf_CleanPrompt,Bdf_trig_cleanPrompt,2000,150)

    #Special case
    Bdf_NoBurstLateWindow = es.NoBurstClusters(Bdf_SinglePulses,12000,150)
    Bdf_NBlatewindow = Bdf_NoBurstLateWindow.loc[(Bdf_NoBurstLateWindow['clusterTime']>12000)].reset_index(drop=True)
    Bdf_NBCBCut = Bdf_NBlatewindow.loc[Bdf_NBlatewindow['clusterChargeBalance']<0.4].reset_index(drop=True)


    #Get prompt clusters within a 100 ns window of the mean SiPM single pulse time
    Sdf_SPPrompt = es.SiPMCorrelatedPrompts(Sdf_SinglePulses,100,1000,2000)
    print("NUMBER OF PROMPT CLUSTERS PASSING SINGLE SIPM PULSE CUTS: " + str(len(Sdf_SinglePulses.eventTimeTank)))
    Sdf_SPPrompt = Sdf_SPPrompt.loc[Sdf_SPPrompt['clusterChargeBalance']<0.4].reset_index(drop=True)
    Sdf_SPPrompt_trig = es.SiPMCorrelatedPrompts_WholeFile(Sdf_SPPrompt,Sdf_trig)
    print("NUMBER OF PROMPT CLUSTERS PASSING CORRELATED PROMPT CUTS: " + str(len(Sdf_SPPrompt.eventTimeTank)))
    print("NUMBER OF TRIGS WITH AT LEAST ONE CORRELATED PROMPT: " + str(len(Sdf_SPPrompt_trig.eventTimeTank)))
    MSData = abp.MakeClusterMultiplicityPlot(Sdf_SPPrompt,Sdf_SPPrompt_trig)
    #s_bins, s_edges = np.histogram(MSData,bins=20, range=(0,20))
    #plt.hist(MSData,bins=20, range=(0,20), label="Source",alpha=0.7)
    plt.hist(MSData,bins=100,label="Source",alpha=0.7)
    plt.xlabel("False start candidate cluster multiplicity")
    plt.title("Cluster multiplicity of false starts in source data \n (One pulse each SiPM, cluster within prompt window)")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    S1Delta,S2Delta = abp.SiPMClusterDifferences(Sdf_SPPrompt,100)
    plt.hist(S1Delta,bins=100,label="S1Time-clusterTime",alpha=0.7)
    plt.hist(S2Delta,bins=100,label="S2Time-clusterTime",alpha=0.7)
    plt.xlabel("SiPM time - Cluster time (ns)")
    plt.title("Difference between SiPM peak time and cluster time \n (One pulse each SiPM, cluster within prompt window)")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    Sdf_SPPromptLoPE = Sdf_SPPrompt.loc[Sdf_SPPrompt['clusterPE']<80].reset_index(drop=True)
    print("NUMBER OF PROMPT CLUSTERS PASSING CORRELATED PROMPT CUTS LT 80 PE: " + str(len(Sdf_SPPromptLoPE.eventTimeTank)))
    NumTrigs = len(set(Sdf_SPPromptLoPE.eventNumber))
    print("NUMBER OF TRIGS WITH A PASSING CORRELATED PROMPT CUTS CLUSTER LT 80 PE: " + str(NumTrigs))
    labels = {'title': 'Comparison of cluster PE to total SiPM Charge \n (Position 0, AmBe source installed)', 
            'xlabel': 'Cluster PE', 'ylabel': 'Total SiPM charge [nC]'}
    ranges = {'xbins': 30, 'ybins':40, 'xrange':[0,60],'yrange':[0,0.5],'promptTime':2000}
    #abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    abp.Make2DHist_PEVsQ(Sdf_SPPrompt,labels,ranges)
    plt.show()

    plt.hist(np.hstack(Sdf_SPPrompt['clusterPE']),bins=30,range=(0,80),alpha=0.5,histtype='stepfilled',linewidth=6)
    plt.hist(np.hstack(Sdf_SPPrompt['clusterPE']),bins=30,range=(0,80),alpha=0.75,histtype='step',linewidth=6,color='green')
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("False start candidate PE distribution (AmBe central source data)")
    plt.xlabel("Cluster PE")
    plt.show()

    Sdf_SPPrompt_10N = Sdf_SPPrompt.loc[Sdf_SPPrompt['clusterHits']==10].reset_index(drop=True)
    Sdf_SPPrompt_11N = Sdf_SPPrompt.loc[Sdf_SPPrompt['clusterHits']==11].reset_index(drop=True)
    Sdf_SPPrompt_12N = Sdf_SPPrompt.loc[Sdf_SPPrompt['clusterHits']==12].reset_index(drop=True)
    Sdf_SPPrompt_8N = Sdf_SPPrompt.loc[Sdf_SPPrompt['clusterHits']==8].reset_index(drop=True)
    Sdf_SPPrompt_9N = Sdf_SPPrompt.loc[Sdf_SPPrompt['clusterHits']==9].reset_index(drop=True)
    #plt.hist(np.hstack(Sdf_SPPrompt_10N['clusterPE']),bins=30,range=(0,80),alpha=0.5,histtype='stepfilled',linewidth=6)
    plt.hist(np.hstack(Sdf_SPPrompt_8N['clusterPE']),bins=30,range=(0,80),label='8 hits',alpha=0.75,histtype='step',linewidth=6,color='green')
    plt.hist(np.hstack(Sdf_SPPrompt_9N['clusterPE']),bins=30,range=(0,80),label='9 hits',alpha=0.75,histtype='step',linewidth=6,color='black')
    plt.hist(np.hstack(Sdf_SPPrompt_10N['clusterPE']),bins=30,range=(0,80),label='10 hits',alpha=0.75,histtype='step',linewidth=6,color='blue')
    #plt.hist(np.hstack(Sdf_SPPrompt_11N['clusterPE']),bins=30,range=(0,80),alpha=0.5,histtype='stepfilled',linewidth=6)
    plt.hist(np.hstack(Sdf_SPPrompt_11N['clusterPE']),bins=30,range=(0,80),label='11 hits',alpha=0.75,histtype='step',linewidth=6,color='red')
    #plt.hist(np.hstack(Sdf_SPPrompt_12N['clusterPE']),bins=30,range=(0,80),alpha=0.5,histtype='stepfilled',linewidth=6)
    plt.hist(np.hstack(Sdf_SPPrompt_12N['clusterPE']),bins=30,range=(0,80),label='12 hits',alpha=0.75,histtype='step',linewidth=6,color='purple')
    #plt.hist(np.hstack(Sdf_SPPrompt_8N['clusterPE']),bins=30,range=(0,80),alpha=0.5,histtype='stepfilled',linewidth=6)
    #plt.hist(np.hstack(Sdf_SPPrompt_9N['clusterPE']),bins=30,range=(0,80),alpha=0.5,histtype='stepfilled',linewidth=6)
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Cluster PE distribution for clusters of varying nhit")
    plt.xlabel("Cluster PE")
    plt.show()

    labels = {'title': "Amplitude of SiPM1 hits", 'xlabel':'Pulse amplitude [V]','llabel':'SiPM1'}
    ranges = {'bins': 160, 'range':(0,0.3)}
    abp.MakeSiPMVariableDistribution(Sdf_trig_goodSiPM, "SiPMhitAmplitude",1,labels,ranges,True)
    labels = {'title': "Amplitude of SiPM hits (Runs 1594-1596)", 'xlabel':'Pulse amplitude [V]','llabel':'SiPM2'}
    abp.MakeSiPMVariableDistribution(Sdf_trig_goodSiPM, "SiPMhitAmplitude",2,labels,ranges,False)
    plt.show()

    labels = {'title': "Total charge of SiPM1 hits", 'xlabel':'Pulse charge [nC]','llabel':'SiPM1'}
    ranges = {'bins': 250, 'range':(0,5.0)}
    abp.MakeSiPMVariableDistribution(Sdf_trig_goodSiPM, "SiPMhitQ",1,labels,ranges,True)
    labels = {'title': "Total Charge of SiPM hits (Runs 1594-1596)", 'xlabel':'Pulse charge [nC]','llabel':'SiPM2'}
    abp.MakeSiPMVariableDistribution(Sdf_trig_goodSiPM, "SiPMhitQ",2,labels,ranges,False)
    plt.show()


    labels = {'title': "Amplitude of SiPM1 hits (No Source)", 'xlabel':'Pulse amplitude [V]','llabel':'SiPM1'}
    ranges = {'bins': 160, 'range':(0,0.3)}
    abp.MakeSiPMVariableDistribution(Bdf_trig_cleanSiPM, "SiPMhitAmplitude",1,labels,ranges,True)
    labels = {'title': "Amplitude of SiPM hits (Runs 1611-1612)", 'xlabel':'Pulse amplitude [V]','llabel':'SiPM2'}
    abp.MakeSiPMVariableDistribution(Bdf_trig_cleanSiPM, "SiPMhitAmplitude",2,labels,ranges,False)
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    labels = {'title': "Charge of SiPM hits (No Source)", 'xlabel':'Pulse charge [nC]','llabel':'SiPM1'}
    ranges = {'bins': 250, 'range':(0,5.0)}
    abp.MakeSiPMVariableDistribution(Bdf_trig_cleanSiPM, "SiPMhitQ",1,labels,ranges,True)
    labels = {'title': "Total charge of SiPM hits (Runs 1611-1612)", 'xlabel':'Pulse charge [nC]','llabel':'SiPM2'}
    abp.MakeSiPMVariableDistribution(Bdf_trig_cleanSiPM, "SiPMhitQ",2,labels,ranges,False)
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    labels = {'title': "Sum of SiPM Charges (Source)", 'xlabel':'Total SiPM charge [nC]','ylabel':'Number of acquisitions'}
    ranges = {'bins': 200, 'range':(0,10.0)}
    abp.SiPMVariableSum(Sdf_trig_goodSiPM, "SiPMhitQ",labels,ranges)
    plt.show()

    labels = {'title': "Sum of SiPM Charges (No Source)", 'xlabel':'Total SiPM charge [nC]','ylabel':'Number of acquisitions'}
    ranges = {'bins': 200, 'range':(0,10.0)}
    abp.SiPMVariableSum(Bdf_trig_cleanSiPM, "SiPMhitQ",labels,ranges)
    plt.show()

    labels = {'title': "Sum of SiPM Charges (Source, all preliminary cuts applied)", 'xlabel':'Total SiPM charge [nC]','ylabel':'Number of acquisitions'}
    ranges = {'bins': 200, 'range':(0,10.0)}
    abp.SiPMVariableSum(Sdf_trig_CleanWindow, "SiPMhitQ",labels,ranges)
    plt.show()

    labels = {'title': "Sum of SiPM Charges (No Source, all preliminary cuts applied)", 'xlabel':'Total SiPM charge [nC]','ylabel':'Number of acquisitions'}
    ranges = {'bins': 200, 'range':(0,10.0)}
    abp.SiPMVariableSum(Bdf_trig_BurstCut, "SiPMhitQ",labels,ranges)
    plt.show()

    labels = {'title': "Sum of SiPM Charges \n (Single SiPM pulse cut only)", 'xlabel':'Total SiPM charge [nC]','llabel':'Source data','ylabel':'Number of acquisitions'}
    ranges = {'bins': 200, 'range':(0,10.0)}
    abp.SiPMVariableSum(Sdf_trig_goodSiPM, "SiPMhitQ",labels,ranges)
    labels = {'title': "Sum of SiPM Charges \n (Single SiPM pulse cut only)", 'xlabel':'Total SiPM charge [nC]','llabel':'Background data', 'ylabel':'Number of acquisitions'}
    abp.SiPMVariableSum(Bdf_trig_cleanSiPM, "SiPMhitQ",labels,ranges)
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    labels = {'title': "Sum of SiPM Charges \n (All preliminary cuts applied)", 'xlabel':'Total SiPM charge [nC]','llabel':'Source data','ylabel':'Number of acquisitions'}
    ranges = {'bins': 200, 'range':(0,10.0)}
    abp.SiPMVariableSum(Sdf_trig_CleanWindow, "SiPMhitQ",labels,ranges)
    labels = {'title': "Sum of SiPM Charges \n (All preliminary cuts applied)", 'xlabel':'Total SiPM charge [nC]','llabel':'Background data', 'ylabel':'Number of acquisitions'}
    abp.SiPMVariableSum(Bdf_trig_BurstCut, "SiPMhitQ",labels,ranges)
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()


    plt.hist(np.hstack(Sdf_SinglePulses['SiPMhitT']),bins=30,range=(0,1500),alpha=0.5,histtype='stepfilled',linewidth=6)
    plt.hist(np.hstack(Sdf_SinglePulses['SiPMhitT']),bins=30,range=(0,1500),alpha=0.75,histtype='step',linewidth=6)
    plt.title("Distribution of pulse peak times for SiPMs (Prompt window)")
    plt.xlabel("Peak time [ns]")
    plt.show()



    #plt.hist(Sdf_SinglePulses['clusterTime'],,bins=30,range=(0,1500),alpha=0.5,histtype='stepfilled',linewidth=6)
    #plt.hist(np.hstack(Sdf_SinglePulses['SiPMhitT']),bins=30,range=(0,1500),alpha=0.75,histtype='step',linewidth=6)
    plt.hist(Bdf_SinglePulses['clusterTime'],70,alpha=0.2,histtype='stepfilled',linewidth=6)
    plt.hist(Bdf_SinglePulses['clusterTime'],70,label='No source',alpha=0.8,histtype='step',linewidth=6)
    plt.hist(Sdf_SinglePulses['clusterTime'],70,alpha=0.2,histtype='stepfilled',linewidth=6)
    plt.hist(Sdf_SinglePulses['clusterTime'],70,label='Source',alpha=0.8,histtype='step',linewidth=6)
    plt.xlabel("Cluster time (ns)")
    plt.ylabel("Number of clusters")
    plt.title("Time distribution of all hit clusters \n AmBe data, (SiPM cut only, central deployment position)")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    plt.hist(Bdf_CleanWindow['clusterTime'],70,alpha=0.2,histtype='stepfilled',linewidth=6)
    plt.hist(Bdf_CleanWindow['clusterTime'],70,label='No source',alpha=0.8,histtype='step',linewidth=6)
    plt.hist(Sdf_CleanWindow['clusterTime'],70,alpha=0.2,histtype='stepfilled',linewidth=6)
    plt.hist(Sdf_CleanWindow['clusterTime'],70,label='Source',alpha=0.8,histtype='step',linewidth=6)
    plt.xlabel("Neutron candidate time (ns)")
    plt.ylabel("Number of candidates")
    plt.title("Time distribution of neutron candidates \n (All preliminary cuts applied, central AmBe data)")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    plt.hist(Sdf_CleanWindow['clusterTime'],35,range=(15000,65000),alpha=0.2,histtype='stepfilled',linewidth=6)
    plt.hist(Sdf_CleanWindow['clusterTime'],35,range=(15000,65000),label='Source data',alpha=0.8,histtype='step',linewidth=6)
    hist,bin_edges = np.histogram(Sdf_CleanWindow['clusterTime'],35,range=(15000,65000))
    bin_lefts = bin_edges[0:len(bin_edges)-1]
    print(len(hist))
    print(len(bin_lefts))
    bin_width = bin_lefts[1] - bin_lefts[0]
    bin_centers = bin_lefts + bin_width/2.
    #try making some nice bins
    init_params = [200, 30000,10000,10]
    popt, pcov = scp.curve_fit(expoPFlat, bin_centers,hist,p0=init_params, maxfev=6000)
    print("WE HERE")
    print(popt)
    print(pcov)
    myy = expoPFlat(bin_centers,popt[0],popt[1],popt[2],popt[3])
    myy_line = np.ones(len(bin_centers))*popt[3]
    tau_mean = int(popt[1]/1000)
    tau_unc = int(np.sqrt(pcov[1][1])/1000)
    plt.plot(bin_centers,myy,marker='None',linewidth=6,label=r'Best total fit $\tau = %i\pm%i \mu s$'%(tau_mean,tau_unc),color='black')
    plt.plot(bin_centers,myy_line,marker='None',linewidth=6,label=r'Flat bkg. fit',color='gray')
    plt.xlabel("Neutron candidate time (ns)")
    plt.ylabel("Number of candidates")
    plt.title("Time distribution of neutron candidates \n (All preliminary cuts applied, AmBe central source data)")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    #Calculate chi^2/ndof
    chisq = np.sum(((hist - myy)**2)/hist)
    ndof = np.size(hist)
    print("CHISQ/NDOF: %s/%s"%(str(chisq),str(ndof)))

    #Find the baseline with the bin resolution used in Vincent's plot
    plt.hist(Sdf_CleanWindow['clusterTime'],75,range=(15000,65000),alpha=0.8)
    plt.show()
    plt.hist(Sdf_CleanWindow['clusterTime'],75,range=(15000,65000),alpha=0.8)
    dhist,dbin_edges = np.histogram(Sdf_CleanWindow['clusterTime'],75,range=(15000,65000))
    dbin_lefts = dbin_edges[0:len(dbin_edges)-1]
    dbin_width = dbin_lefts[1] - dbin_lefts[0]
    dbin_centers = dbin_lefts + dbin_width/2.
    init_params = [40, 30000,15000,5]
    popt, pcov = scp.curve_fit(expoPFlat, dbin_centers,dhist,p0=init_params, maxfev=6000)
    print("WE ARE HERE")
    print(popt)
    print(pcov)
    myy = expoPFlat(dbin_centers,popt[0],popt[1],popt[2],popt[3])
    myy_line = np.ones(len(dbin_centers))*popt[3]
    flat_bkg = popt[3]
    tau_mean = int(popt[1]/1000)
    tau_unc = int(np.sqrt(pcov[1][1])/1000)
    plt.plot(dbin_centers,myy,marker='None',linewidth=6,label=r'Best total fit $\tau = %i\pm%i \mu s$'%(tau_mean,tau_unc),color='black')
    plt.plot(dbin_centers,myy_line,marker='None',linewidth=6,label=r'Flat bkg. fit',color='gray')
    plt.xlabel("Cluster time (ns)")
    plt.ylabel("Number of clusters")
    plt.title("Time distribution of delayed hit clusters (MCBins) \n (One pulse in each SiPM, no prompt cluster)")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()
    
    #Get MC
    #mchist,mcbin_lefts = fr.ReadHistFile(MCDELTAT)
    #mcrange = np.where(mcbin_lefts<70100)[0]
    #mchist = mchist[mcrange]
    #mcbin_lefts = mcbin_lefts[mcrange]
    mchist,mcbin_lefts = fr.ReadHistFileSimple(MCDELTAT)
    print(mchist)
    print(mcbin_lefts)
    mchist,mcbin_lefts = hu.AggregateBins(mchist,mcbin_lefts,100,0,67000)
    mchist_unc = np.sqrt(mchist) 
    mchist_normed = mchist/np.sum(mchist)
    mchist_normed_unc = mchist_unc/np.sum(mchist)

    #Plot data over MC range, subtract baseline
    dhist,dbin_edges = np.histogram(Sdf_CleanWindow['clusterTime'],100,range=(0,67000))
    dbin_lefts = dbin_edges[0:len(dbin_edges)-1]
    dhist_nobkg = dhist-popt[3]
    #dhist_nobkg = dhist
    neg_bins = np.where(dhist_nobkg<0)[0]
    dhist_nobkg[neg_bins] = 0
    dhist_nobkg_unc = np.sqrt(dhist_nobkg) #TODO: could propagate uncertainty on flat fit too
    dhist_nobkg_normed = dhist_nobkg/np.sum(dhist_nobkg)
    dhist_nobkg_normed_unc = dhist_nobkg_unc/np.sum(dhist_nobkg)

    plt.errorbar(dbin_lefts,dhist_nobkg_normed,yerr=dhist_nobkg_normed_unc,
            linestyle='None',marker='o',label='AmBe source data')
    plt.errorbar(mcbin_lefts,mchist_normed,yerr=mchist_normed_unc,
            marker='o',linestyle='None',label='RATPAC MC')
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Comparison of neutron candidate time distribution in central data/MC")
    plt.xlabel("Neutron candidate time (ns)")
    plt.ylabel("Arb. units")
    plt.show()


    #CleanEventNumbers = Sdf_CleanPrompt["eventNumber"]
    #Return DF with triggers that have no prompt and one of each SiPM pulse
    #MSData = abp.MakeClusterMultiplicityPlot(Sdf_CleanPrompt,Sdf_trig_CleanPrompt)
    MSData = abp.MakeClusterMultiplicityPlot(Sdf_CleanWindow_CBClean,Sdf_trig_CleanWindow)
    print("NUMBER OF TRIGS: " + str(len(MSData)))
    print("NUMBER OF ZERO MULTIPLICITY TRIGS: " + str(len(np.where(MSData==0)[0])))
    s_bins, s_edges = np.histogram(MSData,bins=20, range=(0,20))
    print("SIGNAL_BINS: " + str(s_bins))
    print("SIGNAL_BIN_EDGES: " + str(s_edges))
    plt.hist(MSData,bins=20, range=(0,20), label="Source",alpha=0.7)
    plt.xlabel("Delayed cluster multiplicity")
    plt.title("Cluster multiplicity for delayed window of central source run \n (One pulse each SiPM, no prompt cluster)")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    plt.hist(Bdf_NBCBCut['clusterTime'],100,range=(12000,65000),label='No source', color='purple',alpha=0.7)
    #plt.hist(Bdf_SinglePulses['clusterTime'],100,label='No source', color='purple',alpha=0.7)
    plt.xlabel("Cluster time (ns)")
    plt.ylabel("Number of candidates")
    plt.title("Region to characterize flat background distribution \n (No source data, SiPM cut + Burst cut)")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    Bdf_trig_goodSiPM = Bdf_trig.loc[(Bdf_trig['SiPM1NPulses']==1) & (Bdf_trig['SiPM2NPulses']==1)].reset_index(drop=True)
    Bdf_trig_CleanWindow = es.NoBurst_WholeFile(Bdf_SinglePulses,Bdf_trig_goodSiPM,12000,150)
    #Bdf_trig_bkgclean = es.FilterByEventNumber(Bdf_trig,CleanBkgNumbers)
    #Get cluster multiplicity of late window in events passing prompt criteria
    MBData = abp.MakeClusterMultiplicityPlot(Bdf_NBCBCut,Bdf_trig_CleanWindow)


    MBData = abp.MakeClusterMultiplicityPlot(Bdf_latewindow,Bdf_trig_BurstCut)
    print("NUMBER OF TRIGS: " + str(len(MBData)))
    print("NUMBER OF ZERO MULTIPLICITY TRIGS (LATE WINDOW): " + str(len(np.where(MBData==0)[0])))
    b_bins, b_edges = np.histogram(MBData,bins=20, range=(0,20))
    print("BKG_BINS: " + str(b_bins))
    print("BKG_BIN_EDGES: " + str(b_edges))
    plt.hist(MBData,bins=20, range=(0,20), label="No source",alpha=0.7)
    plt.xlabel("Delayed cluster multiplicity")
    plt.ylabel("Number of acquisitions")
    plt.title("Cluster multiplicity for delayed window of central source run \n (One pulse each SiPM, no prompt cluster)")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    #Plotting the PE distribution
    Sdf_latewindow = Sdf_SinglePulses.loc[(Sdf_SinglePulses['clusterTime']>2000)].reset_index(drop=True)
    plt.hist(Sdf_CleanWindow['clusterPE'],70,alpha=0.2,histtype='stepfilled',linewidth=6)
    plt.hist(Sdf_CleanWindow['clusterPE'],70,label='Source ($>2 \, \mu s$)',alpha=0.8,histtype='step',linewidth=6)
    plt.hist(Bdf_NBCBCut['clusterPE'],70,alpha=0.2,histtype='stepfilled',linewidth=6)
    plt.hist(Bdf_NBCBCut['clusterPE'],70,label='No source ($>12 \, \mu s$)',alpha=0.8,histtype='step',linewidth=6)
    plt.xlabel("PE")
    plt.ylabel("Number of clusters")
    plt.title("PE distribution for delayed clusters \n (SiPM cut only, AmBe central data")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

    #Get MC
    #mchist,mcbin_lefts = fr.ReadHistFile(MCPE)
    mchist,mcbin_lefts = fr.ReadHistFileSimple(MCPE)
    mchist,mcbin_lefts = hu.AggregateBins(mchist,mcbin_lefts,75,0,150)
    mchist_normed = mchist/np.sum(mchist)
    mchist_unc = np.sqrt(mchist)
    mchist_normed_unc = mchist_unc/np.sum(mchist)

    #Plot data over MC range, subtract baseline
    dhist,dbin_edges = np.histogram(Sdf_CleanWindow['clusterPE'],75,range=(0,150))
    dbin_lefts = dbin_edges[0:len(dbin_edges)-1]
    dhist_unc = np.sqrt(dhist) #TODO: could propagate uncertainty on flat fit too
    dhist_normed = dhist/np.sum(dhist)
    dhist_normed_unc = dhist_unc/np.sum(dhist)

    plt.errorbar(dbin_lefts,dhist_normed,yerr=dhist_normed_unc,
            linestyle='None',marker='o',label='AmBe source data')
    plt.errorbar(mcbin_lefts,mchist_normed,yerr=mchist_normed_unc,
            marker='o',linestyle='None',label='RATPAC MC')
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Comparison of neutron candidate PE distribution in central data/MC")
    plt.xlabel("Neutron candidate PE")
    plt.ylabel("Arb. units")
    plt.show()

    #Same as above, but plotting past 25 PE
    mchist,mcbin_lefts = fr.ReadHistFileSimple(MCPE)
    mchist,mcbin_lefts = hu.AggregateBins(mchist,mcbin_lefts,70,20,160)
    mchist_normed = mchist/np.sum(mchist)
    mchist_unc = np.sqrt(mchist)
    mchist_normed_unc = mchist_unc/np.sum(mchist)

    dhist,dbin_edges = np.histogram(Sdf_CleanWindow['clusterPE'],70,range=(20,160))
    dbin_lefts = dbin_edges[0:len(dbin_edges)-1]
    dhist_unc = np.sqrt(dhist) 
    dhist_normed = dhist/np.sum(dhist)
    dhist_normed_unc = dhist_unc/np.sum(dhist)
    plt.errorbar(dbin_lefts,dhist_normed,yerr=dhist_normed_unc,
            linestyle='None',marker='o',label='AmBe source data')
    plt.errorbar(mcbin_lefts,mchist_normed,yerr=mchist_normed_unc,
            marker='o',linestyle='None',label='RATPAC MC')
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Comparison of neutron candidate PE distribution in central data/MC \n (Preliminary cuts applied except CB cut)")
    plt.xlabel("Neutron candidate PE")
    plt.ylabel("Arb. units")
    plt.show()



    #Plotting the PE distribution
    Bdf_latewindow_CBClean = Bdf_NBlatewindow.loc[Bdf_NBlatewindow['clusterChargeBalance']<0.4]
    Sdf_CleanWindow_CBClean = Sdf_CleanWindow.loc[Sdf_CleanWindow['clusterChargeBalance']<0.4]
    plt.hist(Sdf_CleanWindow_CBClean['clusterPE'],70,alpha=0.2,histtype='stepfilled',linewidth=6)
    plt.hist(Sdf_CleanWindow_CBClean['clusterPE'],70,label='Source ($>2 \, \mu s$)',alpha=0.8,histtype='step',linewidth=6)
    plt.hist(Bdf_latewindow_CBClean['clusterPE'],70,alpha=0.2,histtype='stepfilled',linewidth=6)
    plt.hist(Bdf_latewindow_CBClean['clusterPE'],70,label='No source \n (No tank cut, $>12 \, \mu s$ clusters)',alpha=0.8,histtype='step',linewidth=6)
    plt.xlabel("PE")
    plt.ylabel("Number of clusters")
    plt.title("PE distribution for delayed clusters \n (All preliminary cuts applied, AmBe central runs)")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()


    labels = {'title': 'Comparison of total PE to charge balance parameter \n (Position 0, AmBe source installed)', 
            'xlabel': 'Total PE', 'ylabel': 'Charge balance parameter'}
    ranges = {'xbins': 50, 'ybins':50, 'xrange':[0,80],'yrange':[0,1]}
    #abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    abp.Make2DHist(Sdf_CleanWindow,'clusterPE','clusterChargeBalance',labels,ranges)
    plt.show()

    labels = {'title': 'Comparison of total PE to charge balance parameter \n (Position 0, no AmBe source installed, >12 $\mu$s)', 
            'xlabel': 'Total PE', 'ylabel': 'Charge balance parameter'}
    ranges = {'xbins': 50, 'ybins':50, 'xrange':[0,80],'yrange':[0,1]}
    #abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    Bdf_latewindow = Bdf.loc[Bdf['clusterTime']>12000]
    abp.Make2DHist(Bdf_latewindow,'clusterPE','clusterChargeBalance',labels,ranges)
    plt.show()

    labels = {'title': 'Comparison of total PE to charge balance parameter', 
            'xlabel': 'Total PE', 'ylabel': 'Charge balance parameter','llabel':'No source',
            'color':'Reds'}
    ranges = {'xbins': 50, 'ybins':50, 'xrange':[0,150],'yrange':[0,1]}
    #abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)

    Bdf_window = Bdf_latewindow.loc[(Bdf_latewindow['clusterPE']<150) & \
            (Bdf_latewindow['clusterChargeBalance']>0) & (Bdf_latewindow['clusterChargeBalance']<1)]
    Sdf_window = Sdf_CleanWindow.loc[(Sdf_CleanWindow['clusterPE']<150) & \
            (Sdf_CleanWindow['clusterChargeBalance']>0) & (Sdf_CleanWindow['clusterChargeBalance']<1)]
    abp.MakeKDEPlot(Bdf_window,'clusterPE','clusterChargeBalance',labels,ranges)
    labels = {'title': 'Comparison of total PE to charge balance parameter', 
            'xlabel': 'Total PE', 'ylabel': 'Charge balance parameter','llabel':'No source',
            'color':'Blues'}
    abp.MakeKDEPlot(Sdf_window,'clusterPE','clusterChargeBalance',labels,ranges)
    plt.show()
    #labels = {'title': 'Comparison of total PE to max PE (Source)', 
    #        'xlabel': 'Total PE', 'ylabel': 'Max PE'}
    #ranges = {'xbins': 40, 'ybins':40, 'xrange':[0,60],'yrange':[0,15]}
    ##abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    #abp.Make2DHist(Sdf_CleanWindow,'clusterPE','clusterMaxPE',labels,ranges)
    #plt.show()

    #labels = {'title': 'Comparison of total PE to max PE (No source, >20 $\mu$s)', 
    #        'xlabel': 'Total PE', 'ylabel': 'Max PE'}
    #ranges = {'xbins': 40, 'ybins':40, 'xrange':[0,60],'yrange':[0,15]}
    ##abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    #Bdf_latewindow = Bdf.loc[Bdf['clusterTime']>12000]
    #abp.Make2DHist(Bdf_latewindow,'clusterPE','clusterMaxPE',labels,ranges)
    #plt.show()
    
    #Sdf_CleanWindow_CBCut = Sdf_CleanWindow.loc[Sdf_CleanWindow['clusterChargeBalance']<0.4].reset_index(drop=True)
    #labels = {'title': 'Comparison of total PE to Charge Point Y-component \n (Source, Charge Balance<0.4)', 
    #        'xlabel': 'Total PE', 'ylabel': 'Charge Point Y'}
    #ranges = {'xbins': 30, 'ybins':30, 'xrange':[0,60],'yrange':[-1,1]}
    ##abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    #abp.Make2DHist(Sdf_CleanWindow_CBCut,'clusterPE','clusterChargePointY',labels,ranges)
    #plt.show()

    #labels = {'title': 'Comparison of total PE to Charge Point Z-component \n (Source, Charge Balance<0.4)', 
    #        'xlabel': 'Total PE', 'ylabel': 'Charge Point Z'}
    #ranges = {'xbins': 30, 'ybins':30, 'xrange':[0,60],'yrange':[-1,1]}
    ##abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    #abp.Make2DHist(Sdf_CleanWindow_CBCut,'clusterPE','clusterChargePointZ',labels,ranges)
    #plt.show()

    #labels = {'title': 'Comparison of total PE to Charge Point Y-component (No source, >20 $\mu$s)', 
    #        'xlabel': 'Total PE', 'ylabel': 'Charge Point Y'}
    #ranges = {'xbins': 30, 'ybins':30, 'xrange':[0,60],'yrange':[-1,1]}
    ##abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    #abp.Make2DHist(Bdf_latewindow,'clusterPE','clusterChargePointY',labels,ranges)
    #plt.show()


    #Bdf_latewindow_CBCut = Bdf_latewindow.loc[Bdf_latewindow['clusterChargeBalance']>0.4]
    #labels = {'title': 'Comparison of total PE to Charge Point Y-component \n (No source, >20 $\mu$s, Charge Balance > 0.4)', 
    #        'xlabel': 'Total PE', 'ylabel': 'Charge Point Y'}
    #ranges = {'xbins': 30, 'ybins':30, 'xrange':[0,60],'yrange':[-1,1]}
    ##abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    #abp.Make2DHist(Bdf_latewindow_CBCut,'clusterPE','clusterChargePointY',labels,ranges)
    #plt.show()



if __name__=='__main__':
    slist = glob.glob(SIGNAL_DIR+"*.ntuple.root")
    blist = glob.glob(BKG_DIR+"*.ntuple.root")

    livetime_estimate = es.EstimateLivetime(slist)
    print("SIGNAL LIVETIME ESTIMATE IN SECONDS IS: " + str(livetime_estimate))
    livetime_estimate = es.EstimateLivetime(blist)
    print("BKG LIVETIME ESTIMATE IN SECONDS IS: " + str(livetime_estimate))

    mybranches = ['eventNumber','eventTimeTank','clusterTime','SiPMhitT','SiPMhitQ','SiPMhitAmplitude','clusterChargeBalance','clusterPE','SiPM1NPulses','SiPM2NPulses','SiPMNum','clusterHits']
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

    BProcessor = rp.ROOTProcessor(treename="phaseIITriggerTree")
    for f1 in blist:
        BProcessor.addROOTFile(f1,branches_to_get=mybranches)
    Bdata = BProcessor.getProcessedData()
    Bdf_trig = pd.DataFrame(Bdata)
    PlotDemo(Sdf,Bdf,Sdf_trig,Bdf_trig)



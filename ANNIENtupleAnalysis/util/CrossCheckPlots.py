#This script has functions needed to make a Cumulative Distribution Plot from
#Different variables output in  PhaseIITreeMaker root file.

import glob

import sys
import uproot
import lib.ROOTProcessor as rp
import lib.EventSelection as es

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sco
import scipy.special as scs


PICKLE_DIR = "./Data/CentralData2020/pickles/"
BKG_DATA_DIR = "./Data/CentralData2020/Background/"

gauss= lambda x,A,mu,sigma: A*np.exp(-((x-mu)**2)/((2*sigma)**2))
gaussplusflat= lambda x,A,mu,sigma, b: A*np.exp(-((x-mu)**2)/((2*sigma)**2)) + b

def EffFunction(x,C,s,b):
    #Attempting to use the error function to parameterize the 
    #trigger efficiency shape
    return C*((scs.erf(s*x + b) + 1)/2.0)

def PlotDemo(Sdf,Sdf_trig,Bdf,Bdf_trig):
    
    #Cuts can be applied to pandas dataframes; EventSelection.py has some cuts defined,
    #And some are directly applied here as well.  Examples of applying cuts at both the 
    #Cluster level and trigger level are shown.

    #Plot the sum of SiPM charges for events prior to cuts
    #Simple 1D histogram; plot total PE of all clusters
    Sig_SiPM_ChargeSum = []
    for i in Sdf_trig.index.values:
        Sig_SiPM_ChargeSum.append(np.sum(Sdf_trig['SiPMhitQ'][i]))
    #Sig_SiPM_ChargeSum = np.array(Sig_SiPM_ChargeSum)/np.sum(Sig_SiPM_ChargeSum)
    Bkg_SiPM_ChargeSum = []
    for i in Bdf_trig.index.values:
        Bkg_SiPM_ChargeSum.append(np.sum(Bdf_trig['SiPMhitQ'][i]))
    #Bkg_SiPM_ChargeSum = np.array(Bkg_SiPM_ChargeSum)/np.sum(Bkg_SiPM_ChargeSum)
    plt.hist(Sig_SiPM_ChargeSum,bins=2600,range=(0,1),alpha=0.5,histtype='stepfilled',linewidth=6,color='blue',label="Source (Run 57)")
    plt.hist(Bkg_SiPM_ChargeSum,bins=2600,range=(0,1),alpha=0.5,histtype='stepfilled',linewidth=6,color='orange',label="No source (Run 69)")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Sum of SiPM Charges in AmBe Central Data")
    plt.xlabel("Summed SiPM Charge (nC)")
    plt.show()


    #Plot charges and amplitudes for all SiPM pulses
    #Simple 1D histogram; plot total PE of all clusters
    Sig_SiPM1_Charges = []
    Sig_SiPM1_Amplitudes = []
    Sig_SiPM2_Charges = []
    Sig_SiPM2_Amplitudes = []
    for i in Sdf_trig.index.values:
        Sig_SiPM1ind=np.where(np.array(Sdf_trig["SiPMNum"][i]) == 1)[0]
        Sig_SiPM2ind=np.where(np.array(Sdf_trig["SiPMNum"][i]) == 2)[0]
        Sig_SiPM1_Charges = Sig_SiPM1_Charges + list(np.array(Sdf_trig["SiPMhitQ"][i])[Sig_SiPM1ind])
        Sig_SiPM2_Charges = Sig_SiPM2_Charges + list(np.array(Sdf_trig["SiPMhitQ"][i])[Sig_SiPM2ind])
        Sig_SiPM1_Amplitudes = Sig_SiPM1_Amplitudes + list(np.array(Sdf_trig["SiPMhitAmplitude"][i])[Sig_SiPM1ind])
        Sig_SiPM2_Amplitudes = Sig_SiPM2_Amplitudes + list(np.array(Sdf_trig["SiPMhitAmplitude"][i])[Sig_SiPM2ind])

    plt.hist(Sig_SiPM1_Charges,bins=2600,range=(0,1),alpha=0.5,histtype='stepfilled',linewidth=6,color='blue',label="SiPM 1")
    plt.hist(Sig_SiPM2_Charges,bins=2600,range=(0,1),alpha=0.5,histtype='stepfilled',linewidth=6,color='orange',label="SiPM 2")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("SiPM Charges in AmBe Central Data \n (No preliminary cuts applied)")
    plt.xlabel("SiPM Charge (nC)")
    plt.show()

    plt.hist(Sig_SiPM1_Amplitudes,bins=2600,range=(0,1),alpha=0.5,histtype='stepfilled',linewidth=6,color='blue',label="SiPM 1")
    plt.hist(Sig_SiPM2_Amplitudes,bins=2600,range=(0,1),alpha=0.5,histtype='stepfilled',linewidth=6,color='orange',label="SiPM 2")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("SiPM Amplitudes in AmBe Central Data \n (No preliminary cuts applied)")
    plt.xlabel("SiPM Amplitude")
    plt.show()



    #Fit the background charge histogram
    bin_vals, bin_edges = np.histogram(Bkg_SiPM_ChargeSum,bins=260, range=(0.01,0.6))
    print("LEN BIN VALS: " + str(len(bin_vals)))
    print("LEN BIN EDGES: " + str(len(bin_edges)))
    bin_edges = bin_edges[0:len(bin_edges)-1]
    popt, pcov = sco.curve_fit(EffFunction, bin_edges, bin_vals, p0=[10, 10, 0.1], maxfev=6000)
    print("BEST CONST: " + str(popt[0]))
    print("BEST STRETCH: " + str(popt[1]))
    print("BEST SHIFT: " + str(popt[2]))
    plt.plot(bin_edges,bin_vals,color='orange', label="No source (Run 69)")
    fit_range = np.arange(-0.3,0.6,0.001)
    plt.plot(fit_range, EffFunction(fit_range,popt[0],popt[1],popt[2]),label='fit (0.01,0.6 nC)',color='blue')
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Sum of SiPM Charges in AmBe Central Background Data")
    plt.xlabel("Summed SiPM Charge (nC)")
    plt.show()

    #Now get the trigger efficiency correction
    eff_correction = EffFunction(fit_range,popt[0],popt[1],popt[2])
    eff_xaxis = fit_range
    trig_eff = eff_correction/np.max(eff_correction)
    plt.hist(Sig_SiPM_ChargeSum,bins=2600,range=(0,1),alpha=0.5,histtype='stepfilled',linewidth=6,color='red',label="Source (Run 57)")
    plt.plot(eff_xaxis, trig_eff,color='blue',label='Trigger Eff. Fit')
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Sum of SiPM Charges in AmBe Central Data")
    plt.xlabel("Summed SiPM Charge (nC)")
    plt.show()


    #Fit the signal charge sum histogram
    bin_vals, bin_edges = np.histogram(Sig_SiPM_ChargeSum,bins=260, range=(0.05,0.15))
    print("LEN BIN VALS: " + str(len(bin_vals)))
    print("LEN BIN EDGES: " + str(len(bin_edges)))
    bin_edges = bin_edges[0:len(bin_edges)-1]
    popt, pcov = sco.curve_fit(gaussplusflat, bin_edges, bin_vals, p0=[10000, 0.1, 0.05,10], maxfev=6000)
    print("BEST MEAN: " + str(popt[1]))
    print("BEST SIGA: " + str(popt[2]))
    print("BEST FLAT: " + str(popt[3]))
    plt.hist(Sig_SiPM_ChargeSum,bins=2600,range=(0,1),alpha=0.5,histtype='stepfilled',linewidth=6,color='red',label="Source (Run 57)")
    fit_range = np.arange(0.0,1,0.001)
    plt.plot(fit_range, gaussplusflat(fit_range,popt[0],popt[1],popt[2],popt[3]),label='fit (0.05,0.15 nC)',color='blue')
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Sum of SiPM Charges in AmBe Central Data")
    plt.xlabel("Summed SiPM Charge (nC)")
    plt.show()


    #NOW, APPLY TRIGGER EFFICIENCY SCALING AND TRY TO FIT GAUSS + FLAT BKG
    corrected_bin_vals = []
    for j,binedge in enumerate(bin_edges):
        correction_ind = np.where(binedge>eff_xaxis)[0][-1]
        print("THIS BIN VAL: " + str(bin_vals[j]))
        print("THIS TRIG EFF: " + str(trig_eff[correction_ind]))
        corrected_bin_vals.append(float(bin_vals[j]) * trig_eff[correction_ind])
    corrected_bin_vals = np.array(corrected_bin_vals)
    #Fit the signal charge sum histogram
    popt, pcov = sco.curve_fit(gaussplusflat, bin_edges, corrected_bin_vals, p0=[10000, 0.1, 0.05,10], maxfev=6000)
    print("BEST MEAN: " + str(popt[1]))
    print("BEST SIGA: " + str(popt[2]))
    print("BEST BKG: " + str(popt[3]))
    plt.plot(bin_edges,corrected_bin_vals,color='orange', label='corrected SiPM charge sum')
    fit_range = np.arange(0.0,1,0.001)
    plt.plot(fit_range, gauss(fit_range,popt[0],popt[1],popt[2]),label='gauss fit (0.05,0.15 nC)',color='blue')
    plt.plot(fit_range, np.ones(len(fit_range))*popt[3],label='flat bkg fit (0.05,0.15 nC)',color='black')
    plt.plot(fit_range, gaussplusflat(fit_range,popt[0],popt[1],popt[2],popt[3]),label='total fit (0.05,0.15 nC)',color='purple')
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Sum of SiPM Charges in AmBe Central Data \n (trigger efficiency correction applied)")
    plt.xlabel("Summed SiPM Charge (nC)")
    plt.show()


    #Cluster level cuts
    #Sdf_SinglePulses = es.SingleSiPMPulses(Sdf)
    #Sdf_PlusCleanSiPMPrompt = es.NoPromptClusters(Sdf_SinglePulses,2000)
    #Sdf_PlusNoHighPEClusters = es.NoBurstClusters(Sdf_PlusCleanSiPMPrompt,2000,150)
    #Sdf_PlusGoodCB = Sdf_PlusNoHighPEClusters.loc[Sdf_PlusNoHighPEClusters['clusterChargeBalance']<0.4]
    ##Trigger level cuts
    #Sdf_trig_goodSiPM = Sdf_trig.loc[(Sdf_trig['SiPM1NPulses']==1) & (Sdf_trig['SiPM2NPulses']==1)].reset_index(drop=True)
    #Sdf_trig_CleanPrompt = es.NoPromptClusters_WholeFile(Sdf_SinglePulses,Sdf_trig_goodSiPM,2000)
    #Sdf_trig_CleanWindow = es.NoBurst_WholeFile(Sdf_PlusCleanSiPMPrompt,Sdf_trig_CleanPrompt,2000,150)

    #Sdf_clean = Sdf_PlusGoodCB
    #Sdf_trig_clean = Sdf_trig_CleanWindow

if __name__=='__main__':
    Sdf = pd.read_pickle(PICKLE_DIR+"SdfCentralData2020.pkl")
    Sdf_trig = pd.read_pickle(PICKLE_DIR+"Sdf_trigCentralData2020.pkl")
    Bdf = pd.read_pickle(PICKLE_DIR+"BdfCentralData2020.pkl")
    Bdf_trig = pd.read_pickle(PICKLE_DIR+"Bdf_trigCentralData2020.pkl")
    PlotDemo(Sdf,Sdf_trig,Bdf,Bdf_trig)



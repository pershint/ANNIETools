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
#PICKLE_DIR = "./Data/V3_5PE100ns/Pos0pickles/"

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

    plt.hist(Sig_SiPM1_Charges,bins=1000,range=(0,1),alpha=0.5,histtype='stepfilled',linewidth=6,color='blue',label="SiPM 1")
    plt.hist(Sig_SiPM2_Charges,bins=1000,range=(0,1),alpha=0.5,histtype='stepfilled',linewidth=6,color='orange',label="SiPM 2")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("SiPM Charges in AmBe Central Data \n (No preliminary cuts applied)")
    plt.xlabel("SiPM Charge (nC)")
    plt.show()

    plt.hist(Sig_SiPM1_Amplitudes,bins=100,range=(0,0.2),alpha=0.5,histtype='stepfilled',linewidth=6,color='blue',label="SiPM 1")
    plt.hist(Sig_SiPM2_Amplitudes,bins=100,range=(0,0.2),alpha=0.5,histtype='stepfilled',linewidth=6,color='orange',label="SiPM 2")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("SiPM Amplitudes in AmBe Central Data \n (No preliminary cuts applied)")
    plt.xlabel("SiPM Amplitude")
    plt.show()

    #Plot charges and amplitudes for all SiPM pulses with preliminary cuts
    #Simple 1D histogram; plot total PE of all clusters
    Sdf_SinglePulses = es.SingleSiPMPulses(Sdf)
    Sdf_CleanPrompt = es.NoPromptClusters(Sdf_SinglePulses,2000)
    Sdf_trig_SinglePulses = Sdf_trig.loc[(Sdf_trig['SiPM1NPulses']==1) & (Sdf_trig['SiPM2NPulses']==1)].reset_index(drop=True)
    Sdf_trig_CleanPrompt = es.NoPromptClusters_WholeFile(Sdf_SinglePulses,Sdf_trig_SinglePulses,2000)
    Sdf_trig_CleanWindow = es.NoBurst_WholeFile(Sdf_CleanPrompt,Sdf_trig_CleanPrompt,2000,150)
    Sig_SiPM1_Charges = []
    Sig_SiPM1_Amplitudes = []
    Sig_SiPM1_Times = []
    Sig_SiPM2_Charges = []
    Sig_SiPM2_Times = []
    Sig_SiPM2_Amplitudes = []
    for i in Sdf_trig_CleanWindow.index.values:
        Sig_SiPM1ind=np.where(np.array(Sdf_trig_CleanWindow["SiPMNum"][i]) == 1)[0]
        Sig_SiPM2ind=np.where(np.array(Sdf_trig_CleanWindow["SiPMNum"][i]) == 2)[0]
        Sig_SiPM1_Charges = Sig_SiPM1_Charges + list(np.array(Sdf_trig_CleanWindow["SiPMhitQ"][i])[Sig_SiPM1ind])
        Sig_SiPM1_Times = Sig_SiPM1_Times + list(np.array(Sdf_trig_CleanWindow["SiPMhitT"][i])[Sig_SiPM1ind])
        Sig_SiPM2_Charges = Sig_SiPM2_Charges + list(np.array(Sdf_trig_CleanWindow["SiPMhitQ"][i])[Sig_SiPM2ind])
        Sig_SiPM2_Times = Sig_SiPM2_Times + list(np.array(Sdf_trig_CleanWindow["SiPMhitT"][i])[Sig_SiPM2ind])
        Sig_SiPM1_Amplitudes = Sig_SiPM1_Amplitudes + list(np.array(Sdf_trig_CleanWindow["SiPMhitAmplitude"][i])[Sig_SiPM1ind])
        Sig_SiPM2_Amplitudes = Sig_SiPM2_Amplitudes + list(np.array(Sdf_trig_CleanWindow["SiPMhitAmplitude"][i])[Sig_SiPM2ind])

    plt.hist(Sig_SiPM1_Charges,bins=1000,range=(0,1),alpha=0.5,histtype='stepfilled',linewidth=6,color='blue',label="SiPM 1")
    plt.hist(Sig_SiPM2_Charges,bins=1000,range=(0,1),alpha=0.5,histtype='stepfilled',linewidth=6,color='orange',label="SiPM 2")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("SiPM Charges in AmBe Central Data \n (Preliminary cuts applied)")
    plt.xlabel("SiPM Charge (nC)")
    plt.show()
    
    plt.hist(Sig_SiPM1_Times,bins=500,range=(0,2000),alpha=0.5,histtype='stepfilled',linewidth=6,color='blue',label="SiPM 1")
    plt.hist(Sig_SiPM2_Times,bins=500,range=(0,2000),alpha=0.5,histtype='stepfilled',linewidth=6,color='orange',label="SiPM 2")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("SiPM Times in AmBe Central Data \n (Preliminary cuts applied)")
    plt.xlabel("SiPM Pulse Time (ns)")
    plt.show()

    plt.hist(Sig_SiPM1_Amplitudes,bins=100,range=(0,0.2),alpha=0.5,histtype='stepfilled',linewidth=6,color='blue',label="SiPM 1")
    plt.hist(Sig_SiPM2_Amplitudes,bins=100,range=(0,0.2),alpha=0.5,histtype='stepfilled',linewidth=6,color='orange',label="SiPM 2")
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("SiPM Amplitudes in AmBe Central Data \n (No preliminary cuts applied)")
    plt.xlabel("SiPM Amplitude")
    plt.show()



if __name__=='__main__':
    Sdf = pd.read_pickle(PICKLE_DIR+"SdfCentralData2020.pkl")
    Sdf_trig = pd.read_pickle(PICKLE_DIR+"Sdf_trigCentralData2020.pkl")
    Bdf = pd.read_pickle(PICKLE_DIR+"BdfCentralData2020.pkl")
    Bdf_trig = pd.read_pickle(PICKLE_DIR+"Bdf_trigCentralData2020.pkl")
    #Sdf = pd.read_pickle(PICKLE_DIR+"SdfPos0.pkl")
    #Sdf_trig = pd.read_pickle(PICKLE_DIR+"Sdf_trigPos0.pkl")
    #Bdf = pd.read_pickle(PICKLE_DIR+"BdfPos0.pkl")
    #Bdf_trig = pd.read_pickle(PICKLE_DIR+"Bdf_trigPos0.pkl")
    PlotDemo(Sdf,Sdf_trig,Bdf,Bdf_trig)



import glob

import sys
import uproot
import lib.ROOTProcessor as rp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import pandas as pd

sns.set_context('poster')
sns.set(font_scale=2.5)
sns.set_style("whitegrid")
sns.axes_style("darkgrid")
xkcd_colors = ['dark teal','adobe','purple','salmon']
sns.set_palette(sns.xkcd_palette(xkcd_colors))

def EstimateLivetime(filelist):
    '''
    Estimate live time using the smallest and 
    largest time stamps in each separate file.  One or two 
    events are being set to an unphysically small or large number though,
    have to investigate.
    '''
    total_time = 0
    mybranches = ['eventTimeTank']
    for f1 in filelist:
        f1Processor = rp.ROOTProcessor(treename="phaseIITriggerTree")
        f1Processor.addROOTFile(f1,branches_to_get=mybranches)
        f1data = f1Processor.getProcessedData()
        f1data_pd = pd.DataFrame(f1data)
        early_time = np.min(f1data_pd.loc[(f1data_pd["eventTimeTank"]>1E6)].values)/1E9
        late_time = np.max(f1data_pd.loc[(f1data_pd["eventTimeTank"]<2.0E18)].values)/1E9
        print("EARLY_TIME: " + str(early_time))
        print("LATE_TIME: " + str(late_time))
        print("LATE - EARLY TIME: " + str(late_time - early_time))
        total_time+=(late_time-early_time)
    return total_time

def ValidPromptClusterEvents(df,clusterTimeCut):
    '''
    Given a dataframe and prompt cut, return all clusters associated
    with an event that only has one SiPM pulse per pulse and no clusters
    at less than the clusterTimeCut variable.
    '''
    NoPromptClusterDict = {}
    OnePulses = np.where((df.SiPM1NPulses==1) & (df.SiPM2NPulses==1))[0]
    DirtyPromptEvents = []
    for j in range(len(df)):  #disgusting...
        if df["clusterTime"][j] < clusterTimeCut:
            DirtyPromptEvents.append(df["eventNumber"][j])
    CleanIndices = []
    for j in range(len(df)):
        if df["eventNumber"][j] not in DirtyPromptEvents:
            CleanIndices.append(j)
    CleanIndices = np.array(CleanIndices)
    Cleans = np.intersect1d(OnePulses,CleanIndices)
    df_CleanPrompt = df.loc[Cleans]
    return df_CleanPrompt

def MakeClusterTimeDistribution(df):
    '''
    Plot the time distribution for all clusters in the file.
    '''
    CleanPromptDF = ValidPromptClusterEvents(df,2000)
    plt.hist(CleanPromptDF['clusterTime'],100)
    plt.xlabel("Cluster time (ns)")
    plt.title("Time distribution of hit clusters")
    plt.show()

def MakeSiPMVariableDistribution(df, variable, sipm_num, labels, ranges, SingleSiPMPulses):
    '''
    Plot the SiPM variable distributions SiPMhitQ, SiPMhitT, or SiPMhitAmplitude.  
    If SingleSiPMPulses is True, only plot the amplitudes for events where there was one
    of each SiPM pulse in the event.
    '''
    variableval = []
    numbers = []
    if SingleSiPMPulses:
        variableval = np.hstack(df.loc[((df['SiPM1NPulses']==1) & (df['SiPM2NPulses']==1)), variable].values)
        numbers = np.hstack(df.loc[((df['SiPM1NPulses']==1) & (df['SiPM2NPulses']==1)), 'SiPMNum'].values)
    else:
        variableval = np.hstack(df[variable])
        numbers = np.hstack(df['SiPMNum'])
    variableval = variableval[np.where(numbers==sipm_num)[0]]
    plt.hist(variableval,bins=ranges['bins'],range=ranges['range'])
    plt.xlabel(labels["xlabel"])
    appendage = ""
    if SingleSiPMPulses:
        appendage = "\n (Only one SiPM1 and SiPM2 pulse in an acquisition)"
    plt.title(labels["title"]+"%s"%(appendage))
    plt.show()

def MakePMTVariableDistribution(df, variable, labels, ranges, SingleSiPMPulses):
    '''
    Plot the Tank PMT variable distributions, including hitX, hitY, hitZ, and hitT, and hitQ.  
    If SingleSiPMPulses is True, only plot entries for clusters where there was one
    of each SiPM pulse in the event.
    '''
    variableval = []
    numbers = []
    if SingleSiPMPulses:
        variableval = np.hstack(df.loc[((df['SiPM1NPulses']==1) & (df['SiPM2NPulses']==1)), variable].values)
    else:
        variableval = np.hstack(df[variable])
    plt.hist(variableval,bins=ranges['bins'],range=ranges['range'])
    plt.xlabel(labels["xlabel"])
    appendage = ""
    if SingleSiPMPulses:
        appendage = "\n (Only one SiPM1 and SiPM2 pulse in an acquisition)"
    plt.title(labels["title"]+"%s"%(appendage))
    plt.show()

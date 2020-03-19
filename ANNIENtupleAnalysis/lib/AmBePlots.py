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
xkcd_colors = ['dark teal','purple','adobe']
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

def MakeClusterTimeDistribution(df,llabel):
    '''
    Plot the time distribution for all clusters in the file.
    '''
    CleanPromptDF = ValidPromptClusterEvents(df,2000)
    plt.hist(CleanPromptDF['clusterTime'],100,label=llabel,alpha=0.8)
    plt.xlabel("Cluster time (ns)")
    plt.title("Time distribution of hit clusters")

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
    plt.hist(variableval,bins=ranges['bins'],range=ranges['range'],label=labels['llabel'],alpha=0.8)
    plt.xlabel(labels["xlabel"])
    appendage = ""
    if SingleSiPMPulses:
        appendage = "\n (Only one SiPM1 and SiPM2 pulse in an acquisition)"
    plt.title(labels["title"]+"%s"%(appendage))

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
    plt.hist(variableval,bins=ranges['bins'],range=ranges['range'],label=labels['llabel'],alpha=0.8)
    plt.xlabel(labels["xlabel"])
    appendage = ""
    if SingleSiPMPulses:
        appendage = "\n (Only one SiPM1 and SiPM2 pulse in an acquisition)"
    plt.title(labels["title"]+"%s"%(appendage))

def MakeHexJointPlot(df,xvariable,yvariable,labels,ranges):
    g = sns.jointplot(x=df[xvariable],y=df[yvariable],
            kind="hex",xlim=ranges['xrange'],
            ylim=ranges['yrange'],
            joint_kws=dict(gridsize=ranges['bins']),
            stat_func=None).set_axis_labels(labels['xlabel'],labels['ylabel'])
    plt.subplots_adjust(left=0.2,right=0.8,
            top=0.90,bottom=0.2)
    cbar_ax = g.fig.add_axes([0.85,0.2,0.05,0.62])
    plt.colorbar(cax=cbar_ax)
    g.fig.suptitle(labels['title'])
    plt.show()

def MakeHexJointPlot(df,xvariable,yvariable,labels,ranges):
    g = sns.jointplot(x=df[xvariable],y=df[yvariable],
            kind="hex",xlim=ranges['xrange'],
            ylim=ranges['yrange'],
            joint_kws=dict(gridsize=ranges['bins']),
            stat_func=None).set_axis_labels(labels['xlabel'],labels['ylabel'])
    plt.subplots_adjust(left=0.2,right=0.8,
            top=0.90,bottom=0.2)
    cbar_ax = g.fig.add_axes([0.85,0.2,0.05,0.62])
    plt.colorbar(cax=cbar_ax)
    g.fig.suptitle(labels['title'])
    plt.show()

def Make2DHist(df,xvariable,yvariable,labels,ranges):
    plt.hist2d(df[xvariable],df[yvariable], bins=(ranges['xbins'],ranges['ybins']),
            range=[ranges['xrange'],ranges['yrange']],
            cmap = plt.cm.inferno)
    plt.colorbar()
    plt.title(labels['title'])
    plt.xlabel(labels['xlabel'])
    plt.ylabel(labels['ylabel'])

def MakeKDEPlot(df,xvariable,yvariable,labels,ranges):
    sns.kdeplot(df[xvariable],df[yvariable],shade=True,shadow_lowest=False, Label=labels['llabel'],
            cmap=labels['color'])
    plt.xlabel(labels['xlabel'])
    plt.ylabel(labels['ylabel'])
    plt.title(labels['title'])
    #def MakeHeatMap(df,xvariable,yvariable,labels):
#    g = sns.jointplot(xvariable,yvariable,data=df,kind="hex",xlim=ranges['xrange'],
#            ylim=ranges['yrange'],
#            joint_kws=dict(gridsize=ranges['bins']),
#            stat_func=None).set_axis_labels(labels['xlabel'],labels['ylabel'])
#    plt.subplots_adjust(left=0.2,right=0.8,
#            top=0.90,bottom=0.2)
#    cbar_ax = g.fig.add_axes([0.85,0.2,0.05,0.62])
#    plt.colorbar(cax=cbar_ax)
#    g.fig.suptitle(labels['title'])
#    plt.show()

def ShowPlot():
    leg = plt.legend(loc=4,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.show()

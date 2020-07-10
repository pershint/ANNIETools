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


DATA_DIR = "./Data/Demo/"

expoPFlat= lambda x,C1,tau,mu,B: C1*np.exp(-(x-mu)/tau) + B


def PlotDemo(Sdf,Sdf_trig):
    
    #Cuts can be applied to pandas dataframes; EventSelection.py has some cuts defined,
    #And some are directly applied here as well.  Examples of applying cuts at both the 
    #Cluster level and trigger level are shown.

    #Cluster level cuts
    Sdf_SinglePulses = es.SingleSiPMPulses(Sdf)
    Sdf_PlusCleanSiPMPrompt = es.NoPromptClusters(Sdf_SinglePulses,2000)
    Sdf_PlusNoHighPEClusters = es.NoBurstClusters(Sdf_PlusCleanSiPMPrompt,2000,150)
    Sdf_PlusGoodCB = Sdf_PlusNoHighPEClusters.loc[Sdf_PlusNoHighPEClusters['clusterChargeBalance']<0.4]
    #Trigger level cuts
    Sdf_trig_goodSiPM = Sdf_trig.loc[(Sdf_trig['SiPM1NPulses']==1) & (Sdf_trig['SiPM2NPulses']==1)].reset_index(drop=True)
    Sdf_trig_CleanPrompt = es.NoPromptClusters_WholeFile(Sdf_SinglePulses,Sdf_trig_goodSiPM,2000)
    Sdf_trig_CleanWindow = es.NoBurst_WholeFile(Sdf_PlusCleanSiPMPrompt,Sdf_trig_CleanPrompt,2000,150)

    Sdf_clean = Sdf_PlusGoodCB
    Sdf_trig_clean = Sdf_trig_CleanWindow

    #Access hit information in first cluster
    print(np.array(Sdf_clean['hitX'][0]))
    print(np.array(Sdf_clean['hitY'][0]))
    print(np.array(Sdf_clean['hitZ'][0]))
    print(np.array(Sdf_clean['hitT'][0]))
    print(np.array(Sdf_clean['hitQ'][0]))
    print(np.array(Sdf_clean['hitPE'][0]))

    #Example of how to filter and only show hits in front-half of tank
    front_hits = np.where(np.array(Sdf_clean['hitZ'][0])>0)[0]
    print(np.array(Sdf_clean['hitX'][0])[front_hits])
    print(np.array(Sdf_clean['hitY'][0])[front_hits])
    print(np.array(Sdf_clean['hitZ'][0])[front_hits])
    print(np.array(Sdf_clean['hitT'][0])[front_hits])
    print(np.array(Sdf_clean['hitPE'][0])[front_hits])

    #Access some cluster-level information for all clusters and of first cluster
    print(Sdf_clean['clusterPE'])
    print(Sdf_clean['clusterPE'][0])
    print(Sdf_clean['clusterChargeBalance'])
    print(Sdf_clean['clusterChargeBalance'][0])

    #Simple 1D histogram; plot total PE of all clusters
    plt.hist(Sdf_clean['clusterPE'],bins=30,range=(0,80),alpha=0.75,histtype='stepfilled',linewidth=6,color='blue')
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Cluster PE distribution ")
    plt.xlabel("Cluster PE")
    plt.ylabel("Number of clusters")
    plt.show()

    #Simple 1D histogram; PE distribution of all hits in all clusters
    plt.hist(np.hstack(Sdf_clean['hitPE']),bins=120,range=(0,40),alpha=0.75,histtype='stepfilled',linewidth=6,color='blue')
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Hit PE distribution")
    plt.xlabel("Hit PE")
    plt.ylabel("Number of hits")
    plt.show()

    #Simple 1D histogram; the exact same plot as above, but I think the step & stepfilled
    #Combo looks nice!
    plt.hist(np.hstack(Sdf_clean['hitPE']),bins=120,range=(0,40),histtype='step',linewidth=6,color='blue')
    plt.hist(np.hstack(Sdf_clean['hitPE']),bins=120,range=(0,40),histtype='stepfilled',linewidth=6,color='blue', alpha=0.6)
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Hit PE distribution")
    plt.xlabel("Hit PE")
    plt.ylabel("Number of hits")
    plt.show()

    #Simple 2D histogram
    labels = {'title': 'Charge balance parameter as a function of cluster PE', 
            'xlabel': 'Total PE', 'ylabel': 'Charge balance parameter'}
    ranges = {'xbins': 50, 'ybins':50, 'xrange':[0,80],'yrange':[0,1]}
    variables = {'x': 'clusterPE', 'y': 'clusterChargeBalance'}
    #abp.MakeHexJointPlot(Sdf,'clusterPE','clusterChargeBalance',labels,ranges)
    plt.hist2d(Sdf_clean[variables['x']],Sdf_clean[variables['y']], bins=(ranges['xbins'],ranges['ybins']),
            range=[ranges['xrange'],ranges['yrange']],
            cmap = plt.cm.inferno)
    plt.colorbar()
    plt.title(labels['title'])
    plt.xlabel(labels['xlabel'])
    plt.ylabel(labels['ylabel'])
    plt.show()


    #Example 2D histogram

if __name__=='__main__':
    slist = glob.glob(DATA_DIR+"*.ntuple.root")

    livetime_estimate = es.EstimateLivetime(slist)
    print("SIGNAL LIVETIME ESTIMATE IN SECONDS IS: " + str(livetime_estimate))

    #Define which values to get from the phaseIITankClusterTree
    #These are also used to name the labels in the pandas dataframe
    mybranches = ['eventNumber','eventTimeTank','clusterTime','SiPMhitT','SiPMhitQ','SiPMhitAmplitude','clusterChargeBalance','clusterPE','SiPM1NPulses','SiPM2NPulses','SiPMNum','clusterHits','hitX',
            'hitY','hitZ','hitT','hitQ','hitPE','hitDetID']
    
    #Load data from the cluster-level event tree
    SProcessor = rp.ROOTProcessor(treename="phaseIITankClusterTree")
    for f1 in slist:
        SProcessor.addROOTFile(f1,branches_to_get=mybranches)
    Sdata = SProcessor.getProcessedData()
    Sdf = pd.DataFrame(Sdata)

    #Load data from the trigger-level event tree
    SProcessor = rp.ROOTProcessor(treename="phaseIITriggerTree")
    for f1 in slist:
        SProcessor.addROOTFile(f1,branches_to_get=mybranches)
    Sdata = SProcessor.getProcessedData()
    Sdf_trig = pd.DataFrame(Sdata)

    PlotDemo(Sdf,Sdf_trig)



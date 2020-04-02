import glob

import sys
import uproot
import lib.ROOTProcessor as rp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from . import EventSelection as es

import pandas as pd

sns.set_context('poster')
sns.set(font_scale=2.5)
sns.set_style("whitegrid")
sns.axes_style("darkgrid")
xkcd_colors = ['adobe','red','dark teal','purple']
#xkcd_colors = ['dark red' for x in range(100)]
sns.set_palette(sns.xkcd_palette(xkcd_colors))

def EstimateEnergyPerClusterRelation(visible_energy,event_times,df_clusters,minE, maxE, nbins):
    '''
    Estimate the mean number of delayed clusters observed per interaction as a function of energy.
    '''
    cluster_multiplicity = []
    for time in event_times:
        num_clusters = len(np.where(df_clusters["eventTimeTank"].values == time)[0])
        cluster_multiplicity.append(num_clusters)
    #Neat.  Get the mean of cluster counts for energy ranges.
    cluster_multiplicity = np.array(cluster_multiplicity)
    Energy_bins = np.arange(minE,maxE + (maxE-minE)/nbins,(maxE-minE)/nbins)
    average_multiplicities = []
    sigma_multiplicities = []
    for j,e in enumerate(Energy_bins):
        if j == 0:
            continue
        indices_thisbin = np.where((visible_energy>=Energy_bins[j-1]) & (visible_energy<Energy_bins[j]))[0]
        multiplicity_thisbin = cluster_multiplicity[indices_thisbin]
        avg_multiplicity = np.average(multiplicity_thisbin)
        average_multiplicities.append(avg_multiplicity)
        std_multiplicity = np.std(multiplicity_thisbin)
        sigma_multiplicities.append(std_multiplicity)
    Energy_bin_lefts = Energy_bins[0:len(Energy_bins)-1]
    return Energy_bin_lefts,np.array(average_multiplicities),np.array(sigma_multiplicities)

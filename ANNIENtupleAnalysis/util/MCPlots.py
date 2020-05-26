#This script has functions needed to make a Cumulative Distribution Plot from
#Different variables output in  PhaseIITreeMaker root file.

import glob

import sys
import lib.HistogramUtils as hu
import lib.HistGrabber as hg

import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as scp
import numpy as np
import seaborn as sns

sns.set_context('poster')
sns.set(font_scale=2.5)
sns.set_style("whitegrid")
sns.axes_style("darkgrid")

MCNFILE = "./Data/MCProfiles/Analyzer_AmBe_Housing_Center-0-0-0_100k_Nogammas.root"
MCPFILE = "./Data/MCProfiles/Analyzer_AmBe_Housing_Center-0-0-0_100k_Onlygammas.root"
HISTBASE = "h_HitsDelayed_withEdep_min"
HISTENDS = ["250keV","500keV","1MeV","2MeV","3MeV","4MeV"]

SIGNAL_DIR = "./Data/V3_5PE100ns/CentralData/"
#BKG_DIR = "./Data/BkgCentralData/"
BKG_DIR = "./Data/V3_5PE100ns/BkgCentralData/"


for h in HISTENDS:
    mcbins,mclefts = hg.GetHist(MCNFILE,HISTBASE+h)
    ourrange = np.where(mclefts<150)[0]
    plt.errorbar(mclefts[ourrange],mcbins[ourrange],linewidth=3,marker='o',markersize=11,label=">%s"%(h))
leg = plt.legend(loc=1,fontsize=24)
leg.set_frame_on(True)
leg.draw_frame(True)
plt.title("Tank PE distribution for increasing BGO energy thresholds \n (100k AmBe neutrons, RATPAC simulation)")
plt.ylabel("Events")
plt.xlabel("Total PE")
plt.show()

for h in HISTENDS:
    mcbins,mclefts = hg.GetHist(MCPFILE,HISTBASE+h)
    ourrange = np.where(mclefts<150)[0]
    plt.errorbar(mclefts[ourrange],mcbins[ourrange],linewidth=3,marker='o',markersize=11,label=">%s"%(h))
leg = plt.legend(loc=1,fontsize=24)
leg.set_frame_on(True)
leg.draw_frame(True)
plt.title("Tank PE distribution for increasing BGO energy thresholds \n (100k AmBe gammas, RATPAC simulation)")
plt.ylabel("Events")
plt.xlabel("Total PE")
plt.show()

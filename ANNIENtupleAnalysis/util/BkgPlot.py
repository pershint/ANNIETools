# coding: utf-8

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
    
sns.set_context("poster")
sns.set(font_scale=2.5)
sns.set_style("whitegrid")
sns.axes_style("darkgrid")
def myfunction():
    rate = [0.055,0.051,0.045]
    vert_rate = []
    vert_radius = [102]
    pos_vunc = []
    neg_vunc = []
    radius = [0, 75, 102]
    pos_unc = [0.010, 0.008, 0.010]
    neg_unc = [0.011, 0.012, 0.011]
    plt.errorbar(radius,rate, yerr=[neg_unc,pos_unc], marker='o',color='orange',label='$y = 0 \, cm$',markersize=12,elinewidth=4)
    plt.errorbar(vert_radius,vert_rate, yerr=[neg_vunc,pos_vunc], marker='o',color='teal',linestyle='None',label='$y = +100 \, cm$',markersize=12,elinewidth=4)
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Best-fit background rate $\lambda_{n}$ as background varies")
    plt.xlabel("Radius [cm]")
    plt.ylabel("Background rate fit $\lambda_{n}$ [candidates/trigger]")
    plt.show()
    
myfunction()

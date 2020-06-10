# coding: utf-8

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
    
sns.set_context("poster")
sns.set(font_scale=2.5)
sns.set_style("whitegrid")
sns.axes_style("darkgrid")
def myfunction():
    ### 5 HIT, 100 NS CLUSTER VALUES ####
    eff = [0.64,0.57,0.46]
    vert_eff = [0.35]
    vert_radius = [102]
    pos_vunc = [np.sqrt(0.03**2 + 0.01**2)]
    neg_vunc = [0.02]
    radius = [0, 75, 102]
    pos_unc = [np.sqrt(0.03**2 + 0.01**2), np.sqrt(0.03**2 + 0.01**2), np.sqrt(0.02**2 + 0.01**2)]
    neg_unc = [0.02,0.02,np.sqrt(0.03**2 + 0.01**2)]
    plt.errorbar(vert_radius,vert_eff, yerr=[neg_vunc,pos_vunc], marker='o',color='red',linestyle='None',label='$y = +100 \, cm$',markersize=12,elinewidth=4)
    plt.errorbar(radius,eff, yerr=[neg_unc,pos_unc], marker='o',color='blue',label='$y = 0 \, cm$',markersize=12,linestyle='None',elinewidth=4)
    leg = plt.legend(loc=1,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Neutron detection efficiency as detector radius varies")
    plt.xlabel("Radius [cm]")
    plt.ylabel("Neutron detection efficiency $\epsilon_{n}$")
    plt.show()
    
myfunction()

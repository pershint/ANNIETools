import scipy.optimize as scp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import lib.TMLoader as tm

sns.set_context('poster')
sns.set(font_scale=2.0)
sns.set_style("whitegrid")
sns.axes_style("darkgrid")

#gauss1= lambda x,C1,m1,s1: C1*(1./(s1*np.sqrt(2*np.pi)))*np.exp(-(1./2.)*((x-m1)**2)/s1**2)
gauss1= lambda x,C1,m1,s1: C1*np.exp(-(1./2.)*((x-m1)**2)/s1**2)

DATADIR = "./data/"

if __name__ == '__main__':
    mydata = tm.LoadMuonInfoInDir(DATADIR)
    plt.hist(mydata["TotPE"],bins=40)
    plt.xlabel("Total PE")
    plt.ylabel("Events")
    plt.title("Total photoelectron distribution for \n through-going muon candidates")
    bin_edges = np.arange(0,np.max(mydata["TotPE"]),150)
    print(bin_edges)
    bin_centers = []
    for j in range(len(bin_edges)):
        if j==0: continue
        bin_centers.append(bin_edges[j-1] + ((bin_edges[j] - bin_edges[j-1])/2))
    bin_centers = np.array(bin_centers)
    print(len(bin_centers))
    hist,bins = np.histogram(mydata["TotPE"],bins=bin_edges)
    print(hist)
    plt.show()
    init_params = [40,4200,400]
    popt, pcov = scp.curve_fit(gauss1, bin_centers, hist, p0=init_params, maxfev=6000)
    print(popt)
    print(pcov)
    myy = gauss1(bin_centers,popt[0],popt[1],popt[2])
    plt.hist(mydata["TotPE"],bins=40)
    plt.xlabel("Total PE")
    plt.ylabel("Events")
    plt.title("Total photoelectron distribution for \n through-going muon candidates")
    plt.plot(bin_centers,myy,marker='None',linewidth=6,label='Best fit')
    plt.show()


    cut_energies = mydata[(mydata.Depth > 50) & (mydata.TrackAngle<0.4) & (np.sqrt(mydata.EntryX**2 + mydata.EntryY**2)<0.6)].TotPE.values

    bin_edges = np.arange(0,np.max(cut_energies),150)
    print(bin_edges)
    bin_centers = []
    for j in range(len(bin_edges)):
        if j==0: continue
        bin_centers.append(bin_edges[j-1] + ((bin_edges[j] - bin_edges[j-1])/2))
    bin_centers = np.array(bin_centers)
    print(len(bin_centers))
    hist,bins = np.histogram(cut_energies,bins=bin_edges)
    print(hist)
    plt.show()
    init_params = [40,4200,400]
    popt, pcov = scp.curve_fit(gauss1, bin_centers, hist, p0=init_params, maxfev=6000)
    print(popt)
    print(pcov)
    myy = gauss1(bin_centers,popt[0],popt[1],popt[2])
    plt.hist(cut_energies,bins=30)
    plt.plot(bin_centers,myy,marker='None',linewidth=6,label='Best fit')
    plt.xlabel("Total PE")
    plt.ylabel("Events")
    plt.title("Total photoelectron distribution for through-going muon candidates, aggressive cuts")
    plt.show()


    g = sns.jointplot("TotPE","EntryR",data=mydata,kind="hex",
            joint_kws=dict(gridsize=15),
            stat_func=None).set_axis_labels("Total PE","Entry distance from MRD center")
    plt.subplots_adjust(left=0.2,right=0.8,
            top=0.90,bottom=0.2)
    cbar_ax = g.fig.add_axes([0.85,0.2,0.05,0.62])
    plt.colorbar(cax=cbar_ax)
    g.fig.suptitle("Total PE observed for varying entry distance from MRD center")
    plt.show()


    g = sns.jointplot("TotPE","TrackAngle",data=mydata,kind="hex",
            joint_kws=dict(gridsize=15),
            stat_func=None).set_axis_labels("Total PE","Track angle from beam axis (rad)")
    plt.subplots_adjust(left=0.2,right=0.8,
            top=0.90,bottom=0.2)
    cbar_ax = g.fig.add_axes([0.85,0.2,0.05,0.62])
    plt.colorbar(cax=cbar_ax)
    g.fig.suptitle("Total PE observed for varying track angles in MRD")
    plt.show()

    fig,ax = plt.subplots()
    ax.plot(mydata["TotPE"],mydata["TrackAngle"],alpha=0.8,linestyle='None',marker='o',markersize=9)
    plt.xlabel("Total PE")
    plt.ylabel("Track angle from beam axis")
    plt.title("Comparison of Total PE deposited in tank to track angle in MRD")
    plt.show()

    fig,ax = plt.subplots()
    ax.plot(mydata["TotPE"],mydata["Depth"],alpha=0.8,linestyle='None',marker='o',markersize=9)
    plt.xlabel("Total PE")
    plt.ylabel("Penetration depth (cm)")
    plt.title("Comparison of Total PE deposited in tank to penetration depth in MRD")
    plt.show()

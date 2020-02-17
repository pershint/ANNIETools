import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster")
sns.set(font_scale=1.4)

def GetPMTMRDTriggerInfo(filepath):
    PMTTimestamps = []
    MRDTimestamps = {"Timestamp": [], "TriggerType": []}
    datalines = []
    with open(filepath,"r") as f:
        datalines = f.readlines()
    for line in datalines:
        if line.find("STAMP") == -1:
            continue
        dataline = line.split(",")
        if line.find("MRD") != -1:
            MRDTimestamps["Timestamp"].append(int(dataline[1]))
            MRDTimestamps["TriggerType"].append(dataline[2].rstrip("\n"))
        if line.find("TANK") != -1:
            PMTTimestamps.append(int(dataline[1].rstrip("\n")))
    return PMTTimestamps, MRDTimestamps

if __name__ == "__main__":
    print("MAKING TRIGGER TIMESTAMP PLOTS")
    PMT,MRD = GetPMTMRDTriggerInfo("thelog1347_Stamps.log")
    PMTdat = np.array(PMT)
    MRDdat = np.array(MRD["Timestamp"])
    CosmicInds = []
    BeamInds = []
    print(MRD["TriggerType"])
    for j,ts in enumerate(MRD["TriggerType"]):
        if ts == "Cosmic":
            CosmicInds.append(j)
        elif ts == "Beam":
            BeamInds.append(j)
    print(CosmicInds)
    Cosmicdat = MRDdat[np.array(CosmicInds)]
    Beamdat = MRDdat[np.array(BeamInds)]
    PMTdat = np.sort(PMTdat)
    PMTdat = (PMTdat/1.0E6) - 21600000
    Cosmicdat = np.sort(Cosmicdat)
    pmtmin = PMTdat.min()
    pmtmax = PMTdat.max()
    Beamdat = np.sort(Beamdat)
    Combineddat = np.concatenate((PMTdat,MRDdat))
    hmin = Combineddat.min()
    hmax = Combineddat.max()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    #plt.hist(PMTdat,bins = int(hmax-hmin),range=(hmin,hmax),label="PMT Triggers")
    #plt.hist(Cosmicdat,bins = int(hmax-hmin),range=(hmin,hmax),label="Cosmic MRD Triggers")
    #plt.hist(Beamdat,bins = int(hmax-hmin),range=(hmin,hmax),label="Beam MRD Triggers")
    plt.hist(PMTdat,bins = 10000,range=(hmax-20000,hmax-10000),label="PMT Triggers",linewidth=50,alpha=0.5,color='blue')
    plt.hist(Beamdat,bins = 10000,range=(hmax-20000,hmax-10000),label="Beam Triggers",linewidth=50,alpha=0.5,color='red')
    plt.hist(Cosmicdat,bins = 10000,range=(hmax-20000,hmax-10000),label="Cosmic Triggers",linewidth=50,alpha=0.5,color='green')
    #plt.hist(Cosmicdat,bins = int(hmax-hmin),range=(hmin,hmax),label="Cosmic MRD Triggers")
    #plt.hist(Beamdat,bins = int(hmax-hmin),range=(hmin,hmax),label="Beam MRD Triggers")
    leg = ax.legend(loc=4,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.xlabel("Timestamps in UTC (ms)")
    plt.ylabel("Number of triggers")
    plt.title("Timestamps in run 1347 for PMT and MRD subsystems")
    plt.show()

    MRDsync = np.where((Beamdat>(pmtmin-5)) & (Beamdat<(pmtmax+15)))[0]
    MRDPairing = MRDdat[MRDsync]
    print("TOTAL TIME CONSIDERED (ms): " + str(pmtmax-pmtmin))
    print("LEN OF MRD  IN PMT WINDOW: " + str(len(MRDPairing)))
    print("LEN OF PMT EVENTS: " + str(len(PMTdat)))

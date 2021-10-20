import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import numpy as np
import math

from . import Detector as d

sns.set_context('poster')
sns.set(font_scale=2.0)
sns.set_style("whitegrid")
sns.axes_style("darkgrid")

REFERENCE_TUBE = 381
VC = 0.2998 #meters per nanosecond
n = 1.34  #index of refraction in water

def EstimateVisualizeRelativeDelays(DetectorFile,PeakFitDict):
    PMTGetter = d.TankPMTLoader()
    PMTGetter.ParseTankGeoFile(DetectorFile)
    pmtpos = PMTGetter.GetPMTPositions()
    LEDToPMTKey = {3: 382, 0:405, 1:344, 2:404, 4:456, 5:393}
    ledpos = PMTGetter.GetApproxLEDPositions(LEDToPMTKey)
    print("PMT POSNS: " + str(pmtpos))
    print("LED POSNS: " + str(ledpos))

    dat = PeakFitDict
    df = pd.DataFrame(dat)
    TUBES = np.array(list(set(df["channel"])))
    TUBES = np.arange(332,464,1)
    fig,ax = plt.subplots()
    LEDs = np.arange(0,6,1)
    channel_delays = {}
    for led in LEDs:
        ref_pos = np.linalg.norm(np.array(pmtpos[REFERENCE_TUBE]) - np.array(ledpos[led]))
        print("REFERENCE POS: " + str(ref_pos))
        #ref_time = df[(df.LED==led) & (df.channel == REFERENCE_TUBE)].mu.values -(ref_pos*n/VC) #CORRECT FOR LIGHT PROPAGATION TIME
        ref_time = df[(df.LED==led) & (df.channel == REFERENCE_TUBE)].mu.values[0]
        ref_mean = df[(df.LED==led) & (df.channel == REFERENCE_TUBE)].mu.values[0]
        print("REF TIME: " + str(ref_time))
        myx = []
        myy = []
        myxunc = []
        for cnum in TUBES:
            if cnum not in list(df["channel"]):
                print("DID NOT FIND ANY DATA FOR CHANNELNUM %i"%(cnum))
                continue
            delay = df[((df["LED"] == led) & (df["channel"]==cnum))].mu.values - ref_time
            #TRIES TO CORRECT FOR LIGHT PROPAGATION TIME
            #thispmt_transittime = np.linalg.norm(np.array(pmtpos[cnum]) - np.array(ledpos[led]))*n/VC
            #delay = df[((df["LED"] == led) & (df["channel"]==cnum))].mu.values -thispmt_transittime - ref_time
            if(len(delay)<1):
                continue
            delay_unc = df[((df["LED"] == led) & (df["channel"]==cnum))].mu_unc.values
            myx.append(delay[0])
            myy.append(cnum)
            myxunc.append(delay_unc[0])
            if len(delay)==0: 
                continue
            if cnum not in channel_delays:
                channel_delays[cnum] = [delay]
            else:
                channel_delays[cnum].append(delay)
        ax.errorbar(np.array(myx),np.array(myy),xerr=np.array(myxunc),alpha=0.8,label="LED %i Delays"%(led),linestyle='None',marker='o',markersize=6)
    
    #Now, calculate each channel's mean delay relative to the reference tube
    Earliest_mean = 999999
    for channel in channel_delays:
        mean = np.average(channel_delays[channel])
        if mean < Earliest_mean:
            Earliest_mean = mean
    final_channels = []
    final_delays = []
    final_delay_uncs = []
    for channel in channel_delays:
        print("CHANNEL: " + str(channel))
        final_channels.append(channel)
        #print("MEAN,%i,%f"%(channel,np.average(channel_delays[channel]) - Earliest_mean))
        #final_delays.append(np.average(channel_delays[channel]) - Earliest_mean)
        print("MEAN,%i,%f"%(channel,np.average(channel_delays[channel]) - ref_mean))
        final_delays.append(np.average(channel_delays[channel]) - ref_mean)
        print("STD,%i,%f"%(channel,np.std(channel_delays[channel])))
        final_delay_uncs.append(np.std(channel_delays[channel]))
    #ax.bar(y = myy,x = range(len(myx)), yerr = myyerr,label = cnum)
    #ax.xticks(range(len(myx)),myx)
    leg = plt.legend(loc=2,fontsize=15)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    ax.set_xlabel("Time delay (ns)") 
    ax.set_ylabel("Channel Key") 
    plt.title(("LED arrival times relative to PMT %i"%(REFERENCE_TUBE)))
    plt.show()



    fig,ax = plt.subplots()
    for channel in channel_delays:

        ax.errorbar(np.average(channel_delays[channel]) - Earliest_mean,
                    channel, xerr = np.std(channel_delays[channel]),
                    alpha=0.8,linestyle='None',marker='o',markersize=6)
    ax.set_xlabel("Mean of calculated LED delays (ns)") 
    ax.set_ylabel("Channel Key") 
    plt.title(("Mean of all 6 LED arrival times to each PMT"))
    plt.show()

    CableDelays = {}
    CableDelayUncs = {}
    for j,cnum in enumerate(final_channels):
        CableDelays[str(cnum)] = final_delays[j]
        CableDelayUncs[str(cnum)] = final_delay_uncs[j]
    with open("./output/RelativeDelays.json","w") as f:
        json.dump(CableDelays,f,sort_keys=True,indent=4)
    with open("./output/RelativeDelayUncs.json","w") as fu:
        json.dump(CableDelayUncs,fu,sort_keys=True,indent=4)

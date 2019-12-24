import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import numpy as np
import math

if __name__=='__main__':
    with open("PeakFitResults.json","r") as f:
        dat = json.load(f)
    df = pd.DataFrame(dat)
    TUBES = np.array(list(set(df["channel"])))
    TUBES = np.arange(332,460,1)
    #for cnum in TUBES:
    #    if cnum not in list(df["channel"]):
    #        print("DID NOT FIND ANY DATA FOR CHANNELNUM %i"%(cnum))
    #        continue
    mintime = np.min(df["mu"])
    fig,ax = plt.subplots()
    myx = df.loc[((df["LED"] == 0)), "mu"] - mintime
    myxunc = df.loc[((df["LED"] == 0)), "mu_unc"]
    myy = df.loc[((df["LED"] == 0)), "channel"]
    ax.errorbar(myx,myy,xerr=myxunc,alpha=0.8,label="LED 0 Delays",linestyle='None',marker='o',markersize=6)
    #ax.bar(y = myy,x = range(len(myx)), yerr = myyerr,label = cnum)
    #ax.xticks(range(len(myx)),myx)
    print(myx)
    print(myy)
    ax.set_xlabel("Time delay relative to earliest (ns)") 
    ax.set_ylabel("Channel Key") 
    plt.title(("Mean LED arrival time relative to earliest mean"))
    plt.show()

    CableDelays = {}
    CableDelayUncs = {}
    myy = np.array(myy)
    myx = np.array(myx)
    myxunc = np.array(myxunc)
    for j,cnum in enumerate(myy):
        print(cnum)
        print(j)
        try:
            print(myx[j])
        except KeyError:
            print("CHANNEL %s IS FUCKED"%(cnum))
            continue
        CableDelays[str(cnum)] = myx[j]
        CableDelayUncs[str(cnum)] = myxunc[j]
    with open("RelativeDelays.json","w") as f:
        json.dump(CableDelays,f,sort_keys=True,indent=4)
    with open("RelativeDelayUncs.json","w") as fu:
        json.dump(CableDelayUncs,fu,sort_keys=True,indent=4)

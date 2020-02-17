import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as scp

sns.set_context("poster")


THEFILE = "./DB/GainFitStatus_V0.csv"

def line(x,m,b):
    return m*x + b

def ParseFile(infile):
    with open(infile,"r") as f:
        lines = f.readlines()
        data = {"Channel":[], "TestStandGain":[],
                "Fit1E7":[], "Fit5E6":[]}
        for j,line in enumerate(lines):
            if line.find("-9999")!=-1: continue
            if j == 0: continue
            sline = line.split(",")
            data["Channel"].append(int(sline[0]))
            data["TestStandGain"].append(int(sline[3]))
            data["Fit1E7"].append(int(sline[4]))
            data["Fit5E6"].append(int(sline[5]))
    return data

if __name__=="__main__":
    LUX_TUBES = np.arange(332,352,1)
    ETEL_TUBES = np.arange(352,372,1)
    WM_TUBES = np.array([382,393,404,405])
    ANNIE_TUBES = np.arange(372,416,1)
    ANNIE_TUBES = np.setdiff1d(ANNIE_TUBES,WM_TUBES)
    print("ANNIE_TUBES: " + str(ANNIE_TUBES))
    WB_TUBES = np.arange(416,464,1)
    print(WB_TUBES)
    mydata = ParseFile(THEFILE)
    df = pd.DataFrame(mydata)
    myx_lux = []
    myy_lux = []
    myy_lux6 = []
    myx_etel = []
    myy_etel = []
    myy_etel6 = []
    myx_annie = []
    myy_annie = []
    myy_annie6 = []
    myx_wm = []
    myy_wm = []
    myy_wm6 = []
    myx_wb = []
    myy_wb = []
    myy_wb6 = []


    for cnum in LUX_TUBES:
        if cnum not in list(df["Channel"]):
            print("DID NOT FIND ANY DATA FOR CHANNELNUM %i"%(cnum))
            continue
        myx_lux.append(df[((df.Channel==cnum) & (df["Fit1E7"]!=-9999) & (df["TestStandGain"]!=-9999))].TestStandGain.item()) 
        myy_lux.append(df[((df.Channel==cnum) & (df["Fit1E7"]!=-9999) & (df["TestStandGain"]!=-9999))].Fit1E7.item()) 
        myy_lux6.append(df[((df.Channel==cnum) & (df["Fit1E7"]!=-9999) & (df["TestStandGain"]!=-9999))].Fit5E6.item()) 
    
    for cnum in ETEL_TUBES:
        if cnum not in list(df["Channel"]):
            print("DID NOT FIND ANY DATA FOR CHANNELNUM %i"%(cnum))
            continue
        myx_etel.append(df[((df.Channel==cnum) & (df["Fit1E7"]!=-9999) & (df["TestStandGain"]!=-9999))].TestStandGain.item()) 
        myy_etel.append(df[((df.Channel==cnum) & (df["Fit1E7"]!=-9999) & (df["TestStandGain"]!=-9999))].Fit1E7.item()) 
        myy_etel6.append(df[((df.Channel==cnum) & (df["Fit1E7"]!=-9999) & (df["TestStandGain"]!=-9999))].Fit5E6.item()) 
    
    for cnum in ANNIE_TUBES:
        if cnum not in list(df["Channel"]):
            print("DID NOT FIND ANY DATA FOR CHANNELNUM %i"%(cnum))
            continue
        myx_annie.append(df[((df.Channel==cnum) & (df["Fit1E7"]!=-9999) & (df["TestStandGain"]!=-9999))].TestStandGain.item()) 
        myy_annie.append(df[((df.Channel==cnum) & (df["Fit1E7"]!=-9999) & (df["TestStandGain"]!=-9999))].Fit1E7.item()) 
        myy_annie6.append(df[((df.Channel==cnum) & (df["Fit1E7"]!=-9999) & (df["TestStandGain"]!=-9999))].Fit5E6.item()) 

    for cnum in WM_TUBES:
        if cnum not in list(df["Channel"]):
            print("DID NOT FIND ANY DATA FOR CHANNELNUM %i"%(cnum))
            continue
        print(df[((df.Channel==cnum) & (df["Fit1E7"]!=-9999) & (df["TestStandGain"]!=-9999))].TestStandGain.item())
        myx_wm.append(df[((df.Channel==cnum) & (df["Fit1E7"]!=-9999) & (df["TestStandGain"]!=-9999))].TestStandGain.item()) 
        myy_wm.append(df[((df.Channel==cnum) & (df["Fit1E7"]!=-9999) & (df["TestStandGain"]!=-9999))].Fit1E7.item()) 
        myy_wm6.append(df[((df.Channel==cnum) & (df["Fit1E7"]!=-9999) & (df["TestStandGain"]!=-9999))].Fit5E6.item()) 

    for cnum in WB_TUBES:
        if cnum not in list(df["Channel"]):
            print("DID NOT FIND ANY DATA FOR CHANNELNUM %i"%(cnum))
            continue
        myx_wb.append(df[((df.Channel==cnum) & (df["Fit1E7"]!=-9999) & (df["TestStandGain"]!=-9999))].TestStandGain.item()) 
        myy_wb.append(df[((df.Channel==cnum) & (df["Fit1E7"]!=-9999) & (df["TestStandGain"]!=-9999))].Fit1E7.item()) 
        myy_wb6.append(df[((df.Channel==cnum) & (df["Fit1E7"]!=-9999) & (df["TestStandGain"]!=-9999))].Fit5E6.item()) 


    # Plot voltages comparing python script and teststand voltages
    fig,ax = plt.subplots()
    ax.plot(myx_lux,myy_lux,alpha=0.8,linestyle='None',marker='o',markersize=9,label="LUX")
    ax.plot(myx_etel,myy_etel,alpha=0.8,linestyle='None',marker='o',markersize=9,label="ETEL")
    ax.plot(myx_annie,myy_annie,alpha=0.8,linestyle='None',marker='o',markersize=9,label="ANNIE")
    ax.plot(myx_wm,myy_wm,alpha=0.8,linestyle='None',marker='o',markersize=9,label="WATCHMAN")
    ax.plot(myx_wb,myy_wb,alpha=0.8,linestyle='None',marker='o',markersize=9,label="WATCHBOY")
    x = np.arange(0,2000,1)
    y = np.arange(0,2000,1)
    ax.plot(x,y,marker="None",color='black')
    leg = ax.legend(loc=2,fontsize=15)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.xlabel("Teststand 1E7 Fit (V)")
    plt.ylabel("Python 1E7 fit (V)")
    plt.title("Comparison of python-based and teststand-based 1E7 gain voltage")
    plt.show()

    fig,ax = plt.subplots()
    ax.plot(myx_lux,myy_lux6,alpha=0.8,linestyle='None',marker='o',markersize=9,label="LUX")
    ax.plot(myx_etel,myy_etel6,alpha=0.8,linestyle='None',marker='o',markersize=9,label="ETEL")
    ax.plot(myx_annie,myy_annie6,alpha=0.8,linestyle='None',marker='o',markersize=9,label="ANNIE")
    ax.plot(myx_wm,myy_wm6,alpha=0.8,linestyle='None',marker='o',markersize=9,label="WATCHMAN")
    ax.plot(myx_wb,myy_wb6,alpha=0.8,linestyle='None',marker='o',markersize=9,label="WATCHBOY")
    x = np.arange(0,2000,1)
    y = np.arange(0,2000,1)
    ax.plot(x,y,marker="None",color='black')

    leg = ax.legend(loc=2,fontsize=15)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.xlabel("Teststand 1E7 Fit (V)")
    plt.ylabel("Python 5E6 fit (V)")
    plt.title("Comparison of python-based 5E6 and teststand-based 1E7 gain voltage")
    plt.show()

    #Fit a line to the ANNIE, LUX, and ETEL tubes.
    popt,pcov = scp.curve_fit(line, myx_lux, myy_lux6)
    print("BEST FIT FOR LUX S2: " + str(line(1239.0,popt[0],popt[1])))
    print("BEST FIT FOR LUX N5: " + str(line(1747.0,popt[0],popt[1])))
    print("BEST FIT FOR LUX W1: " + str(line(1316.0,popt[0],popt[1])))
    print("BEST FIT FOR LUX E5: " + str(line(1716.0,popt[0],popt[1])))
    # Plot voltages comparing python script and teststand voltages
    fig,ax = plt.subplots()
    ax.plot(myx_lux,myy_lux6,alpha=0.8,linestyle='None',marker='o',markersize=9,label="LUX")
    x = np.arange(0,2200,1)
    x = np.arange(0,2000,1)
    y = np.arange(0,2000,1)
    ax.plot(x,y,marker="None",color='black',label = 'No correction')
    ax.plot(x,line(x,popt[0],popt[1]),marker='None',label = 'LUX best fit')
    leg = ax.legend(loc=2,fontsize=15)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.xlabel("Teststand 1E7 Fit (V)")
    plt.ylabel("Python 5E6 fit (V)")
    plt.title("Best-fit correction from teststand 1E7 to python-based 5E6 gain voltage")
    plt.show()

    #Fit a line to the ANNIE, LUX, and ETEL tubes.
    popt,pcov = scp.curve_fit(line, myx_etel, myy_etel6)
    # Plot voltages comparing python script and teststand voltages
    print("BEST FIT FOR ETEL 719: " + str(line(1707.0,popt[0],popt[1])))
    print("BEST FIT FOR ETEL 134: " + str(line(1811.0,popt[0],popt[1])))
    print("BEST FIT FOR ETEL 127: " + str(line(1403.0,popt[0],popt[1])))
    print("BEST FIT FOR ETEL 715: " + str(line(1463.0,popt[0],popt[1])))
    fig,ax = plt.subplots()
    ax.plot(myx_etel,myy_etel6,alpha=0.8,linestyle='None',marker='o',markersize=9,label="ETEL")
    x = np.arange(0,2200,1)
    x = np.arange(0,2000,1)
    y = np.arange(0,2000,1)
    ax.plot(x,y,marker="None",color='black',label = 'No correction')
    ax.plot(x,line(x,popt[0],popt[1]),marker='None',label = 'ETEL best fit')
    leg = ax.legend(loc=2,fontsize=15)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.xlabel("Teststand 1E7 Fit (V)")
    plt.ylabel("Python 5E6 fit (V)")
    plt.title("Best-fit correction from teststand 1E7 to python-based 5E6 gain voltage")
    plt.show()

    #Fit a line to the ANNIE, LUX, and ETEL tubes.
    popt,pcov = scp.curve_fit(line, myx_annie, myy_annie6)
    print("BEST FIT FOR ANNIE0366: " + str(line(1560.0,popt[0],popt[1])))
    # Plot voltages comparing python script and teststand voltages
    fig,ax = plt.subplots()
    ax.plot(myx_annie,myy_annie6,alpha=0.8,linestyle='None',marker='o',markersize=9,label="ANNIE")
    x = np.arange(0,2200,1)
    x = np.arange(0,2000,1)
    y = np.arange(0,2000,1)
    ax.plot(x,y,marker="None",color='black',label = 'No correction')
    ax.plot(x,line(x,popt[0],popt[1]),marker='None',label = 'ANNIE best fit')
    leg = ax.legend(loc=2,fontsize=15)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.xlabel("Teststand 1E7 Fit (V)")
    plt.ylabel("Python 5E6 fit (V)")
    plt.title("Best-fit correction from teststand 1E7 to python-based 5E6 gain voltage")
    plt.show()

    #Fit a line to the ANNIE, LUX, and ETEL tubes.
    popt,pcov = scp.curve_fit(line, myx_wb, myy_wb6)
    print("BEST FIT FOR WB 42: " + str(line(1425.0,popt[0],popt[1])))
    # Plot voltages comparing python script and teststand voltages
    fig,ax = plt.subplots()
    ax.plot(myx_wb,myy_wb6,alpha=0.8,linestyle='None',marker='o',markersize=9,label="WATCHBOY")
    x = np.arange(0,2200,1)
    x = np.arange(0,2000,1)
    y = np.arange(0,2000,1)
    ax.plot(x,y,marker="None",color='black',label = 'No correction')
    ax.plot(x,line(x,popt[0],popt[1]),marker='None',label = 'WATCHBOY best fit')
    leg = ax.legend(loc=2,fontsize=15)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.xlabel("Teststand 1E7 Fit (V)")
    plt.ylabel("Python 5E6 fit (V)")
    plt.title("Best-fit correction from teststand 1E7 to python-based 5E6 gain voltage")
    plt.show()

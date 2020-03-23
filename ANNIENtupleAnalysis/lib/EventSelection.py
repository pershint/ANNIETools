import numpy as np
import copy

#Some methods for returning a dataframe cleaned based on event selection

def SingleSiPMPulses(df):
    '''
    Return a dataframe with events that only have one SiPM pulse in each SiPM
    '''
    newdf = df.loc[(df["SiPM1NPulses"]==1) & (df["SiPM2NPulses"]==1)]
    newdf.reset_index(drop=True)
    return newdf

def SingleSiPMPulsesDeltaT(df,TimeThreshold):
    '''
    Return a dataframe with events that only have one SiPM pulse in each SiPM,
    where the pulse peaks are within the TimeThreshold difference.
    '''
    DirtyTriggers = []
    for j in df.index.values:  #disgusting...
        TwoPulses = False
        if df["SiPM1NPulses"][j]!=1 or df["SiPM1NPulses"][j]!=1:
            DirtyTriggers.append(df["eventTimeTank"][j])
        else:
            if TwoPulses:
                if TwoPulses and abs(df["SiPMhitT"][j][0] - df["SiPMhitT"][1]) > TimeThreshold:
                    DirtyTriggers.append(df["eventTimeTank"][j])
    CleanIndices = []
    for j in df.index.values:  #disgusting...
        if df["eventTimeTank"][j] not in DirtyTriggers:
            CleanIndices.append(j)
    CleanIndices = np.array(CleanIndices)
    newdf = df.loc[CleanIndices]
    newdf.reset_index(drop=True)
    return newdf

def NoPromptClusters_WholeFile(df_cluster,df_trig,clusterTimeCut):
    '''
    Return a filtered trigger DataFrame which has all triggers that either
    have no cluster at all, or only have clusters in the time greater than
    clusterTimeCut.
    '''
    #Get Tank Trigger times that have a prompt cluster in the first two microseconds
    DirtyPromptEvents = []
    for j in df_cluster.index.values:  #disgusting...
        if df_cluster["clusterTime"][j] < clusterTimeCut:
            DirtyPromptEvents.append(df_cluster["eventTimeTank"][j])
    #Get indices for trigger entries that don't have an event time in DirtyPromptEvents
    CleanIndices = []
    for j in df_trig.index.values:  #disgusting...
        if df_trig["eventTimeTank"][j] not in DirtyPromptEvents:
            CleanIndices.append(j)
    CleanIndices = np.array(CleanIndices)
    return df_trig.loc[CleanIndices].reset_index(drop=True)

def NoPromptClusters(df_cluster,clusterTimeCut):
    '''
    Return clusters from events that have no prompt cluster with a time earlier
    than the TimeCut variable.
    '''
    DirtyPromptEvents = []
    for j in df_cluster.index.values:  #disgusting...
        if df_cluster["clusterTime"][j] < clusterTimeCut:
            DirtyPromptEvents.append(df_cluster["eventTimeTank"][j])
    CleanIndices = []
    for j in df_cluster.index.values:  #disgusting...
        if df_cluster["eventTimeTank"][j] not in DirtyPromptEvents:
            CleanIndices.append(j)
    CleanIndices = np.array(CleanIndices)
    return df_cluster.loc[CleanIndices].reset_index(drop=True)

def FilterByEventNumber(df,eventnums):
    ReturnIndices = []
    for j in df.index.values:  #disgusting...
        if df["eventTimeTank"][j] in eventnums:
            ReturnIndices.append(j)
    return df.loc[np.array(ReturnIndices)].reset_index(drop=True)

def FilterByEventTime(df,eventnums):
    ReturnIndices = []
    for j in df.index.values:  #disgusting...
        if df["eventTimeTank"][j] in eventnums:
            ReturnIndices.append(j)
    return df.loc[np.array(ReturnIndices)].reset_index(drop=True)

def ValidPromptClusterEvents(df,clusterTimeCut):
    '''
    Given a dataframe and prompt cut, return all clusters associated
    with an event that only has one SiPM pulse per pulse and no clusters
    at less than the clusterTimeCut variable.
    '''
    NoPromptClusterDict = {}
    OnePulses = np.where((df.SiPM1NPulses==1) & (df.SiPM2NPulses==1))[0]
    DirtyPromptEvents = []
    for j in df.index.values:  #disgusting...
        if df["clusterTime"][j] < clusterTimeCut:
            DirtyPromptEvents.append(df["eventTimeTank"][j])
    CleanIndices = []
    for j in df.index.values:  #disgusting...
        if df["eventTimeTank"][j] not in DirtyPromptEvents:
            CleanIndices.append(j)
    CleanIndices = np.array(CleanIndices)
    Cleans = np.intersect1d(OnePulses,CleanIndices)
    df_CleanPrompt = df.loc[Cleans]
    return df_CleanPrompt.reset_index(drop=True)


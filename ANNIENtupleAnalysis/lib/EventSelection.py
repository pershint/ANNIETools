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


########## BEAM-RELATED EVENT SELECTION FUNCTIONS ###########

def MaxPEClusters(df):
    '''
    Prune down the data frame to clusters that only have the largest photoelectron
    value.
    '''
    CurrentEventNum = None
    HighestPE_indices = []
    HighestPE = 0
    for j in df.index.values:  #disgusting...
        if CurrentEventNum is None:
            CurrentEventNum = df["eventTimeTank"][j]
            HighestPE = df["clusterPE"][j]
            HighestPE_index = j
        if CurrentEventNum != df["eventTimeTank"][j]:
            HighestPE_indices.append(HighestPE_index)
            HighestPE = df["clusterPE"][j]
            HighestPE_index = j
            CurrentEventNum = df["eventTimeTank"][j]
        else:
            if df["clusterPE"][j] > HighestPE:
                HighestPE_index = j
    HighestPE_indices = np.array(HighestPE_indices)
    return df.loc[HighestPE_indices].reset_index(drop=True)

def MaxHitClusters(df):
    '''
    Prune down the data frame to clusters that only have the largest clusterHits 
    value.
    '''
    CurrentEventNum = None
    HighestNhit_indices = []
    HighestNhit = 0
    for j in df.index.values:  #disgusting...
        if CurrentEventNum is None:
            CurrentEventNum = df["eventTimeTank"][j]
            HighestNhit = df["clusterHits"][j]
            HighestNhit_index = j
        if CurrentEventNum != df["eventTimeTank"][j]:
            HighestNhit_indices.append(HighestNhit_index)
            HighestNhit = df["clusterHits"][j]
            HighestNhit_index = j
            CurrentEventNum = df["eventTimeTank"][j]
        else:
            if df["clusterHits"][j] > HighestNhit:
                HighestNhit_index = j
    HighestNhit_indices = np.array(HighestNhit_indices)
    return df.loc[HighestNhit_indices].reset_index(drop=True)

def MatchingEventTimes(df1,df2):
    '''
    Take two dataframes and return an array of clusterTimes that 
    come from clusters with matching eventTimeTanks.
    '''
    PMTIndices = []
    MRDIndices = []
    for j in df1.index.values:
        eventTime = df1["eventTimeTank"][j]
        Match = np.where(df2["eventTimeTank"].values == eventTime)[0]
        if len(Match) > 0:
            PMTIndices.append(j)
            MRDIndices.append(Match[0])
        if len(Match) > 1:
            print("OH SHIT, WHY ARE THERE MULTIPLE CLUSTERS AT THIS TIME NOW???")
            print("EVENT TIME TANK IS " + str(eventTime))
    return np.array(PMTIndices), np.array(MRDIndices)


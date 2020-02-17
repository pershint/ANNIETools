import numpy as np

def WeightedMean(w,x):
    return np.sum((w*x)/np.sum(w))

def WeightedStd(w,x):
    nonzerow = np.where(w > 0)[0]
    M = np.sum(w[nonzerow])
    WMean = WeightedMean(w,x)
    return np.sqrt(np.sum(w*((x-WMean)**2))/(((M-1)/M)*np.sum(w)))

def WeightedSEM(w,x):
    '''
    Approximation from Cochran, 1977
    '''
    WMean = WeightedMean(w,x)
    PMean = np.average(w)
    P_i = w/np.sum(w)
    nonzerow = np.where(w > 0)[0]
    M = np.sum(w[nonzerow])
    return(M/(M-1))*(1/(np.sum(P_i)**2))*(np.sum((P_i*x - PMean*WMean)**2) - 
            2*WMean*np.sum((P_i - PMean)*(P_i*x - PMean*WMean)) + 
            (WMean**2) * np.sum((P_i - PMean)**2))


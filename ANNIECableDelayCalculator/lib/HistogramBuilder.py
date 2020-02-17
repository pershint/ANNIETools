#Simple function that converts histograms in a given root
#file into numpy array format

def GetBins(rootfile,hist_name):
    thehist = rootfile.Get(hist_name)
    bin_centers, evts,evts_unc =(), (), () #pandas wants ntuples
    fit_bin_centers, fit_evts,fit_evts_unc =(), (), () #pandas wants ntuples
    for i in xrange(int(thehist.GetNbinsX()+1)):
        if i==0:
            continue
        bin_centers =  bin_centers + ((float(thehist.GetBinWidth(i))/2.0) + float(thehist.GetBinLowEdge(i)),)
        evts = evts + (thehist.GetBinContent(i),)
        evts_unc = evts_unc + (thehist.GetBinError(i),)
        fit_bin_centers =  fit_bin_centers + ((float(thehist.GetBinWidth(i))/2.0) + float(thehist.GetBinLowEdge(i)),)
        fit_evts = fit_evts + (thehist.GetBinContent(i),)
        fit_evts_unc = fit_evts_unc + (thehist.GetBinError(i),)
    bin_centers = np.array(bin_centers)
    evts = np.array(evts)
    evts_unc = np.array(evts_unc)
    return bin_centers, evts, evts_unc


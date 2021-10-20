import json
import lib.ArgParser as ap
import lib.CableDelayEstimator as cde
import lib.DelayPlotter as dp


if __name__ == "__main__":
    if ap.APPEND is not None:
        print(" ####### USING THIS FILE TO FIT LED PEAK TIMES ####### ")
    with open(ap.APPEND,"r") as f:
        myfile = ROOT.TFile.Open(ap.APPEND)
    results, missing = cde.EstimateCableDelays(myfile)
    #Save out outputs to JSON so you can see what they look like
    with open("output/PeakFitResults.json","w") as f:
        json.dump(results,f,indent=4)
    with open("output/MissingPeaks.json","w") as f:
        json.dump(missing,f,indent=4)

    #Now, run script that estimates RELATIVE cable delays and uncertainties
    DetectorFile = "./DB/FullTankPMTGeometry.csv"
    dp.EstimateVisualizeRelativeDelays(DetectorFile,results)

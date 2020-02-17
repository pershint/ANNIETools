import numpy as np
import glob
import pandas as pd

def LoadMuonInfoInDir(dir_path):
    MuonData = {"EntryX":[], "EntryY":[], "EntryZ":[],
            "EntryR":[],
            "TrackAngle":[], "TotQ":[], "TotPE":[],
            "Depth":[],"NHit":[]}
    the_files = glob.glob(dir_path + "/*")
    for f in the_files:
        print(f)
        with open(f,"r") as f:
            lines = f.readlines()
            for j,line in enumerate(lines):
                if line.find("THROUGH")!=-1:
                    entryline = lines[j+1].split(",")
                    MuonData["EntryX"].append(float(entryline[3]))
                    MuonData["EntryY"].append(float(entryline[4]))
                    MuonData["EntryZ"].append(float(entryline[5].rstrip("\n")))
                    MuonData["EntryR"].append(np.sqrt(float(entryline[3])**2 + float(entryline[4])**2))
                    trackline = lines[j+2].split(",")
                    MuonData["TrackAngle"].append(float(trackline[1].rstrip("\n")))
                    penline = lines[j+3].split(",")
                    MuonData["Depth"].append(float(penline[1].rstrip("\n")))
                    qline = lines[j+4].split(",")
                    MuonData["TotQ"].append(float(qline[2]))
                    MuonData["TotPE"].append(float(qline[3].rstrip("\n")))
                    hitline = lines[j-3].split(" ")
                    print(hitline)
                    MuonData["NHit"].append(float(hitline[6].rstrip("\n")))
    print("MUONDATA:")
    print(MuonData)
    return pd.DataFrame(MuonData)

if __name__ == '__main__':
    LoadMuonInfoInDir("./data/")

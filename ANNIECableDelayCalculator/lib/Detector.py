import numpy as np


class TankPMTLoader(object):
    def __init__(self):
        self.have_tank = False
        self.legend_line = ""
        self.data_lines = []


    def ParseTankGeoFile(self,TankPMTGeo):
        lines = []
        legend_line = None
        data_lines = []
        with open(TankPMTGeo,"r") as f:
            lines = f.readlines()
        getting_data = False
        for j,line in enumerate(lines):
            if line.find("LEGEND") != -1:
                legend_line = lines[j+1]
                continue
            if line.find("DATA_START") != -1:
                getting_data = True
                continue
            if line.find("DATA_END") != -1:
                getting_data = False
                continue
            if getting_data:
                data_lines.append(line)
       
        self.legend_args = legend_line.rstrip("\n").split(",")
        self.data_lines = data_lines
        self.have_tank = True

    def GetPMTPositions(self):
        PMT_positions = {}
        pos_inds = [-9999,-9999,-9999]
        for j,lentry in enumerate(self.legend_args):
            if lentry == "x_pos":
                pos_inds[0] = int(j)
            if lentry == "y_pos":
                pos_inds[1] = int(j)
            if lentry == "z_pos":
                pos_inds[2] = int(j)

        print("POSITION INDS: " + str(pos_inds))
        for data in self.data_lines:
            data_args = data.rstrip("\n").split(",")
            print("DATA ARGS: " + str(data_args))
            PMT_positions[int(data_args[0])] = [float(data_args[pos_inds[0]]), 
                    float(data_args[pos_inds[1]]), float(data_args[pos_inds[2]])]
        return PMT_positions
            
    def GetApproxLEDPositions(self,LEDToPMTKey):
        LED_positions = {}
        pos_inds = np.zeros(3)
        pos_inds = [-9999,-9999,-9999]
        for j,lentry in enumerate(self.legend_args):
            if lentry == "x_pos":
                pos_inds[0] = int(j)
            if lentry == "y_pos":
                pos_inds[1] = int(j)
            if lentry == "z_pos":
                pos_inds[2] = int(j)

        for LEDNum in LEDToPMTKey:
            pmt_channelkey = LEDToPMTKey[LEDNum]

            for data in self.data_lines:
                data_args = data.rstrip("\n").split(",")
                if int(data_args[0]) == pmt_channelkey:
                    LED_positions[int(LEDNum)] = [float(data_args[pos_inds[0]]), 
                            float(data_args[pos_inds[1]]), float(data_args[pos_inds[2]])]
        return LED_positions

if __name__ == "__main__":
    DetectorFile = "FullTankPMTGeometry_Corrected.csv"
    PMTGetter = TankPMTLoader()
    PMTGetter.ParseTankGeoFile(DetectorFile)
    pmtpos = PMTGetter.GetPMTPositions()
    LEDToPMTKey = {3: 382, 0:405, 1:344, 2:404, 4:456, 5:393}
    ledpos = PMTGetter.GetApproxLEDPositions(LEDToPMTKey)
    print("PMT POSNS: " + str(pmtpos))
    print("LED POSNS: " + str(ledpos))

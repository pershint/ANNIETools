import argparse
import sys
#Oscillation variables that will be measured by SNO+/vary
#between SuperK and KamLAND; fixed parameters are found hard-coded in
#./lib/NuSPectrum.py


parser = argparse.ArgumentParser(description='Python-based software for determining and '+\
        ' logging the charge gains of PMTs')
parser.add_argument("--debug",action="store_true")
parser.add_argument("-F", "--fitalgorithm",action="store",dest="FIT",
                  type=str,
                  help='Specify whether to use simple Gauss fits, or the' + \
                  'DEAP fitter algorithm (either "Simple" or "DEAP"')
parser.add_argument("-D", "--database",action="store",dest="DB",
                  type=str,
                  help="Specify the JSON file to either analyze or append fits to")
parser.add_argument("-A", "--append",action="store",dest="APPEND",
                  type=str,
                  help="Path to a ROOT file holding new charge information for tubes")
parser.add_argument("-r", "--runnumber",action="store",dest="RUNNUM",
                  type=str,
                  help="Specify the run number associated with this file")
parser.add_argument("-d", "--date",action="store",dest="DATE",
                  type=str,
                  help="Specify the date this data was taken")
parser.add_argument("-l", "--LEDs",action="store",dest="LED",
                  type=str,
                  help="Specify which LEDs were on (Input is CSV, least to greatest)")
parser.add_argument("-v", "--Voltage",action="store",dest="VOLTS",
                  type=str,
                  help="Specify the voltage setpoint)")
parser.add_argument("-p", "--PIN",action="store",dest="PIN",
                  type=str,
                  help="Specify the PIN setpoint for all LEDs")


parser.set_defaults(DB="./DB/TransparencyGains.json",APPEND=None,debug="False",
        RUNNUM=None,DATE=None,LED=None,PIN=None,FIT="Simple")
args = parser.parse_args()
DB = args.DB
APPEND = args.APPEND
DEBUG = args.debug
RUNNUM = args.RUNNUM
DATE = args.DATE
LED = args.LED
PIN = args.PIN
VOLTS = args.VOLTS
FIT = args.FIT

if RUNNUM is None:
    RUNNUM = str(raw_input("Input run number: "))
if DATE is None:
    DATE = str(raw_input("Input date of run (MM/DD/YYYY): "))
if LED is None:
    LED = str(raw_input("Which LEDs are on (Input is CSV, least to greatest): "))
if PIN is None:
    PIN = str(raw_input("PIN setpoint for LEDs: "))
if VOLTS is None:
    VOLTS = str(raw_input("Voltage setpoint for PMTs: "))


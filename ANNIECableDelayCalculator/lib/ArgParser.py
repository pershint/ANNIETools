import argparse
import sys

#Simple example of how to make your code have a simple command line interface (CLI)

parser = argparse.ArgumentParser(description='Python-based software for determining and '+\
        ' logging the charge gains of PMTs')
parser.add_argument("--debug",action="store_true")
parser.add_argument("-A", "--append",action="store",dest="APPEND",
                  type=str,
                  help="Path to a ROOT file holding new charge information for tubes")
parser.set_defaults(APPEND="./data/testdata.root",debug="False")
args = parser.parse_args()
APPEND = args.APPEND
DEBUG = args.debug

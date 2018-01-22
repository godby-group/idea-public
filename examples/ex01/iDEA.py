# Some python magic to make it find the iDEA directory
# even if it hasn't been added to the PYTHONPATH
import sys
import os
sys.path.insert(0, os.path.abspath('../..'))
from iDEA.input import Input
sys.path.pop(0)

# read parameters file
inp = Input.from_python_file('parameters.py')
# perform checks on input parameters
inp.check()
# run job
inp.execute()

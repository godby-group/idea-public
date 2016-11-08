import sys
import os
# Some python magic to make it find the iDEA directory
# even if it hasn't been added to the PYTHONPATH
sys.path.insert(0, os.path.abspath('../..'))

from iDEA.input import Input
from iDEA.job import Job

# read parameters file
inp = Input.from_python_file('parameters.py')
# perform checks on input parameters
inp.check()
# pass parameters to job
job = Job(inp)
# run job
job.run()

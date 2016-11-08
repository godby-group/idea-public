from iDEA.input import Input
from iDEA.job import Job

# read parameters file
inp = Input.from_python_file('parameters.py')
inp.run.verbosity = 'low'

# Converging xmax parameter
for xmax in [4,6,8,10]:
    # Note: the dependent sys.deltax is automatically updated
    inp.sys.xmax = xmax

    # perform checks on input parameters
    inp.check()
    # pass parameters to job
    job = Job(inp)
    job.run()
    E = job.results.NON.gs_non_E
    print(" xmax = {:4.1f}, E = {:6.4f} Ha".format(xmax,E))

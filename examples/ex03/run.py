from iDEA.input import Input
from iDEA.job import Job

# use default values for parameters
inp = Input()

inp.run.NON = True
inp.run.verbosity = 'low'

# Converging xmax parameter
for xmax in [4,6,8,10]:
    # Note: the dependent sys.deltax is automatically updated
    inp.sys.xmax = xmax
    inp.sys.grid = 401

    # perform checks on input parameters
    inp.check()

    # print sys section of input file
    #print(inp.sys)

    # pass parameters to job
    job = Job(inp)
    job.run()
    E = job.results.NON.gs_non_E
    print(" xmax = {:4.1f}, E = {:6.4f} Ha".format(xmax,E))

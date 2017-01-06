from iDEA.input import Input

# use default values for all parameters
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

    inp.execute()
    E = inp.results.NON.gs_non_E
    print(" xmax = {:4.1f}, E = {:6.4f} Ha".format(xmax,E))

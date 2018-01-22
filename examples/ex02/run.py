from iDEA.input import Input

# read parameters file
inp = Input.from_python_file('parameters.py')
inp.run.verbosity = 'low'

# Converging xmax parameter
for xmax in [4,6,8,10]:
    # Note: the dependent sys.deltax is automatically updated
    inp.sys.xmax = xmax

    # perform checks on input parameters
    inp.check()
    inp.execute()
    E = inp.results.non.gs_non_E
    print(" xmax = {:4.1f}, E = {:6.4f} Ha".format(xmax,E))

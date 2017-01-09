import parameters as pm

if pm.run.module == 'iDEA':
    from iDEA.input import Input
else:
    # import iDEA from alternative folder, if specified
    import importlib
    input = importlib.import_module("{}.input".format(pm.run.module))
    Input = input.Input

# read parameters file into Input object
inp = Input.from_python_file('parameters.py')

# perform checks on input parameters
inp.check()

# run job
inp.execute()

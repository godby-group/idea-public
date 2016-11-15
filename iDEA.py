from iDEA.input import Input

# read parameters file
inp = Input.from_python_file('parameters.py')
# perform checks on input parameters
inp.check()

if inp.run.module == 'iDEA':
    import iDEA.job as job
else:
    # import iDEA from alternative folder, if specified
    import importlib
    job = importlib.import_module("{}.job".format(inp.run.module))

# pass parameters to job
j = job.Job(inp)
# run job
j.run()

from source.iDEA.input import Input
from source.iDEA.job import Job

# read parameters file
inp = Input.from_python_file('parameters.py')
# perform checks on input parameters
inp.check()
# pass parameters to job
job = Job(inp)
# run job
job.run()
# save results
job.save()


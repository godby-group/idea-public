# Determine version of source code to be used
import parameters as pm
import os
version = 'ver' + str(pm.run.code_version)

# Ensure output directories exist, if not create them
os.system('mkdir -p outputs/' + str(pm.run.name))
os.system('mkdir -p outputs/' + str(pm.run.name) + '/data')
os.system('mkdir -p outputs/' + str(pm.run.name) + '/raw')
os.system('mkdir -p outputs/' + str(pm.run.name) + '/plots')
os.system('mkdir -p outputs/' + str(pm.run.name) + '/animations')

# Collect relevent source code to be run
os.system('cp source/' + str(version) + '/* outputs/' + str(pm.run.name))
os.system('cp parameters.py' + ' outputs/' + str(pm.run.name))

# Run relevent code
os.system('python outputs/' + str(pm.run.name) + '/run.py')

# Remove temporary code
os.system('mv outputs/' + str(pm.run.name) + '/parameters.py outputs/' + str(pm.run.name) + '/parameters.temp')
os.system('mv outputs/' + str(pm.run.name) + '/ViDEO.py outputs/' + str(pm.run.name) + '/ViDEO.temp')
os.system('rm -f outputs/' + str(pm.run.name) + '/*.py')
os.system('rm -f outputs/' + str(pm.run.name) + '/*.pyc')
os.system('rm -f outputs/' + str(pm.run.name) + '/*.npy')
os.system('rm -f outputs/' + str(pm.run.name) + '/*.f90')
os.system('rm -f outputs/' + str(pm.run.name) + '/*.f90~')
os.system('rm -f outputs/' + str(pm.run.name) + '/*.f')
os.system('rm -f outputs/' + str(pm.run.name) + '/*.f~')
os.system('rm -f outputs/' + str(pm.run.name) + '/*.compiler')
os.system('rm -f outputs/' + str(pm.run.name) + '/*.so')
os.system('rm -f outputs/' + str(pm.run.name) + '/*.py~')
os.system('rm -f outputs/' + str(pm.run.name) + '/*.txt')
os.system('rm -f outputs/' + str(pm.run.name) + '/*.txt~')
os.system('rm -f outputs/' + str(pm.run.name) + '/*.compiler~')
os.system('rm -f *.py~')
os.system('rm -f *.pyc')

# Add ViDEO to the directory to be used for visualisation
os.system('mv outputs/' + str(pm.run.name) + '/ViDEO.temp outputs/' + str(pm.run.name) + '/ViDEO.py')

# Add the parameters file to show details of the run
os.system('mv outputs/' + str(pm.run.name) + '/parameters.temp outputs/' + str(pm.run.name) + '/parameters.py')

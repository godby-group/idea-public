# Determine version of source code to be used
import parameters as pm
import os
version = 'ver' + str(pm.code_version)

# Ensure output directories exist, if not create them
os.system('mkdir -p outputs/' + str(pm.run_name))
os.system('mkdir -p outputs/' + str(pm.run_name) + '/data')
os.system('mkdir -p outputs/' + str(pm.run_name) + '/raw')
os.system('mkdir -p outputs/' + str(pm.run_name) + '/plots')
os.system('mkdir -p outputs/' + str(pm.run_name) + '/animations')

# Collect relevent source code to be run
os.system('cp source/' + str(version) + '/* outputs/' + str(pm.run_name))
os.system('cp parameters.py' + ' outputs/' + str(pm.run_name))

# Run relevent code
os.system('python outputs/' + str(pm.run_name) + '/run.py')

# Remove temporary code
os.system('mv outputs/' + str(pm.run_name) + '/parameters.py outputs/' + str(pm.run_name) + '/parameters.temp')
os.system('mv outputs/' + str(pm.run_name) + '/ViDEO.py outputs/' + str(pm.run_name) + '/ViDEO.temp')
os.system('rm -f outputs/' + str(pm.run_name) + '/*.py')
os.system('rm -f outputs/' + str(pm.run_name) + '/*.pyc')
os.system('rm -f outputs/' + str(pm.run_name) + '/*.npy')
os.system('rm -f outputs/' + str(pm.run_name) + '/*.f90')
os.system('rm -f outputs/' + str(pm.run_name) + '/*.f90~')
os.system('rm -f outputs/' + str(pm.run_name) + '/*.f')
os.system('rm -f outputs/' + str(pm.run_name) + '/*.f~')
os.system('rm -f outputs/' + str(pm.run_name) + '/*.compiler')
os.system('rm -f outputs/' + str(pm.run_name) + '/*.so')
os.system('rm -f outputs/' + str(pm.run_name) + '/*.py~')
os.system('rm -f outputs/' + str(pm.run_name) + '/*.txt')
os.system('rm -f outputs/' + str(pm.run_name) + '/*.compiler~')
os.system('rm -f *.py~')
os.system('rm -f *.pyc')

# Add ViDEO to the directory to be used for visualisation
os.system('mv outputs/' + str(pm.run_name) + '/ViDEO.temp outputs/' + str(pm.run_name) + '/ViDEO.py')

# Add the parameters file to show details of the run
os.system('mv outputs/' + str(pm.run_name) + '/parameters.temp outputs/' + str(pm.run_name) + '/parameters.py')

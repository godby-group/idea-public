"""Handles running iDEA jobs

"""
import numpy as np
import pickle
import copy as cp
import results


class Job(object):
    """iDEA job"""

    def __init__(self, inp):
        """Set up iDEA job to be run

        Parameters
        ----------
        inp : Input object
            Contains all input/parameters required to run the calculation.
            The input is passed on to the various iDEA functions.
        """
        self.pm = inp


    def run(self):
        """Run this job"""
        import os
        import shutil
        import errno
        pm = self.pm
        pm.check()

        def mkdir_p(path):
            try:
                os.makedirs(path)
            except OSError as exc:
                if exc.errno == errno.EEXIST and os.path.isdir(path):
                    pass
                else: raise

        # Note: this is not being used now...
        version = 'ver' + str(pm.run.code_version)

        output_dirs = ['data', 'raw', 'plots', 'animations']
        for d in output_dirs:
            path = '{}/{}'.format(self.pm.output_dir,d)
            mkdir_p(path)
            setattr(pm,d,path)


        ## Store copy of input
        ## Doesn't work like that -- can't pickle module objects...
        #fpath = '{}/{}/{}'.format(pm.run.name,'data','input.p')
        #f = open(fpath, 'wb')
        #pickle.dump(pm,f)
        #f.close()


        ## Collect relevant source code to be run
        #os.system('cp source/' + str(version) + '/* outputs/' + str(pm.run.name))
        #os.system('cp parameters.py' + ' outputs/' + str(pm.run.name))

        ## Run relevant code
        #os.system('python outputs/' + str(pm.run.name) + '/run.py')
        import splash
        # Draw splash to screen
        splash.draw(pm.run.verbosity)
        print('run name: ' + str(pm.run.name))

        # Execute required jobs
        results = results.Results()
        # Execute required jobs
        if(pm.sys.NE == 1):
           if(pm.run.EXT == True):
              import SPiDEA
              SPiDEA.main(pm)
           if(pm.ext.RE == True):
              import RE
              RE.main(pm,'ext')
        elif(pm.sys.NE == 2):
           if(pm.run.EXT == True):
              import EXT2
              EXT2.main(pm)
           if(pm.ext.RE == True):
              import RE
              RE.main(pm,'ext')
        elif(pm.sys.NE == 3):
           if(pm.run.EXT == True):
              import EXT3
              EXT3.main(pm)
           if(pm.ext.RE == True):
              import RE
              RE.main(pm,'ext')
        elif(pm.sys.NE >= 4):
           if(pm.run.EXT == True):
              print('EXT: cannot run exact with more than 3 electrons')

        if(pm.run.NON == True):
              import NON
              NON.main(pm)
        if(pm.non.RE == True):
              import RE
              RE.main(pm,'non')

        if(pm.run.LDA == True):
              import LDA
              LDA.main(pm)

        if(pm.run.MLP == True):
              import MLP
              MLP.main(pm)

        if(pm.run.HF == True):
              import HF
        if(pm.hf.RE == True):
              import RE
              RE.main(pm,'hf')

        if(pm.run.MBPT == True):
              import MBPT
              r = MBPT.main(pm,'mbpt')
              results.add(r, name='MBPT')
        if(pm.mbpt.RE == True):
              import RE

        if(pm.run.LAN == True):
              import LAN
              LAN.main(pm)

        import sprint
        # All jobs done
        string = 'all jobs done \n'
        sprint.sprint(string,1,pm.run.verbosity)


        # Copy parameters file and ViDEO script to output folder
        shutil.copy2(self.pm.filename,self.pm.output_dir)

        # Note: this doesn't work, when using iDEA as a system module
        vfile = 'iDEA/ViDEO.py'
        if os.path.isfile(vfile):
            shutil.copy2('iDEA/ViDEO.py',self.pm.output_dir)
 
        ### Remove temporary code
        ##os.system('mv outputs/' + str(pm.run.name) + '/parameters.py outputs/' + str(pm.run.name) + '/parameters.temp')
        ##os.system('mv outputs/' + str(pm.run.name) + '/ViDEO.py outputs/' + str(pm.run.name) + '/ViDEO.temp')
        ##os.system('rm -f outputs/' + str(pm.run.name) + '/*.py')
        ##os.system('rm -f outputs/' + str(pm.run.name) + '/*.pyc')
        ##os.system('rm -f outputs/' + str(pm.run.name) + '/*.npy')
        ##os.system('rm -f outputs/' + str(pm.run.name) + '/*.f90')
        ##os.system('rm -f outputs/' + str(pm.run.name) + '/*.f')

        ## Add ViDEO to the directory to be used for visualisation
        #os.rename(pm.run.name + '/ViDEO.temp', pm.run.name + '/ViDEO.py')

        ## Add the parameters file to show details of the run
        #os.rename(pm.run.name + '/parameters.temp', pm.run.name + '/parameters.py')

        self.results = results
        return results

    def save(self):
        """Save results to disk."""
        self.results.save(dir=self.pm.output_dir + 'data')

    def post_process(self):
        """Run ViDEO post-processing script"""
        import ViDEO

        ViDEO.main(self.pm)

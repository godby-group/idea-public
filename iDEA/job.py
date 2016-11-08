"""Handles running iDEA jobs

"""
import numpy as np
import pickle
import copy as cp
import results as rs
import sprint


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

        #version = 'ver' + str(pm.run.code_version)

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
        sprint.sprint('run name: ' + str(pm.run.name),1,pm.run.verbosity)

        # Execute required jobs
        self.results = rs.Results()
        results = self.results
        # Execute required jobs
        if(pm.sys.NE == 1):
           if(pm.run.EXT == True):
              import SPiDEA
              results.add(SPiDEA.main(pm), name='EXT')
           if(pm.ext.RE == True):
              import RE
              results.add(RE.main(pm,'ext'), name='RE')
        elif(pm.sys.NE == 2):
           if(pm.run.EXT == True):
              import EXT2
              results.add(EXT2.main(pm), name='EXT')
           if(pm.ext.RE == True):
              import RE
              results.add(RE.main(pm,'ext'), name='RE')
        elif(pm.sys.NE == 3):
           if(pm.run.EXT == True):
              import EXT3
              results.add(EXT3.main(pm), name='EXT')
           if(pm.ext.RE == True):
              import RE
              results.add(RE.main(pm,'ext'), name='RE')
        elif(pm.sys.NE >= 4):
           if(pm.run.EXT == True):
              print('EXT: cannot run exact with more than 3 electrons')

        if(pm.run.NON == True):
              import NON
              results.add(NON.main(pm), name='NON')
        if(pm.non.RE == True):
              import RE
              results.add(RE.main(pm,'non'), name='RE')

        if(pm.run.LDA == True):
              import LDA
              results.add(LDA.main(pm), name='LDA')
        if(pm.run.MLP == True):
              import MLP
              MLP.main(pm)

        if(pm.run.HF == True):
              import HF
              results.add(HF.main(pm), name='HF')
        if(pm.hf.RE == True):
              import RE
              results.add(RE.main(pm,'hf'), name='RE')

        if(pm.run.MBPT == True):
              import MBPT
              results.add(MBPT.main(pm), name='MBPT')
        if(pm.mbpt.RE == True):
              import RE
              results.add(RE.main(pm,'mbpt'), name='RE')

        if(pm.run.LAN == True):
              import LAN
              results.add(LAN.main(pm), name='LAN')

        # All jobs done
        string = 'all jobs done \n'
        sprint.sprint(string,1,pm.run.verbosity)

        # Copy parameters file and ViDEO script to output folder
        # Note: this doesn't work, when there is no actual parameters file
        if os.path.isfile(pm.filename):
            shutil.copy2(pm.filename,pm.output_dir)

        # Note: this doesn't work, when using iDEA as a system module
        vfile = 'iDEA/ViDEO.py'
        if os.path.isfile(vfile):
            shutil.copy2('iDEA/ViDEO.py',pm.output_dir)
        else:
            s  = "Warning: Unable to copy ViDEO.py since running iDEA as python module."
            s += " Simply add the iDEA folder to your PATH variable to use ViDEO.py anywhere"
            sprint.sprint(s,1,pm.run.verbosity)

        return results

    def save(self):
        """Save results to disk."""
        self.results.save(dir=self.pm.output_dir + 'data',verbosity=self.pm.run.verbosity)

    def post_process(self):
        """Run ViDEO post-processing script"""
        import ViDEO

        ViDEO.main(self.pm)

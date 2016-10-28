""" Stores input parameters for iDEA calculations.
"""
import numpy as np
import importlib
import os

class Input(object):

    def __init__(self):
        """Sets default values of some properties."""
        pass

    def check(self):
        """Checks validity of input parameters."""
        pm = self
        if pm.run.time_dependence == True:
            if pm.run.HF == True:
                sprint.sprint('HF: Warning - time-dependence not implemented!')
            if pm.run.MBPT == True:
                sprint.sprint('MBPT: Warning - time-dependence not implemented!')


    @classmethod
    def from_python_file(self,filename):
        """Create Input from Python script."""
        tmp = Input()
        tmp.read_from_python_file(filename)
        return tmp

    def read_from_python_file(self,filename):
        """Update Input from Python script."""
        if not os.path.isfile(filename):
            raise IOError("Could not find file {}".format(filename))

        module, ext = os.path.splitext(filename)
        if ext != ".py":
            raise IOError("File {} does not have .py extension.".format(filename))

        # import module into object
        pm = importlib.import_module(module)

        # overvwrite member variables with those from object
        self.__dict__.update(pm.__dict__)
        self.filename = filename

    ##########################################
    ### TODO: Here add derived parameters ####
    ##########################################

    @property
    def output_dir(self):
        """Returns full path to output directory
        """
        return 'outputs/{}'.format(self.run.name)


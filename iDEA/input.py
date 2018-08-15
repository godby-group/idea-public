""" Stores input parameters for iDEA calculations.
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from six import string_types

import numpy as np
import importlib
import os
import sys
import copy
import time

from . import results as rs


class SpaceGrid(object):
   """Stores basic real space arrays

   These arrays should be helpful in many types of iDEA calculations.
   Storing them in the Input object avoids having to recompute them
   and reduces code duplication.
   """

   def __init__(self, pm):
       self.npt = pm.sys.grid
       self.delta = pm.sys.deltax
       self.grid = np.linspace(-pm.sys.xmax, pm.sys.xmax, pm.sys.grid)

       self.v_ext = np.zeros(self.npt, dtype=np.float)
       for i in range(self.npt):
           self.v_ext[i] = pm.sys.v_ext(self.grid[i])

       self.v_pert = np.zeros(self.npt, dtype=np.float)
       if(pm.sys.im == 1):
           self.v_pert = self.v_pert.astype(np.cfloat)
       for i in range(self.npt):
           self.v_pert[i] = pm.sys.v_pert(self.grid[i])

       self.v_int = np.zeros((pm.sys.grid,pm.sys.grid),dtype='float')
       for i in range(pm.sys.grid):
          for k in range(pm.sys.grid):
             self.v_int[i,k] = pm.sys.interaction_strength/(abs(self.grid[i]-self.grid[k])+pm.sys.acon)

       stencil_first_derivative = pm.re.stencil
       if stencil_first_derivative == 5:
           self.first_derivative = 1.0/12 * np.array([1,-8,0,8,-1], dtype=np.float) / self.delta
           self.first_derivative_indices = [-2,-1,0,1,2]
           self.first_derivative_band = 1.0/12 * np.array([0,-8,1], dtype=np.float) / self.delta
       elif stencil_first_derivative == 7:
           self.first_derivative = 1.0/60 * np.array([-1,9,-45,0,45,-9,1], dtype=np.float) / self.delta
           self.first_derivative_indices = [-3,-2,-1,0,1,2,3]
           self.first_derivative_band = 1.0/60 * np.array([0,-45,9,-1], dtype=np.float) / self.delta
       else:
           raise ValueError("re.stencil = {} not implemented. Please select 5 or 7.".format(stencil_first_derivative))

       stencil_second_derivative = pm.sys.stencil
       if stencil_second_derivative == 3:
           self.second_derivative = np.array([1,-2,1], dtype=np.float) / self.delta**2
           self.second_derivative_indices = [-1,0,1]
           self.second_derivative_band = np.array([-2,1], dtype=np.float) / self.delta**2
       elif stencil_second_derivative == 5:
           self.second_derivative = 1.0/12 * np.array([-1,16,-30,16,-1], dtype=np.float) / self.delta**2
           self.second_derivative_indices = [-2,-1,0,1,2]
           self.second_derivative_band = 1.0/12 * np.array([-30,16,-1], dtype=np.float) / self.delta**2
       elif stencil_second_derivative == 7:
           self.second_derivative = 1.0/180 * np.array([2,-27,270,-490,270,-27,2], dtype=np.float) / self.delta**2
           self.second_derivative_indices = [-3,-2,-1,0,1,2,3]
           self.second_derivative_band = 1.0/180 * np.array([-490,270,-27,2], dtype=np.float) / self.delta**2
       else:
           raise ValueError("sys.stencil = {} not implemented. Please select 3, 5 or 7.".format(stencil_second_derivative))



   def __str__(self):
       """Print variables of section and their values"""
       s = ""
       v = vars(self)
       for key,value in v.items():
           s += input_string(key, value)
       return s

def input_string(key,value):
    """Prints a line of the input file"""
    if isinstance(value, string_types):
        s = "{} = '{}'\n".format(key, value)
    else:
        s = "{} = {}\n".format(key, value)
    return s


class InputSection(object):
   """Generic section of input file"""

   def __str__(self):
       """Print variables of section and their values"""
       s = ""
       v = vars(self)
       for key,value in v.items():
           s += input_string(key, value)
       return s


class SystemSection(InputSection):
    """System section of input file

    Includes some derived quantities.
    """

    @property
    def deltax(self):
        """Spacing of real space grid"""
        return 2.0*self.xmax/(self.grid-1)

    @property
    def deltat(self):
        """Spacing of temporal grid"""
        return 1.0*self.tmax/(self.imax-1)

    @property
    def grid_points(self):
        """Real space grid"""
        return np.linspace(-self.xmax,self.xmax,self.grid)


class Input(object):
    """Stores variables of input parameters file

    Includes automatic generation of dependent variables,
    checking of input parameters, printing back to file and more.
    """

    priority_dict = {
      'low': 2,
      'default': 1,
      'high': 0}

    def __init__(self):
        """Sets default values of some properties."""
        self.filename = ''
        self.log = ''
        self.last_print = time.clock()

        ### Run parameters
        self.run = InputSection()
        run = self.run
        run.name = 'run_name'                #: Name to identify run. Note: Do not use spaces or any special characters (.~[]{}<>?/\)
        run.time_dependence = False          #: Run time-dependent calculation
        run.verbosity = 'default'            #: Output verbosity ('low', 'default', 'high')
        run.save = True                      #: Save results to disk when they are generated
        run.module = 'iDEA'                  #: Specify alternative folder (in this directory) containing modified iDEA module
        run.NON = False                      #: Run Non-Interacting approximation
        run.LDA = False                      #: Run LDA approximation
        run.HF = False                       #: Run Hartree-Fock approximation
        run.HYB = False                      #: Run Hybrid (HF-LDA) calculation
        run.EXT = False                      #: Run Exact Many-Body calculation


        ### System parameters
        self.sys = SystemSection()
        sys = self.sys
        sys.NE = 2                           #: Number of electrons
        sys.grid = 201                       #: Number of grid points (must be odd)
        sys.stencil = 3                      #: Discretisation of 2nd derivative (3 or 5 or 7).
        sys.xmax = 10.0                      #: Size of the system
        sys.tmax = 1.0                       #: Total real time
        sys.imax = 1001                      #: Number of real time iterations (NB: deltat = tmax/(imax-1))
        sys.acon = 1.0                       #: Smoothing of the Coloumb interaction
        sys.interaction_strength = 1.0       #: Scales the strength of the Coulomb interaction
        sys.im = 0                           #: Are there imaginary terms in the perturbing potential? (0: no, 1: yes)

        def v_ext(x):
            """Ground-state external potential
            """
            return 0.5*(0.25**2)*(x**2)
        sys.v_ext = v_ext

        def v_pert(x):
            """Perturbing potential (switched on at t=0)
            """
            return -0.01*x
        sys.v_pert = v_pert


        ### Exact parameters
        self.ext = InputSection()
        ext = self.ext
        ext.itol = 1e-12                     #: Tolerance of imaginary time propagation (Recommended: 1e-12)
        ext.itol_solver = 1e-14              #: Tolerance of linear solver in imaginary time propagation (Recommended: 1e-14)
        ext.rtol_solver = 1e-12              #: Tolerance of linear solver in real time propagation (Recommended: 1e-12)
        ext.itmax = 2000.0                   #: Total imaginary time
        ext.iimax = 1e5                      #: Imaginary time iterations
        ext.ideltat = ext.itmax/ext.iimax    #: Imaginary time step (DERIVED)
        ext.RE = False                       #: Reverse engineer many-body density
        ext.psi_gs = False                   #: Save the reduced ground-state wavefunction to file
        ext.initial_gspsi = 'qho'            #: Initial ground-state wavefunction ('qho' by default. 'non' can be selected.
                                             #: 'hf', 'lda1', 'lda2', 'lda3', 'ldaheg' or 'ext' can be selected if the orbitals/wavefunction
                                             #: are available. An ext wavefunction from another run can be used, but specify the run.name
                                             #: instead e.g. 'run_name').
                                             #: WARNING: If no reliable starting guess can be provided e.g. wrong number of electrons per well,
                                             #: then choose 'qho' - this will ensure stable convergence to the true ground-state.)


        ### Non-interacting approximation parameters
        self.non = InputSection()
        non = self.non
        non.rtol_solver = 1e-13              #: Tolerance of linear solver in real time propagation (Recommended: 1e-13)
        non.RE = False                       #: Reverse-engineer non-interacting density


        ### LDA parameters
        self.lda = InputSection()
        lda = self.lda
        lda.NE = 'heg'                       #: Number of electrons used in construction of the LDA (1, 2, 3 or 'heg')
        lda.scf_type = 'pulay'               #: How to perform scf ('pulay', 'linear', 'cg', 'mixh', 'none')
        lda.mix = 0.2                        #: Mixing parameter for linear & Pulay mixing (float in [0,1])
        lda.pulay_order = 20                 #: Length of history for Pulay mixing (max: lda.max_iter)
        lda.pulay_preconditioner = None      #: Preconditioner for pulay mixing (None, 'kerker', rpa')
        lda.kerker_length = 0.5              #: Length over which density fluctuations are screened (Kerker only)
        lda.tol = 1e-12                      #: Convergence tolerance in the density
        lda.etol = 1e-12                     #: Convergence tolerance in the energy
        lda.max_iter = 10000                 #: Maximum number of self-consistency iterations
        lda.RE = False                       #: Reverse-engineer LDA density


        ### HF parameters
        self.hf = InputSection()
        hf = self.hf
        hf.fock = 1                          #: Include Fock term (0 = Hartree approximation, 1 = Hartree-Fock approximation)
        hf.con = 1e-12                       #: Tolerance
        hf.nu = 0.9                          #: Mixing term
        hf.RE = False                        #: Reverse-engineer hf density


        ### HYB parameters
        self.hyb = InputSection()
        hyb = self.hyb
        hyb.seperate = False                 #: Seperate Vx and Vc in the hybrid (False: a*F + (1-a)Vxc, True: a*F + (1-a)Vx + Vc)
        hyb.functionality = 'o'              #: Functionality of hybrid functionals: 'o' for optimal alpha, 'f' for fractional numbers of electrons,
                                             #: 'a' for single alpha run
        hyb.of_array = (0.5,1.0,6)           #: If finding optimal alpa, this defines an array going from a->b in c steps whose energies are used for
                                             #: optimisation. If fractional run, this defines the numbers of electrons to calculate
        hyb.alpha = 1.0                      #: If single alpha run, this defines the alpha
        hyb.mix = 0.5                        #: Mixing parameter for linear  mixing (float in [0,1])
        hyb.tol = 1e-12                      #: Convergence tolerance in the density
        hyb.max_iter = 10000                 #: Maximum number of self-consistency iterations
        hyb.RE = False                       #: Calculate the external potential for the HYB density


        ### RE parameters
        self.re = InputSection()
        re = self.re
        re.stencil = 5                       #: Discretisation of 1st derivative (5 or 7)
        re.mu = 1.0                          #: 1st convergence parameter in the ground-state reverse-engineering algorithm
        re.p = 0.05                          #: 2nd convergence parameter in the ground-state reverse-engineering algorithm
        re.gs_density_tolerance = 1e-12      #: Tolerance of the error in the ground-state density
        re.starting_guess = 'extre'          #: Starting guess of groud-state Vks (if not available will start with Vxt)
        re.nu = 1.0                          #: 1st convergence parameter in the time-dependent reverse-engineering algorithm
        re.a = 1.0e-6                        #: 2nd convergence parameter in the time-dependent reverse-engineering algorithm
        re.rtol_solver = 1e-12               #: Tolerance of linear solver in real time propagation (Recommended: 1e-12)
        re.td_density_tolerance = 1e-7       #: Tolerance of the error in the time-dependent density
        re.cdensity_tolerance = 1e-7         #: Tolerance of the error in the current density
        re.max_iterations = 20               #: Maximum number of iterations per time step to find the Kohn-Sham potential
        re.damping = True                    #: Damping term used to filter out the noise in the time-dependent Kohn-Sham vector potential
        re.filter_beta = 1.8                 #: 1st parameter in the damping term

    def check(self):
        """Checks validity of input parameters."""
        pm = self

        # Time-dependence
        if pm.run.time_dependence == True:
            if pm.run.HYB == True:
                self.sprint('HYB: Warning - time-dependence not implemented!')
            if (pm.ext.RE or pm.non.RE or pm.lda.RE or pm.hf.RE or pm.hyb.RE):
                self.sprint('RE: Warning - time-dependence not implemented!')

        # EXT
        if pm.run.EXT == True:
            if pm.ext.itol > 1e-6:
                self.sprint('EXT: Warning - value of ext.itol is much larger than 1e-13, this can yeild poor KS potentials')

        # LDA
        if pm.run.LDA == True:
            if pm.lda.scf_type not in [None, 'pulay', 'linear', 'cg', 'mixh']:
                raise ValueError("LDA: Warning - lda.scf_type must be None, 'linear', 'pulay' or 'cg'")
            if pm.lda.pulay_preconditioner not in [None, 'kerker', 'rpa']:
                raise ValueError("LDA: Warning - lda.pulay_preconditioner must be None, 'kerker' or 'rpa'")

        # HF
        if pm.run.HF == True:
            if (pm.hf.fock != 1 and pm.hf.fock != 0):
                raise ValueError('HF: Error - hf.fock must be set to 0 or 1.')
            if (pm.hf.nu > 1 or pm.hf.nu < 0):
                self.sprint('HF: Warning - Value of nu should be between 0 and 1')

        # HYB
        if pm.run.HYB == True:
            if (pm.hyb.alpha  > 1 or pm.hyb.alpha  < 0):
                self.sprint('HYB: Warning - Value of alpha should be between 0 and 1')
            if (pm.hyb.mix > 1 or pm.hyb.mix < 0):
                self.sprint('HYB: Warning - Value of mix should be between 0 and 1')

    def __str__(self):
        """Prints different sections in input file"""
        s = ""
        v = vars(self)
        for key, value in v.items():
            if isinstance(value, InputSection):
                s += "### {} section\n".format(key)
                s += "{}\n".format(value)
            else:
                s += input_string(key,value)
        return s

    def sprint(self, string='', priority=1, newline=True, refresh=0.000005, savelog=True):
        """Customized print function

        Prints to screen and appends to log.

        If newline == False, overwrites last line,
        but refreshes only every refresh seconds.

        parameters
        ----------
        string : string
            string to be printed
        priority: int
            priority of message, possible values are
            0: debug
            1: normal
            2: important
        newline : bool
            If False, overwrite the last line
        refresh : float
            If newline == False, print only every "refresh" seconds
        savelog : bool
            If True, save string to log file
        """
        verbosity = self.run.verbosity
        if(savelog):
            self.log += string + '\n'
        if priority >= self.priority_dict[verbosity]:

            timestamp = time.clock()
            if newline:
                print(string)
                self.last_print = timestamp
            # When overwriting lines, we only print every "refresh" seconds
            elif timestamp - self.last_print > refresh:
                ## this only overwrites, no erase
                #print('\r' + string, end='')

                # Overwrite line
                sys.stdout.write('\r' + string)
                # Delete rest of line starting from cursor position (in case
                # previous line was longer). See
                # https://en.wikipedia.org/wiki/ANSI_escape_code#CSI_codes
                sys.stdout.write(chr(27) + '[K')
                sys.stdout.flush()

                self.last_print = timestamp
            else:
                pass

    @classmethod
    def from_python_file(cls,filename):
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

        # Replace default member variables with those from parameters file.
        # The following recursive approach is adapted from
        # See http://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
        def update(d, u, l=1):
            for k, v in u.items():
                # We need to step into InputSection objects, as those may have varying
                # numbers of parameters defined.
                if isinstance(v, InputSection):
                    r = update(d.get(k, {}).__dict__, v.__dict__, l+1)
                    d[k].__dict__ = r
                    #d[k] = r
                # We only want to copy contents of the input sections
                # No need to copy any of the builtin attributes added
                elif l > 1:
                    d[k] = u[k]
            return d

        self.__dict__ = update(self.__dict__, pm.__dict__)

        self.filename = filename

    ##########################################
    #######   Here add derived parameters ####
    ##########################################

    @property
    def output_dir(self):
        """Returns full path to output directory
        """
        return 'outputs/{}'.format(self.run.name)


    ##########################################
    #######  Running the input file       ####
    ##########################################

    def make_dirs(self):
        """Set up ouput directory structure"""
        import os
        import shutil
        import errno
        pm = self

        def mkdir_p(path):
            try:
                os.makedirs(path)
            except OSError as exc:
                if exc.errno == errno.EEXIST and os.path.isdir(path):
                    pass
                else: raise

        output_dirs = ['data', 'raw', 'plots', 'animations']
        for d in output_dirs:
            path = '{}/{}'.format(pm.output_dir,d)
            mkdir_p(path)
            setattr(pm,d,path)

        # Copy parameters file to output folder, if there is one
        if os.path.isfile(pm.filename):
            shutil.copy2(pm.filename,pm.output_dir)

        # Copy ViDEO file to output folder
        vfile = 'scripts/ViDEO.py'
        if os.path.isfile(vfile):
            # Note: this doesn't work, when using iDEA as a system module
            shutil.copy2('scripts/ViDEO.py',pm.output_dir)
        else:
            pass
            # No longer needed as ViDEO.py is in scrips directory and can be added to PATH
            #s  = "Warning: Unable to copy ViDEO.py since running iDEA as python module."
            #s += " Simply add the scripts folder to your PATH variable to use ViDEO.py anywhere"
            #pm.sprint(s,1)


    def setup_space(self):
        """Prepares for performing calculations

        precomputes quantities on grids, etc.
        """
        self.space = SpaceGrid(self)


    def execute(self):
        """Run this job"""
        pm = self

        pm.check()
        pm.setup_space()

        if pm.run.save:
            pm.make_dirs()
        self.results = rs.Results()

        # Draw splash to screen
        from . import splash
        splash.draw(pm)
        pm.sprint('run name: {}'.format(pm.run.name),1)

        results = pm.results
        # Execute required jobs
        if(pm.run.NON == True):
              from . import NON
              results.add(NON.main(pm), name='non')
        if(pm.non.RE == True):
              from . import RE
              results.add(RE.main(pm,'non'), name='nonre')

        if(pm.run.LDA == True):
              from . import LDA
              results.add(LDA.main(pm), name='lda')

        if(pm.run.HF == True):
              from . import HF
              results.add(HF.main(pm), name='hf')
        if(pm.hf.RE == True):
              from . import RE
              results.add(RE.main(pm,'hf'), name='hfre')

        if(pm.sys.NE == 1):
           if(pm.run.EXT == True):
              from . import EXT1
              results.add(EXT1.main(pm), name='ext')
           if(pm.ext.RE == True):
              from . import RE
              results.add(RE.main(pm,'ext'), name='extre')
        elif(pm.sys.NE == 2):
           if(pm.run.EXT == True):
              from . import EXT2
              results.add(EXT2.main(pm), name='ext')
           if(pm.ext.RE == True):
              from . import RE
              results.add(RE.main(pm,'ext'), name='extre')
        elif(pm.sys.NE == 3):
           if(pm.run.EXT == True):
              from . import EXT3
              results.add(EXT3.main(pm), name='ext')
           if(pm.ext.RE == True):
              from . import RE
              results.add(RE.main(pm,'ext'), name='extre')
        elif(pm.sys.NE >= 4):
           if(pm.run.EXT == True):
              print('EXT: cannot run exact with more than 3 electrons')
        if(pm.run.HYB == True):
              from . import HYB
              results.add(HYB.main(pm), name='hyb')
        if(pm.hyb.RE == True):
              from . import RE
              results.add(RE.main(pm,'hyb{}'.format(pm.hyb.alpha).replace('.','_')), name='hybre')

        # All jobs done
        if pm.run.save:
            # store log in file
            f = open(pm.output_dir + '/iDEA.log', 'w')
            f.write(pm.log)
            f.close()

            # need to get rid of nested functions as they can't be pickled
            tmp = copy.deepcopy(pm)
            del tmp.sys.v_ext
            del tmp.sys.v_pert

            # store pickled version of parameters object
            import pickle
            f = open(pm.output_dir + '/parameters.p', 'wb')
            pickle.dump(tmp, f, protocol=4)
            f.close()

            del tmp

        results.log = pm.log
        pm.log = ''  # avoid appending, when pm is run again

        string = 'all jobs done \n'
        pm.sprint(string,1)

        return results

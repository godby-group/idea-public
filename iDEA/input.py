""" Stores input parameters for iDEA calculations.
"""
from __future__ import print_function
import numpy as np
import importlib
import os
import pprint
import sys
import collections

import results as rs

def input_string(key,value):
    """Prints a line of the input file"""
    if isinstance(value, basestring):
        s = "{} = '{}'\n".format(key, value)
    else:
        s = "{} = {}\n".format(key, value)
    return s


class InputSection():
   """Generic section of input file"""

   def __str__(self):
       """Print variables of section and their values"""
       s = ""
       v = vars(self)
       for key,value in v.iteritems():
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
        return 1.0*self.tmax/(self.imax)

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

        ### Run parameters
        self.run = InputSection()
        run = self.run
        run.name = 'run_name'                #: Name to identify run. Note: Do not use spaces or any special characters (.~[]{}<>?/\) 
        run.time_dependence = False          #: whether to run time-dependent calculation
        run.verbosity = 'default'            #: output verbosity ('low', 'default', 'high')
        run.save = True                      #: whether to save results to disk when they are generated
        run.module = 'iDEA'                  #: specify alternative folder (in this directory) containing modified iDEA module  
        run.EXT = False                      #: Run Exact Many-Body calculation
        run.NON = False                      #: Run Non-Interacting approximation
        run.LDA = False                      #: Run LDA approximation
        run.MLP = False                      #: Run MLP approximation
        run.HF = False                       #: Run Hartree-Fock approximation
        run.MBPT = False                     #: Run Many-body pertubation theory
        run.LAN = False                      #: Run Landauer approximation 

        ### System parameters
        self.sys = SystemSection()
        sys = self.sys
        sys.NE = 2                           #: Number of electrons
        sys.grid = 201                       #: Number of grid points (must be odd)
        sys.xmax = 10.0                      #: Size of the system
        sys.tmax = 1.0                       #: Total real time
        sys.imax = 1000                      #: Number of real time iterations
        sys.acon = 1.0                       #: Smoothing of the Coloumb interaction
        sys.interaction_strength = 1         #: Scales the strength of the Coulomb interaction
        sys.im = 0                           #: Use imaginary potentials
        
        def v_ext(x):
            """Initial external potential
            """
            return 0.5*(0.25**2)*(x**2)
        sys.v_ext = v_ext
        
        def v_pert(x): 
            """Time-dependent perturbation potential
        
            Switched on at t=0.
            """
            y = -0.01*x
            if(sys.im == 1):
                return y + v_pert_im(x)
            return y
        sys.v_pert = v_pert
        
        def v_pert_im(x):                                        
            """Imaginary perturbation potential
            
            Switched on at t=0.
            """
            strength = 1.0                                      
            length_from_edge = 5.0                              
            I = sys.xmax - length_from_edge                         
            if(-sys.xmax < x and x < -I) or (sys.xmax > x and x > I):   
                return -strength*1.0j                           
            return 0.0 
        sys.v_pert_im = v_pert_im                                          
        
        
        ### Exact parameters
        self.ext = InputSection()
        ext = self.ext
        ext.par = 0                       #: Use parallelised solver and multiplication (0: serial, 1: parallel) Note: Recommend using parallel for large runs
        ext.ctol = 1e-14                                  #: Tolerance of complex time evolution (Recommended: 1e-14)
        ext.rtol = 1e-14                                  #: Tolerance of real time evolution (Recommended: 1e-14)
        ext.ctmax = 5000.0                                #: Total complex time
        ext.cimax = 1e5                                   #: Complex iterations
        ext.cdeltat = ext.ctmax/ext.cimax                 #: Complex time grid spacing (DERIVED)
        ext.RE = False                                    #: Reverse engineer many-body density
        ext.OPT = False                                   #: Calculate the external potential for the exact density
        ext.ELF_GS = False                                #: Calculate ELF for the ground-state of the system     
        ext.ELF_TD = False                                #: Calculate ELF for the time-dependent part of the system
        

        ### Non-interacting approximation parameters
        self.non = InputSection()
        non = self.non
        non.rtol = 1e-14                     #: Tolerance of real time evolution (Recommended: 1e-14)
        non.save_eig = False                 #: save eigenfunctions and eigenvalues of Hamiltonian
        non.RE = False                       #: Reverse engineer non-interacting density
        non.OPT = False                      #: Calculate the external potential for the non-interacting density
        

        ### LDA parameters
        self.lda = InputSection()
        lda = self.lda
        lda.NE = 2                           #: Number of electrons used in construction of the LDA
        lda.mix = 0.0                        #: Self consistent mixing parameter (default 0, only use if doesn't converge)
        lda.tol = 1e-12                      #: Self-consistent convergence tolerance
        lda.save_eig = False                 #: save eigenfunctions and eigenvalues of Hamiltonian
        lda.OPT = False                      #: Calculate the external potential for the LDA density 
        

        ### MLP parameters
        self.mlp = InputSection()
        mlp = self.mlp
        mlp.f = 'e'                          #: f mixing parameter (if f='e' the weight is optimzed with the elf)
        mlp.tol = 1e-12                      #: Self-consistent convergence tollerance
        mlp.mix = 0.0                        #: Self-consistent mixing parameter (default 0, only use if doesn't converge)
        mlp.reference_potential = 'non'      #: Choice of reference potential for mixing with the SOA
        mlp.OPT = False                      #: Calculate the external potential for the MLP density

        
        ### HF parameters
        self.hf = InputSection()
        hf = self.hf
        hf.fock = 1                          #: Include Fock term (0 = Hartree approximation, 1 = Hartree-Fock approximation)
        hf.con = 1e-12                       #: Tolerance
        hf.nu = 0.9                          #: Mixing term
        hf.save_eig = False                  #: save eigenfunctions and eigenvalues of Hamiltonian
        hf.RE = False                        #: Reverse-engineer hf density
        hf.OPT = False                       #: Calculate the external potential for the HF density

        
        ### MBPT parameters
        self.mbpt = InputSection()
        mbpt = self.mbpt
        mbpt.h0 = 'non'                      #: starting hamiltonian: 'non','ha','hf','lda'
        mbpt.tau_max = 40.0                  #: Maximum value of imaginary time
        mbpt.tau_npt = 800                   #: Number of imaginary time points (must be even)
        mbpt.norb = 25                       #: Number of orbitals to use
        mbpt.flavour = 'G0W0'                #: 'G0W0', 'GW', 'G0W', 'GW0'
        mbpt.den_tol = 1e-12                 #: density tolerance of self-consistent algorithm
        mbpt.max_iter = 100                  #: Maximum number of self-consistent algorithm
        mbpt.save_diag = ['sigma0_iw']       #: whether to save diagonal components of all space-time quantities
        mbpt.save_full = []                  #: which space-time quantities to save fully
        mbpt.w = 'dynamical'                 #: whether to compute 'full' or 'dynamical' W
        mbpt.hedin_shift = True              #: whether to perform Hedin shift
        mbpt.RE = False                      #: Reverse-engineer mbpt density
        mbpt.OPT = False                     #: Calculate the external potential for the MBPT density
        

        ### LAN parameters
        self.lan = InputSection()
        lan = self.lan
        lan.start = 'non'                    #: Ground-state Kohn-Sham potential to be perturbed


        ### RE parameters
        self.re = InputSection()
        re = self.re
        re.save_eig = True                   #: save Kohn-Sham eigenfunctions and eigenvalues of reverse-engineered potential


        ### OPT parameters
        self.opt = InputSection()
        opt = self.opt
        opt.tol = 1e-3                       #: Tolerance of the error in the density  
        opt.mu = 1.0                         #: 1st convergence parameter
        opt.p = 0.05                         #: 2nd convergence parameter


    def check(self):
        """Checks validity of input parameters."""
        pm = self
        if pm.run.time_dependence == True:
            if pm.run.HF == True:
                self.sprint('HF: Warning - time-dependence not implemented!')
            if pm.run.MBPT == True:
                self.sprint('MBPT: Warning - time-dependence not implemented!')

        if pm.run.MBPT == True:
            if pm.mbpt.norb < pm.sys.NE:
                self.sprint('MBPT: Warning - using {} orbitals for {} electrons'\
                        .format(pm.mbpt.norb, pm.sys.NE))


    def __str__(self):
        """Prints different sections in input file"""
        s = ""
        v = vars(self)
        for key, value in v.iteritems():
            if isinstance(value, InputSection):
                s += "### {} section\n".format(key)
                s += str(value)
                s += "\n"
            else:
                s += input_string(key,value)
        return s

    def sprint(self, string, priority=1, newline=True):
        """Customized print function

        Prints to screen and appends to log.

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
        """
        verbosity = self.run.verbosity
        self.log += string + '\n'
        if priority >= self.priority_dict[verbosity]:
            if newline:
                print(string)
            else:
                #print(string, end='\r')
                sys.stdout.write('\033[K')
                sys.stdout.flush()
                sys.stdout.write('\r' + string)
                sys.stdout.flush()


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
        # We need to step into InputSection objects, as those may have varying
        # numbers of parameters defined.
        # The following recursive approach is adapted from 
	# See http://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
	def update(d, u):
	    for k, v in u.iteritems():
		if isinstance(v, InputSection):
		    r = update(d.get(k, {}).__dict__, v.__dict__)
		    d[k].__dict__ = r
		    #d[k] = r
		else:
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

        #version = 'ver' + str(pm.run.code_version)

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
        

    def execute(self):
        """Run this job"""
        pm = self
        pm.check()
        pm.make_dirs()

        # Draw splash to screen
        import splash
        splash.draw(pm)
        pm.sprint('run name: ' + str(pm.run.name),1)

        self.results = rs.Results()
        results = self.results

        # Execute required jobs
        if(pm.sys.NE == 1):
           if(pm.run.EXT == True):
              import SPiDEA
              results.add(SPiDEA.main(pm), name='ext')
           if(pm.ext.RE == True):
              import RE
              results.add(RE.main(pm,'ext'), name='extre')
           if(pm.ext.OPT == True):
              import OPT
              results.add(OPT.main(pm,'ext'), name='extopt')
        elif(pm.sys.NE == 2):
           if(pm.run.EXT == True):
              import EXT2
              results.add(EXT2.main(pm), name='ext')
           if(pm.ext.RE == True):
              import RE
              results.add(RE.main(pm,'ext'), name='extre')
           if(pm.ext.OPT == True):
              import OPT
              results.add(OPT.main(pm,'ext'), name='extopt')
        elif(pm.sys.NE == 3):
           if(pm.run.EXT == True):
              import EXT3
              results.add(EXT3.main(pm), name='ext')
           if(pm.ext.RE == True):
              import RE
              results.add(RE.main(pm,'ext'), name='extre')
           if(pm.ext.OPT == True):
              import OPT
              results.add(OPT.main(pm,'ext'), name='extopt')
        elif(pm.sys.NE >= 4):
           if(pm.run.EXT == True):
              print('EXT: cannot run exact with more than 3 electrons')

        if(pm.run.NON == True):
              import NON
              results.add(NON.main(pm), name='non')
        if(pm.non.RE == True):
              import RE
              results.add(RE.main(pm,'non'), name='nonre')
        if(pm.non.OPT == True):
              import OPT
              results.add(OPT.main(pm,'non'), name='nonopt')

        if(pm.run.LDA == True):
              import LDA
              results.add(LDA.main(pm), name='lda')
        if(pm.lda.OPT == True):
              import OPT
              results.add(OPT.main(pm,'lda'), name='ldaopt')

        if(pm.run.MLP == True):
              import MLP
              MLP.main(pm)
        if(pm.mlp.OPT == True):
              import OPT
              results.add(OPT.main(pm,'mlp'), name='mlpopt')

        if(pm.run.HF == True):
              import HF
              results.add(HF.main(pm), name='hf')
        if(pm.hf.RE == True):
              import RE
              results.add(RE.main(pm,'hf'), name='hfre')
        if(pm.hf.OPT == True):
              import OPT
              results.add(OPT.main(pm,'hf'), name='hfopt')

        if(pm.run.MBPT == True):
              import MBPT
              results.add(MBPT.main(pm), name='mbpt')
        if(pm.mbpt.RE == True):
              import RE
              results.add(RE.main(pm,'mbpt'), name='mbptre')
        if(pm.mbpt.OPT == True):
              import OPT
              results.add(OPT.main(pm,'mbpt'), name='mbptopt')

        if(pm.run.LAN == True):
              import LAN
              results.add(LAN.main(pm), name='lan')

        # All jobs done
        # store log in file
        f = open(pm.output_dir + '/iDEA.log', 'w')
        f.write(pm.log)
        f.close()

        results.log = pm.log
        pm.log = ''

        string = 'all jobs done \n'
        pm.sprint(string,1)

        return results

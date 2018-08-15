### Library imports
from __future__ import division
from iDEA.input import InputSection, SystemSection
import numpy as np


### Run parameters
run = InputSection()
run.name = 'run_name'                #: Name to identify run. Note: Do not use spaces or any special characters (.~[]{}<>?/\)
run.time_dependence = False          #: Run time-dependent calculation
run.verbosity = 'default'            #: Output verbosity ('low', 'default', 'high')
run.save = True                      #: Save results to disk when they are generated
run.module = 'iDEA'                  #: Specify alternative folder (in this directory) containing modified iDEA module
run.NON = True                       #: Run Non-Interacting approximation
run.LDA = False                      #: Run LDA approximation
run.MLP = False                      #: Run MLP approximation
run.HF = False                       #: Run Hartree-Fock approximation
run.EXT = True                       #: Run Exact Many-Body calculation
run.HYB = False                      #: Run Hybrid (HF-LDA) calculation
run.MBPT = False                     #: Run Many-body pertubation theory
run.LAN = False                      #: Run Landauer approximation
run.MET = False                      #: Run Metric calculation


### System parameters
sys = SystemSection()
sys.NE = 2                           #: Number of electrons
sys.grid = 201                       #: Number of grid points (must be odd)
sys.stencil = 3                      #: Discretisation of 2nd derivative (3 or 5 or 7)
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
ext = InputSection()
ext.itol = 1e-12                     #: Tolerance of imaginary time propagation (Recommended: 1e-12)
ext.itol_solver = 1e-14              #: Tolerance of linear solver in imaginary time propagation (Recommended: 1e-14)
ext.rtol_solver = 1e-12              #: Tolerance of linear solver in real time propagation (Recommended: 1e-12)
ext.itmax = 2000.0                   #: Total imaginary time
ext.iimax = 1e5                      #: Imaginary time iterations
ext.ideltat = ext.itmax/ext.iimax    #: Imaginary time step (DERIVED)
ext.RE = False                       #: Reverse-engineer ext density to give DFT xc potential
ext.OPT = False                      #: Calculate the external potential for the exact density
ext.HFKS = False                     #: Reverse-engineer ext density to give HFKS c potential
ext.excited_states = 0               #: Number of excited states to calculate (0: just calculate the ground-state)
ext.elf_gs = False                   #: Calculate ELF for the ground-state of the system
ext.elf_es = False                   #: Calculate ELF for the excited-states of the system
ext.elf_td = False                   #: Calculate ELF for the time-dependent part of the system
ext.psi_gs = False                   #: Save the reduced 2 or 3 electron ground-state wavefunction to file
ext.psi_es = False                   #: Save the reduced 2 or 3 electron  excited-state wavefunctions to file
ext.initial_gspsi = 'qho'            #: Initial 2 or 3 electron ground-state wavefunction ('qho' by default. 'non' can be selected.
                                     #: 'hf', 'lda1', 'lda2', 'lda3', 'ldaheg' or 'ext' can be selected if the orbitals/wavefunction
                                     #: are available. An ext wavefunction from another run can be used, but specify the run.name
                                     #: instead e.g. 'run_name').
                                     #: WARNING: If no reliable starting guess can be provided e.g. wrong number of electrons per well,
                                     #: then choose 'qho' - this will ensure stable convergence to the true ground-state.)
ext.initial_espsi = 'qho'            #: Initial 2 or 3 electron excited-state wavefunction ('qho' by default. 'ext' can be selected if the
                                     #: wavefunctions are available. An ext wavefunction from another run can be used, but specify
                                     #: the run.name instead e.g. 'run_name').
                                     #: WARNING: If no reliable starting guess can be provided e.g. wrong number of electrons per well,
                                     #: then choose 'qho' - this will ensure stable convergence to the true ground-state.)


### Non-interacting approximation parameters
non = InputSection()
non.rtol_solver = 1e-13              #: Tolerance of linear solver in real time propagation (Recommended: 1e-13)
non.RE = False                       #: Reverse-engineer non-interacting density
non.OPT = False                      #: Calculate the external potential for the non-interacting density
non.HFKS = False                     #: Reverse-engineer non density to give HFKS c potential


### LDA parameters
lda = InputSection()
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
lda.OPT = False                      #: Calculate the external potential for the LDA density
lda.HFKS = False                     #: Reverse-engineer lda density to give HFKS c potential


### MLP parameters
mlp = InputSection()
mlp.f = 'e'                          #: f mixing parameter (if f='e' the weight is optimzed with the elf)
mlp.tol = 1e-12                      #: Self-consistent convergence tollerance
mlp.mix = 1.0                        #: Self-consistent mixing parameter
mlp.reference_potential = 'lda'      #: Choice of reference potential for mixing with the SOA
mlp.tdf = 0                          #: Time-dependent bahviour of f (if tdf = 'a' f is adiabatic, default is statis f)
mlp.TDKS = False                     #: Save the time-dependent KS potential
mlp.OPT = False                      #: Calculate the external potential for the MLP density
mlp.HFKS = False                     #: Reverse-engineer mlp density to give HFKS c potential


### HF parameters
hf = InputSection()
hf.fock = 1                          #: Include Fock term (0 = Hartree approximation, 1 = Hartree-Fock approximation)
hf.con = 1e-12                       #: Tolerance
hf.nu = 0.9                          #: Mixing term
hf.RE = False                        #: Reverse-engineer hf density
hf.OPT = False                       #: Calculate the external potential for the HF density
hf.HFKS = False                      #: Reverse-engineer hf density to give HFKS c potential


### HYB parameters
hyb = InputSection()
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
hyb.OPT = False                      #: Calculate the external potential for the LDA density
hyb.HFKS = False                     #: Reverse-engineer hyb density to give HFKS c potential


### MBPT parameters
mbpt = InputSection()
mbpt.screening = 'dynamic'           #: Approximation to P ('dynamic'=dynamic RPA, 'static'=COSEX, 'inertial'=inertial RPA,'zero'=Hartree-Fock)
mbpt.flavour = 'GW'                  #: Approximation to Sigma ('G0W0', 'GW0', 'GW')
mbpt.h0 = 'non'                      #: Starting hamiltonian: 'non','ha','hf','lda2'
mbpt.ssc = False                     #: Correct the self-screening error using our local vertex to the self-energy
mbpt.tau_max = 80.0                  #: Maximum value of imaginary time
mbpt.tau_npt = 4001                  #: Number of imaginary time points
mbpt.norb = 35                       #: Number of orbitals to use
mbpt.hedin_shift = True              #: Perform Hedin shift
mbpt.den_tol = 1e-06                 #: Density tolerance of self-consistent algorithm
mbpt.max_iter = 100                  #: Maximum iterations of self-consistent algorithm
mbpt.save_full = []                  #: Save space-time quantities (e.g. 'G0_iw', 'S1_it')
mbpt.save_zero = ['G_it','P_iw','W_iw','Sx_iw','Sxc_iw','Sc_iw'] #: save space-time quantities (e.g. 'G0_iw', 'S1_it') at iw/it=0
mbpt.save_diag = []                  #: Save diaginal components of space-time quantities
mbpt.RE = False                      #: Reverse-engineer mbpt density to give DFT xc potential
mbpt.OPT = False                     #: Calculate the external potential for the MBPT density
mbpt.HFKS = False                    #: Reverse-engineer mbpt density to give HFKS c potential


### LAN parameters
lan = InputSection()
lan.start = 'non'                    #: Ground-state Kohn-Sham potential to be perturbed


### RE parameters
re = InputSection()
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


### OPT parameters
opt = InputSection()
opt.tol = 1e-4                       #: Tolerance of the error in the density
opt.mu = 1.0                         #: 1st convergence parameter
opt.p = 0.05                         #: 2nd convergence parameter


### HFKS parameters
hfks = InputSection()
hfks.mu = 1.0                        #: 1st convergence parameter in the ground-state reverse-engineering algorithm
hfks.p = 0.05                        #: 2nd convergence parameter in the ground-state reverse-engineering algorithm
hfks.con = 1e-10                     #: Tolerance of the error in the ground-state density


### Metrics parameters
met = InputSection()
met.type  = 'wavefunction'           #: Type of the metric to be calculated ("wavefunction" or "density")
met.r_name_1 = 'run_name'            #: Run name of the first system (from run.name)
met.r_type_1 = 'non'                 #: Run type of the first system (from name of data file: eg. gs_non)
met.r_name_2 = 'run_name'            #: Run name of the second system
met.r_type_2 = 'ext'                 #: Run type of the second system
met.exact_1 = False                  #: Whether the first system is exact (not KS)
met.exact_2 = True                   #: Whether the second system is exact (not KS)

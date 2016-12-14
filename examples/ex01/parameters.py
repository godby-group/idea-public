# Library imports
from __future__ import division
from iDEA.input import InputSection, SystemSection

### run parameters
run = InputSection()
run.name = 'run_name'       #: Name to identify run. Note: Do not use spaces or any special characters (.~[]{}<>?/\) 
run.time_dependence = False #: whether to run time-dependent calculation
run.verbosity = 'default'   #: output verbosity ('low', 'default', 'high')
run.save = True             #: whether to save results to disk when they are generated

run.EXT = True              #: Run Exact Many-Body calculation
run.NON = True              #: Run Non-Interacting approximation
run.LDA = False             #: Run LDA approximation
run.MLP = False             #: Run MLP approximation
run.HF = False              #: Run Hartree-Fock approximation
run.MBPT = False            #: Run Many-body pertubation theory
run.LAN = False             #: Run Landauer approximation



### system parameters
sys = SystemSection()
sys.NE = 2                  #: Number of electrons
sys.grid = 201              #: Number of grid points (must be odd)
sys.xmax = 10.0             #: Size of the system
sys.tmax = 1.0              #: Total real time
sys.imax = 1000             #: Number of real time iterations
sys.acon = 1.0              #: Smoothing of the Coloumb interaction
sys.interaction_strength = 1#: Scales the strength of the Coulomb interaction
sys.im = 0                  #: Use imaginary potentials

def v_ext(x):
    """Initial external potential"""
    return 0.5*(0.25**2)*(x**2)
sys.v_ext = v_ext

def v_pert(x): 
    """Time-dependent perturbation potential

    Switched on at t=0.
    """
    y = -0.1*x
    if(sys.im == 1):
        return y + im_petrb(x)
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
ext = InputSection()
ext.par = 0            #: Use parallelised solver and multiplication (0: serial, 1: parallel) Note: Recommend using parallel for large runs
ext.ctol = 1e-14       #: Tolerance of complex time evolution (Recommended: 1e-14)
ext.rtol = 1e-14       #: Tolerance of real time evolution (Recommended: 1e-14)
ext.ctmax = 1000.0     #: Total complex time
ext.cimax = 100000     #: Complex iterations
ext.cdeltat = ext.ctmax/(ext.cimax-1)    #: Complex Time Grid spacing (DERIVED)
ext.RE = False         #: Reverse engineer many-body density


### Non-Interacting approximation parameters
non = InputSection()
non.rtol = 1e-14        #: Tolerance of real time evolution (Recommended: 1e-14)
non.save_eig = True     #: save eigenfunctions and eigenvalues of Hamiltonian
non.RE = False          #: Reverse engineer non-interacting density

### LDA parameters
lda = InputSection()
lda.NE = 2              #: Number of electrons used in construction of the LDA
lda.mix = 0.0           #: Self consistent mixing parameter (default 0, only use if doesn't converge)
lda.tol = 1e-12         #: Self-consistent convergence tolerance
lda.save_eig = False    #: save eigenfunctions and eigenvalues of Hamiltonian

### MLP parameters
mlp = InputSection()
mlp.f = 'e'             #: f mixing parameter (if f='e' the weight is optimzed with the elf)
mlp.tol = 1e-12         #: Self-consistent convergence tollerance
mlp.mix = 0.0           #: Self consistent mixing parameter (default 0, only use if doesn't converge)
mlp.reference_potential = 'non'      #: Choice of refernce potential for mixing with the SOA

### HF parameters
hf = InputSection()
hf.fock = 1             #: Include Fock term (0 = Hartree approximation, 1 = Hartree-Fock approximation)
hf.con = 1e-12          #: Tolerance
hf.nu = 0.9             #: Mixing term
hf.save_eig = False    #: save eigenfunctions and eigenvalues of Hamiltonian
hf.RE = False           #: Reverse engineer hf density

### MBPT parameters
mbpt = InputSection()
mbpt.starting_orbitals = 'non'  #: Orbitals to constuct G0 from
mbpt.tau_max = 40.0             #: Maximum value of imaginary time
mbpt.tau_npt = 801              #: Number of imaginary time points (must be even)
mbpt.norb = 35                  #: Number of orbitals to use
mbpt.flavour = 'GW'             #: 'G0W0', 'GW', ...
mbpt.den_tol = 1e-06            #: density tolerance of self-consistent algorithm
mbpt.max_iter = 100             #: Maximum iterations of self-consistent algorithm
mbpt.save_full = ['P0_it']      #: save space-time quantities (e.g. 'G2_iw', 'S1_it')
mbpt.save_diag = ['W0_iw']      #: save diaginal components of space-time quantities
mbpt.w = 'dynamical'            #: compute 'full' W or 'dynamical' W-v
mbpt.hedin_shift = True         #: perform Hedin shift
mbpt.RE = False                 #: Reverse engineer mbpt density

# LAN parameters
lan = InputSection()
lan.start = 'non'               #: Ground-state Kohn-Sham potential to be perturbed

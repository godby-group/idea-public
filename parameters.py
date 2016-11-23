# Library imports
from __future__ import division
from iDEA.input import InputSection, SystemSection

### run parameters
run = InputSection()
run.name = 'run_name'       #: Name to identify run. Note: Do not use spaces or any special characters (.~[]{}<>?/\) 
run.time_dependence = False #: whether to run time-dependent calculation
run.verbosity = 'default'   #: output verbosity ('low', 'default', 'high')
run.save = True             #: whether to save results to disk when they are generated
run.module = 'iDEA'         #: specify alternative folder (in this directory) containing modified iDEA module  

run.EXT = True              #: Run Exact Many-Body calculation
run.NON = True              #: Run Non-Interacting approximation
run.LDA = False             #: Run LDA approximation
run.MLP = False             #: Run MLP approximation
run.HF = False              #: Run Hartree-Fock approximation
run.MBPT = False            #: Run Many-body pertubation theory
run.MBPT2 = True            #: Run Many-body pertubation theory
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
ext.ctmax = 10000.0    #: Total complex time
ext.cimax = int(0.1*(ext.ctmax/sys.deltat)+1)     #: Complex iterations (DERIVED)
ext.cdeltat = ext.ctmax/(ext.cimax-1)             #: Complex Time Grid spacing (DERIVED)
ext.RE = False         #: Reverse engineer many-body density


### Non-Interacting approximation parameters
non = InputSection()
non.rtol = 1e-14        #: Tolerance of real time evolution (Recommended: 1e-14)
non.save_eig = False    #: save eigenfunctions and eigenvalues of Hamiltonian
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
mbpt.tau_N = 800                #: Number of imaginary time points (must be even)
mbpt.number_empty = 25          #: Number of unoccupied orbitals to use
mbpt.self_consistent = 0        #: (0 = one-shot, 1 = fully self-consistent)
mbpt.update_w = True            #: Update screening
mbpt.tolerance = 1e-12          #: Tolerance of the self-consistent algorithm
mbpt.max_iterations = 100       #: Maximum number of iterations in full self-consistency
mbpt.RE = False                 #: Reverse engineer mbpt density

### MBPT2 parameters
mbpt = InputSection()
mbpt.h0 = 'non'                 #: starting hamiltonian: 'non','ha','hf','lda'
mbpt.tau_max = 40.0             #: Maximum value of imaginary time
mbpt.tau_npt = 800              #: Number of imaginary time points (must be even)
mbpt.norb = 25                  #: Number of orbitals to use
mbpt.flavour = 'G0W0'           #: 'G0W0', 'GW', 'G0W', 'GW0'
mbpt.den_tol = 1e-12            #: density tolerance of self-consistent algorithm
mbpt.max_iter = 100             #: Maximum number of self-consistent algorithm
mbpt.save_diag = True           #: whether to save diagonal components of all space-time quantities
mbpt.RE = False                 #: Reverse engineer mbpt density


# LAN parameters
lan = InputSection()
lan.start = 'non'               #: Ground-state Kohn-Sham potential to be perturbed

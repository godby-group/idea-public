# Library imports
from __future__ import division
#from iDEA.input import InputSection

class InputSection():
   pass


### run parameters
run = InputSection()
run.name = 'test6'           # Name to identify run. Note: Do not use spaces or any special characters (.~[]{}<>?/\) 
run.code_version = 0         # Version of iDEA to use (0: As downloaded off the git) (Global: 1.9.0)
run.time_dependence = True   # whether to run time-dependent calculation
run.verbosity = 'default'     # output verbosity ('low', 'default', 'high')
run.save = True              # whether to save results to disk when they are generated

run.EXT = False                     # Run Exact Many-Body calculation
run.NON = True                      # Run Non-Interacting approximation
run.LDA = False                    # Run LDA approximation
run.MLP = False                     # Run MLP approximation
run.HF = False                     # Run Hartree-Fock approximation
run.MBPT = True                    # Run Many-body pertubation theory
run.LAN = False                    # Run Landauer approximation



### system parameters
sys = InputSection()
sys.NE = 2                          # Number of electrons
sys.grid = 21                      # Number of grid points (must be odd)
sys.xmax = 2.0 			# Size of the system
sys.tmax = 0.1 			# Total real time
sys.imax = 100			# Number of real time iterations
sys.acon = 1.0                            # Smoothing of the Coloumb interaction
sys.interaction_strength = 1                              # Scales the strength of the Coulomb interaction
sys.deltax = 2.0*sys.xmax/(sys.grid-1)	      # Spatial Grid spacing (DERIVED)
sys.deltat = sys.tmax/(sys.imax-1)		      # Temporal Grid spacing (DERIVED)

# initial external potential
def v_ext(x):
    return 0.5*(0.25**2)*(x**2)
sys.v_ext = v_ext

# time-dependent perturbation potential begining at t=0
sys.im = 0 # Use imaginary potentials
def v_pert(x): 
    y = -0.1*x
    if(sys.im == 1):
        return y + im_petrb(x)
    return y
sys.v_pert = v_pert

# the imaginary potentials                
def v_pert_im(x):                                        
    strength = 1.0                                      
    length_from_edge = 5.0                              
    I = sys.xmax - length_from_edge                         
    if(-sys.xmax < x and x < -I) or (sys.xmax > x and x > I):   
        return -strength*1.0j                           
    return 0.0 
sys.v_pert_im = v_pert_im                                          

# Derived parameters



### Exact parameters
ext = InputSection()
ext.par = 0                         # Use parallelised solver and multiplication (0: serial, 1: parallel) Note: Recommend using parallel for large runs
ext.ctol = 1e-14                    # Tolerance of complex time evolution (Recommended: 1e-14)
ext.rtol = 1e-14                    # Tolerance of real time evolution (Recommended: 1e-14)
ext.ctmax = 10000.0			# Total complex time
ext.cimax = int(0.1*(ext.ctmax/sys.deltat)+1)     # Complex iterations (DERIVED)
ext.cdeltat = ext.ctmax/(ext.cimax-1)  	      # Complex Time Grid spacing (DERIVED)
ext.RE = False                      # Reverse engineer many-body density


### Non-Interacting approximation parameters
non = InputSection()
non.rtol = 1e-14                # Tolerance of real time evolution (Recommended: 1e-14)
non.RE = True                      # Reverse engineer non-interacting density

### LDA parameters
lda = InputSection()
lda.NE = 2                      # Number of electrons used in construction of the LDA
lda.mix = 0.0                   # Self consistent mixing parameter (default 0, only use if doesn't converge)
lda.tol = 1e-12                 # Self-consistent convergence tolerance

### MLP parameters
mlp = InputSection()
mlp.f = 'e'                         # f mixing parameter (if f='e' the weight is optimzed with the elf)
mlp.tol = 1e-12                 # Self-consistent convergence tollerance
mlp.mix = 0.0                   # Self consistent mixing parameter (default 0, only use if doesn't converge)
mlp.reference_potential = 'non'      # Choice of refernce potential for mixing with the SOA

### HF parameters
hf = InputSection()
hf.fock = 1                        # Include Fock term (0 = Hartree approximation, 1 = Hartree-Fock approximation)
hf.con = 1e-12                  # Tolerance
hf.nu = 0.9                        # Mixing term
hf.RE = False                       # Reverse engineer hf density

### MBPT parameters
mbpt = InputSection()
mbpt.starting_orbitals = 'non'       # Orbitals to constuct G0 from
mbpt.tau_max = 40.0                  # Maximum value of imaginary time
mbpt.tau_N = 800                     # Number of imaginary time points (must be even)
mbpt.number_empty = 2                      # Number of unoccupied orbitals to use
mbpt.self_consistent = 0             # (0 = one-shot, 1 = fully self-consistent)
mbpt.update_w = True                    # Update screening
mbpt.tolerance = 1e-12              # Tolerance of the self-consistent algorithm
mbpt.max_iterations = 100            # Maximum number of iterations in full self-consistency
mbpt.RE = False                     # Reverse engineer mbpt density

# LAN parameters
lan = InputSection()
lan.start = 'non'               # Ground-state Kohn-Sham potential to be perturbed


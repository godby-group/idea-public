# Library imports
import math

# Define run parameters
run_name = 'run_name'           # Name to identify run. Note: Do not use spaces or any special characters (.~[]{}<>?/\) 
code_version = 0                # Version of iDEA to use (0: As downloaded off the git) (Global: 1.5.1)
NE = 2                          # Number of electrons
TD = 0                          # Time dependance
MB = 1                          # Run Exact Many-Body calculation 
NON = 1                         # Run Non-Interacting approximation
LDA = 1                         # Run LDA approximation
MLP = 0                         # Run MLP approximation
HF = 1                          # Run Hartree-Fock approximation
MBPT = 0                        # Run Many-body pertubation theory

# Many-Body parameters
par = 0                         # Use parallelised solver and multiplication (0: serial, 1: parallel) Note: Recommend using parallel for large runs
ctol = 1e-14                    # Tolerance of complex time evolution (Recommended: 1e-14)
rtol = 1e-14                    # Tolerance of real time evolution (Recommended: 1e-14)
ctmax = 10000.0			# Total complex time
MB_RE = 0                       # Reverse engineer many-body density

# Non-Interacting approximation parameters
NON_rtol = 1e-14                # Tolerance of real time evolution (Recommended: 1e-14)
NON_RE = 0                      # Reverse engineer non-interacting density

# LDA parameters
LDA_NE = 3                      # Number of electrons used in construction of the LDA
LDA_mix = 0.01                  # Self consistent mixing parameter
LDA_tol = 1e-12                 # Tollerance of self consistency

# MLP parameters
f=0.0                           # f mixing parameter (if f='e' the weight is optimzed with the elf)
cost=0                          # Calculate cost function (must have exact density)

# HF parameters
hf_con = 1e-12                  # Tollerance
nu = 0.3                        # Mixing term
HF_RE = 0                       # Reverse engineer hf density

# MBPT parameters
tau_max = 20.0                  # Maximum value of imaginary time
tau_N = 200                     # Number of imaginary time points at either side of zero
number = 33                     # Number of unoccupied orbitals to use
self_consistent = 0             # (0 = one-shot, 1 = fully self-consistent)
tollerance = 1e-12              # Tollerance of the self-consistent algorithm
max_iterations = 50             # Maximum number of iterations in full self-consistency
update_w = 1                    # (0 = do not update w, 1 = do update w)
screening = 1                   # (0 = Hartree-Fock, 1 = GW Approximation)
MBPT_RE = 0                     # Reverse engineer mbpt density

# Define grid parameters
grid = 201                      # Number of grid points (must be an odd number)
xmax = 10.0 			# Size of the system
tmax = 100.0 			# Total real time
imax = 5000			# Number of real time iterations

#Definition of initial external potential
def well(x):
    return (0.5)*(0.25**2)*(x**2)

# Defination of the perturbation potential begining at t=0
im = 0 # Use imaginary potentials
def petrb(x): 
    y = 0.0
    if(im == 1):
        return y + im_petrb(x)
    return y

# Definition of the imaginary potentials                
def im_petrb(x):                                        
    strength = 1.0                                      
    length_from_edge = 5.0                              
    I = xmax - length_from_edge                         
    if(-xmax < x and x < -I) or (xmax > x and x > I):   
        return -strength*1.0j                           
    return 0.0                                          

# Derived parameters
jmax = grid			# Number of grid points to represent 1st electronic wavefunction
kmax = jmax 			# Number of grid points to represent 2nd electronic wavefunction
lmax = jmax             	# Number of grid points to represent 3rd electronic wavefunction
deltax = 2.0*xmax/(jmax-1)	# Spatial Grid spacing
deltat = tmax/(imax-1)		# Temporal Grid spacing
cimax = int(ctmax/deltat)+1     # Complex iterations
cdeltat = ctmax/(cimax-1)  	# Complex Time Grid spacing
acon = 1.0                      # Smoothing
antifact = 1                    # No of timesteps between antisym (should be 1)
msglvl = 1                      # (REDUNDANT: must be 1)
inte = 1                        # (REDUNDANT: must be 1)


# Library imports
import math

# Define run parameters
run_name = 'run_name'           # Name to identify run. Note: Do not use spaces or any special characters (.~[]{}<>?/\) 
code_version = 0                # Version of iDEA to use (0: As downloaded off the git) (Global: 1.8.1)
NE = 2                          # Number of electrons
TD = 0                          # Time dependance
EXT = 1                         # Run Exact Many-Body calculation
NON = 1                         # Run Non-Interacting approximation
LDA = 0                         # Run LDA approximation
MLP = 0                         # Run MLP approximation
HF = 0                          # Run Hartree-Fock approximation
MBPT = 0                        # Run Many-body pertubation theory
LAN = 0                         # Run Landauer approximation

# Exact parameters
par = 0                         # Use parallelised solver and multiplication (0: serial, 1: parallel) Note: Recommend using parallel for large runs
ctol = 1e-14                    # Tolerance of complex time evolution (Recommended: 1e-14)
rtol = 1e-14                    # Tolerance of real time evolution (Recommended: 1e-14)
ctmax = 10000.0			# Total complex time
EXT_RE = 0                      # Reverse engineer many-body density

# Non-Interacting approximation parameters
NON_rtol = 1e-14                # Tolerance of real time evolution (Recommended: 1e-14)
NON_RE = 0                      # Reverse engineer non-interacting density

# LDA parameters
LDA_NE = 2                      # Number of electrons used in construction of the LDA
LDA_mix = 0.0                   # Self consistent mixing parameter (default 0, only use if doesn't converge)
LDA_tol = 1e-12                 # Self-consistent convergence tollerance

# MLP parameters
f = 'e'                         # f mixing parameter (if f='e' the weight is optimzed with the elf)
MLP_tol = 1e-12                 # Self-consistent convergence tollerance
MLP_mix = 0.0                   # Self consistent mixing parameter (default 0, only use if doesn't converge)
refernce_potential = 'non'      # Choice of refernce potential for mixing with the SOA

# HF parameters
fock = 1                        # Include Fock term (0 = Hartree approximation, 1 = Hartree-Fock approximation)
hf_con = 1e-12                  # Tollerance
nu = 0.9                        # Mixing term
HF_RE = 0                       # Reverse engineer hf density

# MBPT parameters
starting_orbitals = 'non'       # Orbitals to constuct G0 from
tau_max = 40.0                  # Maximum value of imaginary time
tau_N = 200                     # Number of imaginary time points at either side of zero
number = 33                     # Number of unoccupied orbitals to use
self_consistent = 0             # (0 = one-shot, 1 = fully self-consistent)
update_w = 1                    # (0 = do not update w, 1 = do update w)
tollerance = 1e-12              # Tollerance of the self-consistent algorithm
max_iterations = 100            # Maximum number of iterations in full self-consistency
MBPT_RE = 0                     # Reverse engineer mbpt density

# LAN parameters
lan_start = 'non'               # Ground-state Kohn-Sham potential to be perturbed

# Define grid parameters
grid = 201                      # Number of grid points (must be an odd number)
xmax = 10.0 			# Size of the system
tmax = 2.0 			# Total real time
imax = 1001			# Number of real time iterations

#Definition of initial external potential
def well(x):
    return 0.5*(0.25**2)*(x**2)

# Defination of the perturbation potential begining at t=0
im = 0 # Use imaginary potentials
def petrb(x): 
    y = -0.1*x
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
jmax = grid			      # Number of grid points to represent 1st electronic wavefunction
kmax = jmax 			      # Number of grid points to represent 2nd electronic wavefunction
lmax = jmax             	      # Number of grid points to represent 3rd electronic wavefunction
deltax = 2.0*xmax/(jmax-1)	      # Spatial Grid spacing
deltat = tmax/(imax-1)		      # Temporal Grid spacing
cimax = int(0.1*(ctmax/deltat)+1)     # Complex iterations
cdeltat = ctmax/(cimax-1)  	      # Complex Time Grid spacing
acon = 1.0                            # Smoothing
antifact = 1                          # No of timesteps between antisym (should be 1)
msglvl = 1                            # (REDUNDANT: must be 1)
inte = 1                              # (REDUNDANT: must be 1)


# Library imports
import numpy as np
import scipy as sp
import math

# Define run parameters
run_name = 'run_name'           # Name to identify run. Note: Do not use spaces or any special characters (.~[]{}<>?/\) 
code_version = 0                # Version of iDEA to use (0: As downloaded off the git) (Global: 1.1.0)
NE = 2                          # Number of electrons
TD = 0                          # Time dependance (0: Just find system ground state, 1: Run time dependance with pertubation)
MB = 1                          # Run Many-Body  
NON = 1                         # Run Non-Interacting approximation
LDA = 1                         # Run LDA approximation
MLP = 1                         # Run MLP approximation

# Many-Body parameters
par = 0                         # Use parallelised solver and multiplication (0: serial, 1: parallel) Note: Recommend using parallel for large runs
ctol = 1e-14                    # Tolerance of complex time evolution (Recommended: 1e-14)
rtol = 1e-14                    # Tolerance of real time evolution (Recommended: 1e-14)
ctmax = 200.0			# Total complex time
MB_RE = 1                       # Reverse engineer many-body density

# Non-Interacting approximation parameters
NON_RE = 1                      # Reverse engineer non-interacting density

# LDA parameters
LDA_NE = 3                      # Number of electrons used in construction of the LDA

# Define grid parameters
grid = 201                      # Number of grid points (must be an odd number)
xmax = 20.0 			# Size of the system
tmax = 3.0 			# Total real time
imax = 1001			# Number of real time iterations

# Definition of initial external potential
def well(x):
    return 0.5*(0.25**2)*(x**2)

# Defination of the perturbation potential begining at t=0
def petrb(x):
    return -0.1*x

##############################################################
##################### Derived Parameters #####################
##############################################################

# Derived parameters
jmax = grid			# Number of grid points to represent 1st electronic wavefunction
kmax = jmax 			# Number of grid points to represent 2nd electronic wavefunction
lmax = jmax             	# Number of grid points to represent 3rd electronic wavefunction
deltax = 2.0*xmax/(jmax-1)	# Spatial Grid spacing
deltat = tmax/(imax-1)		# Temporal Grid spacing
antifact = 1                    # No of timesteps between antisym (should be 1)
cimax = int(ctmax/deltat)+1     # Complex iterations
cdeltat = ctmax/(cimax-1)  	# Complex Time Grid spacing
acon = 1                        # Smoothing
msglvl = 1                      # (REDUNDANT: must be 1)
inte = 1                        # (REDUNDANT: must be 1)

# Define array to contain Potential for all space and time
if(NE == 2):
    V1 = np.zeros((2,jmax))
    V2 = np.zeros((2,kmax))
    V = np.zeros((2,jmax,kmax))
if(NE == 3):
    V1 = np.zeros((2,jmax))
    V2 = np.zeros((2,kmax))
    V = np.zeros((2,jmax,kmax,lmax))

# Define potential array for all spacial points (2 electron)
def Potential(i,j,k):
    xk = -xmax + (k*deltax)
    xj = -xmax + (j*deltax)
    t = (i*deltat)
    if (i == 0): 	
        V[i,j,k] = well(xk) + well(xj) + inte*(1.0/(abs(xk-xj) + acon))
    else:
        V[i,j,k] = well(xk) + well(xj) + inte*(1.0/(abs(xk-xj) + acon)) + petrb(xk) + petrb(xj)
    return V[i,j,k]

# Define potential array for all spacial points (3 electron)
def Potential3(i,k,j,l):
    xk = -xmax + (k*deltax)
    xj = -xmax + (j*deltax)
    xl = -xmax + (l*deltax)
    t = (i*deltat)
    if (i == 0): 	
        V[i,k,j,l] = well(xk) + well(xj) + well(xl) + inte*(1.0/(np.abs(xk-xj) + acon)) + inte*(1.0/(np.abs(xl-xj) + acon)) + inte*(1.0/(np.abs(xk-xl) + acon))
    else:
	V[i,k,j,l] = well(xk) + well(xj) + well(xl) + inte*(1.0/(np.abs(xk-xj) + acon)) + inte*(1.0/(np.abs(xl-xj) + acon)) + inte*(1.0/(np.abs(xk-xl) + acon)) + petrb(xk) + petrb(xj) + petrb(xl)
    return V[i,k,j,l]

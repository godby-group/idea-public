#!/usr/bin/python

import numpy as np

# Define run parameters
NE = 2                          # Number of electrons (2 or 3)
TD = 1                          # Time dependance (0: Just find system ground state, 1: Run time dependance with pertubation)
TDDFT = 0                       # Run TDDFT (0: Do not run, 1: Do run)
plot = 1                        # Plot the time dependent densities (0: Do not plot, 1: Do plot)
plottime = 1                    # Timestep to plot to CDensity.dat (3 electron only)
par = 0                         # Use parallelised solver and multiplication (0: serial, 1: parallel) (Recommend using parallel for large runs)
ctol = 1e-14                    # Tolerance of complex time evolution (Recommended: 1e-14)
rtol = 1e-14                    # Tolerance of real time evolution (Recommended: 1e-14)
msglvl = 1                      # Message level (0: No printing, 1: Normal printing, 2: Verbose printing)

# Define grid parameters
grid = 150                      # Number of grid points
jmax = grid			# Number of grid points to represent 1st electronic wavefunction
kmax = jmax 			# Number of grid points to represent 2nd electronic wavefunction
lmax = jmax             	# Number of grid points to represent 3rd electronic wavefunction
imax = 1000			# Number of iterations
xmax = 15.0 			# Size of the system
deltax = 2.0*xmax/(jmax-1)	# Spatial Grid spacing
tmax = 10.0 			# Total time
ctmax = 200.0			# Complex time
deltat = tmax/(imax-1)		# Temporal Grid spacing
Buffer = 2			# Buffer for plotting
cimax = int(ctmax/deltat)+1     # Complex iterations
cdeltat = ctmax/(cimax-1)  	# Complex Time Grid spacing

# Potential Function Parameters
acon = 1			# Electron interaction smoothing parameter
inte = 1			# 1 to turn electron interaction on. 0 to turn it off.

# Antisymmetrisation
antifact = 1   			# No of timesteps between antisym, False for no antisym

# Define array to contain Potential for all space and time
if(NE == 2):
    V1 = np.zeros((2,jmax))
    V2 = np.zeros((2,kmax))
    V = np.zeros((2,jmax,kmax))
if(NE == 3):
    V1 = np.zeros((2,jmax))
    V2 = np.zeros((2,kmax))
    V = np.zeros((2,jmax,kmax,lmax))

# Definition of initial external potential
def well(x):
    return -0.75*np.exp(-0.01*(x-7.5)**4)-0.5*np.exp(-0.2*(x+7.5)**2)

# Defination of the perturbation potential begining at t=0
def petrb(x):
    return -0.1*x

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

#Define potential array for all spacial points (3 electron)
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


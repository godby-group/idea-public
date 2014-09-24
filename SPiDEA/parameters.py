# Library imports
import numpy as np
import scipy as sp

# Defined run parameters
TD = 1                     # Run time dependance (1 = run time dependance, 0 = only find ground state)
saveGround = 1             # Save the ground state density (1 = do, 0 = do not)
saveReal = 1               # Save the real time density (1 = do, 0 = do not)
saveTime = 4999            # Timestep to save real time density
animatePlot = 1            # Show animated plot of real time (1 = do, 0 = do not)      
ctol = 1e-13               # Complex time tollerance

# Defined grid parameters
N = 150                    # Number of grid points
L = 30.0 		   # Size of the system
rI = 5000	           # Number of real time iterations
rT = 100.0 		   # Total real time
cT = 500.0	           # Total complex time

# Derived grid parameters
dx = 2.0*L/(N-1)	   # Spatial Grid spacing
dt = rT/(rI-1)		   # Temporal Grid spacing
cI = int(cT/dt)+1          # Number of complex iterations
cdt = cT/(cI-1)            # Complex Time Grid spacing

# Definition of initial external potential
def Vext(x):
    return 0.5*0.25**2*x**2

# Defination of the perturbation potential begining at t=0
def Vptrb(x):
    return -0.2*x


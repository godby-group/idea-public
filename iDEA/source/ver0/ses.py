######################################################################################
# Name: Single Electron Solver                                                       #
######################################################################################
# Author(s): Matt Hodgson                                                            #
######################################################################################
# Description:                                                                       #
# Performs some required jobs for MB3                                                #
#                                                                                    #
#                                                                                    #
######################################################################################
# Notes:                                                                             #
#                                                                                    #
#                                                                                    #
#                                                                                    #
######################################################################################

# Do not run stand-alone
if(__name__ == '__main__'):
    print('do not run stand-alone')
    quit()

# Libary imports
import numpy as np
import scipy as sp
from scipy import special
from scipy import sparse
from scipy import linalg
from scipy.sparse import linalg as spla
import copy
import os
import math
from matplotlib import pyplot as plt
import parameters as pa

# Integration grid parameters
deltat = pa.deltat
tmax = pa.tmax
imax = pa.imax
jmax = pa.jmax
xmax = pa.xmax
deltax = pa.deltax
inte = pa.inte

# Construct the H matrix given the potential
def constructHamiltonian(sGrid):
    omega = 0.02
    Hmatrix = sparse.lil_matrix((jmax, jmax), dtype=np.cfloat)
    for i in range(0, jmax):
        xi = -xmax + (i*deltax)
        potElement = pa.well(xi)
        element = (1.0)/(deltax**2) + potElement 
        Hmatrix[i,i] = element
    element = (-1.0)/(2.0*(deltax**2))
    for i in range(0, jmax-1): 
        Hmatrix[i,i+1] = element
        Hmatrix[i+1,i] = element
    return Hmatrix

# Get MB ground state density
def get_MB_groundstDen():
    if (inte != 0):
	temptemp = 11
    grid = np.zeros(1001)
    H = constructHamiltonian(grid)
    phi0 = np.zeros(jmax, dtype=np.cfloat)
    phi1 = np.zeros(jmax, dtype=np.cfloat)
    phi2 = np.zeros(jmax, dtype=np.cfloat)
    vals, vecs = sp.linalg.eigh(H.todense(), eigvals=[0, 2])
    for i in range(0, jmax):
        phi0[i] = vecs[i][0]
        phi1[i] = vecs[i][1]
        phi2[i] = vecs[i][2]
    norm = np.sum(np.absolute(phi0)**2)*deltax
    phi0 = phi0 * (1.0/np.sqrt(norm))
    norm = np.sum(np.absolute(phi1)**2)*deltax
    phi1 = phi1 * (1.0/np.sqrt(norm))
    norm = np.sum(np.absolute(phi2)**2)*deltax
    phi2 = phi2 * (1.0/np.sqrt(norm))
    for phi in [phi0, phi1, phi2]:
        norm = np.sum(np.absolute(phi)**2)*deltax    
    gdstateDen = (np.absolute(phi0**2) + np.absolute(phi1**2) + np.absolute(phi2**2))
    return gdstateDen
    
    
   

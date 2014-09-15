#Libary Imports
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

#Integration grid Parameters
deltat = pa.deltat
tmax = pa.tmax
imax = pa.imax
jmax = pa.jmax
xmax = pa.xmax
deltax = pa.deltax
inte = pa.inte

def constructHamiltonian(sGrid): #Construct the H matrix given the potential
    #omega = math.sqrt(2*(0.25**3))
    omega = 0.02
    Hmatrix = sparse.lil_matrix((jmax, jmax), dtype=np.cfloat)
    #Construct diagonal elements
    for i in range(0, jmax):
        xi = -xmax + (i*deltax)
        potElement = pa.well(xi) #The potential term
        element = (1.0)/(deltax**2) + potElement 
        Hmatrix[i,i] = element
    #Construct off-diagonal elements
    element = (-1.0)/(2.0*(deltax**2))
    for i in range(0, jmax-1): 
        Hmatrix[i,i+1] = element
        Hmatrix[i+1,i] = element
    return Hmatrix

def get_MB_groundstDen():
    if (inte != 0):
        #print "[SE Warning] Finding non-interacting ground state - this will not be the true ground state"
	temptemp = 11
    #print "[SE] Finding MB Ground state"
    grid = np.zeros(1001)
    H = constructHamiltonian(grid)
    phi0 = np.zeros(jmax, dtype=np.cfloat)  #Individual electron wavefunctions
    phi1 = np.zeros(jmax, dtype=np.cfloat)
    phi2 = np.zeros(jmax, dtype=np.cfloat)
    vals, vecs = sp.linalg.eigh(H.todense(), eigvals=[0, 2])
    for i in range(0, jmax):
        phi0[i] = vecs[i][0]    #Unpack t=0 eigensols
        phi1[i] = vecs[i][1]
        phi2[i] = vecs[i][2]
    #print "[SE] Normalising sols"
    norm = np.sum(np.absolute(phi0)**2)*deltax
    phi0 = phi0 * (1.0/np.sqrt(norm))
    norm = np.sum(np.absolute(phi1)**2)*deltax
    phi1 = phi1 * (1.0/np.sqrt(norm))
    norm = np.sum(np.absolute(phi2)**2)*deltax
    phi2 = phi2 * (1.0/np.sqrt(norm))
    for phi in [phi0, phi1, phi2]:
        norm = np.sum(np.absolute(phi)**2)*deltax    
        #print "[SE]    Normalisation is {}".format(norm)
    gdstateDen = (np.absolute(phi0**2) + np.absolute(phi1**2) + np.absolute(phi2**2))
    #print "[SE] MB Ground State Constructed"
    #print "[SE] MB Ground State Normalisation :", np.sum(gdstateDen)*deltax
    return gdstateDen
    
    
   

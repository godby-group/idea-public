# coding=utf-8
######################################################################################
# Name: MLP                                                                          #
######################################################################################
# Author(s):                                                                         #
######################################################################################
# Description:                                                                       #
# Computes MLP approximations if pm.MLP=1                                            #
# MLP = pm.f*SOA + (1-pm.f)*LDA                                                      #
# elf (savin et al), cost function                                                   #
######################################################################################
# Notes: TIME DEPENDENT NOT READY YET                                                #
# When calculating density cost function (vs exact MB density) the run_name in       #
# parameter.py is expected to have the form:  name_f0.*  where 'name' is the         #
# run_name of the exact MB calculation, and '*' is the first decimal of the weight   #
#                                                                                    #
######################################################################################


# Library imports
from math import *									
from numpy import *
import numpy as np
from scipy.linalg import eig_banded, solve
import parameters as pm
import sys
import pickle 
from scipy import sparse
from scipy import special
from scipy.sparse import linalg as spla
from scipy import linalg as la
import sprint
import os
import os.path

# Parameters
jmax = pm.jmax 
imax = pm.imax
xmax = pm.xmax 
tmax = pm.tmax
L = 2.0*xmax
dx = pm.deltax
sqdx = sqrt(dx)
dt = pm.deltat
TD = pm.TD
NE = pm.NE
Mix = 0.1   
tol = 1e-12 
Cost = 1 
Run = 1

# Matrices
Psi0 = zeros((imax,jmax), dtype='complex') # Wave function for each particle						
Psi1 = zeros((imax,jmax), dtype='complex')
V_h = zeros((imax,jmax)) # Potentials
V_xc = zeros((imax,jmax)) 
V_hxc = zeros((imax,jmax)) 
n_x = zeros((imax,jmax), dtype ='float') # Charge Density
n_x_old = zeros((imax,jmax), dtype='float') 
J_x = zeros((imax,jmax)) # Current Density 
T = zeros((2,jmax), dtype='complex') # Kinetic Energy operator
T[0,:] = ones(jmax)/dx**2 								
T[1,:] = -0.5*ones(jmax)/dx**2 
V_KS = zeros((imax,jmax)) # Kohn-Sham potential
V_KS_old = zeros((imax,jmax)) 
V_ext = zeros((imax,jmax)) # External potential
CNLHS = sparse.lil_matrix((jmax,jmax),dtype='complex') # Matrix for the left hand side of the Crank Nicholson method
Mat = sparse.lil_matrix((jmax,jmax),dtype='complex')   
Matin = sparse.lil_matrix((jmax,jmax),dtype='complex') # Inverted Matrix for the right hand side of the Crank Nicholson method 
K=[Psi0[0,:],Psi1[0,:]]

ff = zeros((imax,jmax)) # f weight

if (type(pm.f)==float):
        for i in range(imax):
                for j in range(jmax):
                        ff[i,j]=pm.f
elif (pm.f =='e'):
        print('MLP_e: option not implemented yet')
        sys.exit(0)
															
# Potential Generator
def Potential(i,j=0): 
        x = -xmax + i*dx 
        if (j==0): 
            V = pm.well(x)
        else: 
            V = pm.petrb(x)
        return V

# Solve TISE
def TISE(V_KS,j=0):  					                         											
        HGS = copy(T) # Reset Hamiltonian									
        HGS[0,:] += V_KS[:]								
        K, U = eig_banded(HGS, True) # Returns eigenvalues (K) and eigenvectors (U)					 									
        Psi0[j,:] = U[:,0]/sqdx # Normalise the wave functions 							
        Psi1[j,:] = U[:,1]/sqdx
        n_x[j,:] = abs(Psi0[j,:])**2+abs(Psi1[j,:])**2 # Calculate charge density				   
        return n_x[j,:], Psi0[j,:], Psi1[j,:]

# Define function for Fourier transforming into real-space
def realspace(vector):												
	mid_k = int(0.5*(jmax-1))
	fftin = zeros(jmax-1, dtype='complex')
	fftin[0:mid_k+1] = vector[mid_k:jmax]
	fftin[jmax-mid_k:jmax-1] = vector[1:mid_k]
	fftout = fft.ifft(fftin)
	func = zeros(jmax, dtype='complex')
	func[0:jmax-1] = fftout[0:jmax-1]
	func[jmax-1] = func[0]
	return func

# Define function for Fourier transforming into k-space
def momentumspace(func): 												
	mid_k = int(0.5*(jmax-1))
	fftin = zeros(jmax-1, dtype='complex')
	fftin[0:jmax-1] = func[0:jmax-1] + 0.0j
	fftout = fft.fft(fftin)
	vector = zeros(jmax, dtype='complex')
	vector[mid_k:jmax] = fftout[0:mid_k+1]
	vector[1:mid_k] = fftout[jmax-mid_k:jmax-1]
	vector[0] = vector[jmax-1].conjugate()
	return vector

# Define function for generating the Hartree potential for a given charge density
def Hartree(n): 
	n_k = momentumspace(n)*dx							 												
	X_x = zeros(jmax)
	for i in range(jmax):
		x = i*dx-0.5*L
		X_x[i] = 1.0/(abs(x)+1.0)
	X_k = momentumspace(X_x)*dx/L
	V_k = zeros(jmax, dtype='complex')
	V_k[:] = X_k[:]*n_k[:]
	fftout = realspace(V_k).real*L/dx
	V_hx = zeros(jmax)
	V_hx[0:0.5*(jmax+1)] = fftout[0.5*(jmax-1):jmax]
	V_hx[0.5*(jmax+1):jmax-1] = fftout[1:0.5*(jmax-1)]
	V_hx[jmax-1] = V_hx[0]
	return V_hx

# LDA Exchange-Correlation
def XC(n,j=0):
        V_xc = zeros((imax,jmax))
        if (pm.LDA_NE == 1):
          V_xc[j,:] = ((-1.389 + 2.44*n[j,:] - 2.05*(n[j,:])**2)*n[j,:]**0.653)
        elif (pm.LDA_NE == 2):
          V_xc[j,:] = ((-1.19 + 1.77*n[j,:] - 1.37*(n[j,:])**2)*n[j,:]**0.604)
        elif (pm.LDA_NE == 3):
          V_xc[j,:] = ((-1.24 + 2.1*n[j,:] - 1.7*(n[j,:])**2)*n[j,:]**0.61)
        return V_xc[j,:]

# Print statements 
def PS(text): 
        sys.stdout.write('\033[K')
	sys.stdout.flush()
	sys.stdout.write('\r' + text)
	sys.stdout.flush()

# Solve the Crank Nicolson equation
def CrankNicolson(V_KS, Psi0, Psi1, n, j): 
	Mat = LHS(V_KS, j) # The Hamiltonian here is using the Kohn-Sham potential. 												
	Mat = Mat.tocsr()
	Matin = -(Mat-sparse.identity(jmax, dtype=cfloat)) + sparse.identity(jmax, dtype=cfloat)
	B0 = Matin*Psi0[j-1,:] # Solve the Crank Nicolson equation to get the wave-function at dt later.
	Psi0[j,:] = spla.spsolve(Mat, B0) 		
	B1 = Matin*Psi1[j-1,:]
	Psi1[j,:] = spla.spsolve(Mat, B1)						 										
	n[j,:] = abs(Psi0[j,:])**2+abs(Psi1[j,:])**2
	return n, Psi0, Psi1

# Left hand side of the Crank Nicolson method
def LHS(V_KS, j): 												
	for i in range(jmax):
	    CNLHS[i,i] = 1.0+0.5j*dt*(1.0/dx**2+V_KS[i])
	    if i < jmax-1:
		CNLHS[i,i+1] = -0.5j*dt*(0.5/dx)/dx
	    if i > 0:
		CNLHS[i,i-1] = -0.5j*dt*(0.5/dx)/dx
	return CNLHS

# given n, return potential V_SOA
def SOA(n,j=0):
        V_SOA = zeros((imax,jmax))
        if (j==0):
        	V_SOA[j,:] = (1.0/(4*n_x[j,:]))*gradient(gradient(n_x[j,:], dx), dx)-(1.0/(8*n_x[j,:]**2))*gradient(n_x[j,:], dx)**2
        	return V_SOA[j,:]
        else:
        	print('time dependent MLP not implemented yet')
        	sys.exit(0)
 
# Find groundstate values
if (TD==0):
	j=0
for i in range(jmax): # Initial guess for V_KS (External Potential)
    V_KS[j,i] = Potential(i,j) 
    V_KS_old[j,i] = Potential(i,j)
V_ext[j,:] = V_KS[j,:] 
n_x[j,:], Psi0[j,:] , Psi1[j,:] = TISE(V_KS[j,:],j) # Solve Schrodinger Equation initially

n_x_old[j,:] = n_x[j,:]
V_SOA = zeros((imax,jmax))
V_xc_LDA = zeros((imax,jmax))
V_LDA = zeros((imax,jmax))

V_SOA[j,:]=SOA(n_x)
V_xc_LDA[j,:]=XC(n_x)
V_h[j,:] = Hartree(n_x[j,:])
V_LDA[j,:]=V_xc_LDA[j,:]+V_ext[j,:]+V_h[j,:] 
V_KS[j,:] = ff[j,:]*V_SOA[j,:]+(1-ff[j,:])*V_LDA[j,:] # Initial V_MLP ks
#print (n_x) 
while(Cost>tol):  
    n_x[j,:], Psi0[j,:] , Psi1[j,:] = TISE(V_KS[j,:],j) # Solve Schrodinger Equation
    V_SOA[j,:]= SOA(n_x)
    V_xc_LDA[j,:]=XC(n_x)
    V_h[j,:] = Hartree(n_x[j,:])
    V_LDA[j,:]=V_xc_LDA[j,:]+V_ext[j,:]+V_h[j,:]
    V_KS[j,:] = ff[j,:]*V_SOA[j,:]+(1-ff[j,:])*V_LDA[j,:]    
    V_KS[j,:] = Mix*V_KS[j,:] + (1.0-Mix)*V_KS_old[j,:] # Mix KS potential# 
    
    Cost = sum(abs(n_x[j,:]-n_x_old[j,:])*dx)
    string = 'MLP (f='+str(pm.f)+'): ground-state KS potential: run = ' + str(Run) + ', charge density cost (convergence)= ' + str(Cost)
    PS(string)
    n_x_old[j,:] = n_x[j,:]
    V_KS_old[j,:] = V_KS[j,:]
    Run = Run + 1
print "\n"

# find a file in a path
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

# calculate cost
def cost(arr1,arr2,s=0,f=jmax):
	cost = np.sum(np.abs(arr1[s:f]-arr2[s:f]))*pm.deltax
	return cost

# open mb data
# see notes in the heades
file_name = 'outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_2gs_ext_den.db'
file_obj = open(file_name,'r')
mb_den = pickle.load(file_obj)

# Calculate cost
if (pm.cost == 1):
	n_x_e = zeros(jmax) # define n_x and K so that getElf is ok with them
	n_x_e[:] = n_x[j,:]
	cost=zeros(jmax)
	cost[:]=np.abs(mb_den[:]-n_x_e[:])*pm.deltax

#############################################################################
###############################################################################

n_ext  = zeros((imax,jmax))
Psi0_ext = zeros((imax,jmax))
Psi1_ext = zeros((imax,jmax))

# ideally we want exact MB density and psi in calculating elf
#n_ext[j,:], Psi0_ext[j,:], Psi1_ext[j,:] = TISE(V_KS_ext[j,:]) # Solve Schrodinger Equation initially
#n_ext=n_ext[0,:]
# Dobson
#Psi_ext=np.array([Psi0_ext[0,:],Psi1_ext[0,:]])
# Gross at al version of Becke-Edgecomb ELF: be-elf + current term for td calculations
#K=[Psi0_ext[0,:],Psi1_ext[0,:]]

#but we use the MLP solution
n_ext=n_x[0,:]
# Dobson
# Psi_ext=np.array([Psi0_ext[0,:],Psi1_ext[0,:]])
# Gross at al version of Becke-Edgecomb ELF: be-elf + current term for td calculations
K=[Psi0[0,:],Psi1[0,:]]

# Calculate elf
def getElf(den, KS, j=None, posDef=False):

    # The single particle kinetic energy density terms
    grad1 = np.gradient(KS[0], pm.deltax)
    grad2 = np.gradient(KS[1], pm.deltax)

    # Gradient of the density
    gradDen = np.gradient(den, pm.deltax)

    # Unscaled measure
    c = np.arange(den.shape[0])
    if j == None:
        c = (np.abs(grad1)**2 + np.abs(grad2)**2)   \
            - (1./4.)* ((np.abs(gradDen)**2)/den)
    elif (j.shape == den.shape):
        c = (np.abs(grad1)**2 + np.abs(grad2)**2)   \
            - (1./4.)* ((np.abs(gradDen)**2)/den) - (j**2)/(den)

    else:
        print "Error: Invalid Current Density given to ELF"
        print "       Either j wasn't a numpy array or it "
        print "       was of the wrong dimensions         "
        return None

    # Force a positive-definate approximation if requested
    if posDef == True:
        for i in range(den.shape[0]):
            if c[i] < 0.0:
                c[i] = 0.0

    elf = np.arange(den.shape[0])

    # Scaling reference to the homogenous electron gas
    c_h = getc_h(den)

    # Finaly scale c to make ELF
    elf = (1 + (c/c_h)**2)**(-1)

    return elf

# getc_h from iDEA-ELF
def getc_h(den):
    """ C for the 1D electron gas. Used as a scaling reference"""
    c_h = np.arange(den.shape[0])

    c_h = (1./6.)*(np.pi**2)*(den**3)

    return c_h
elf = getElf(n_ext,K)

# Output results
if (TD == 0):
   f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(NE) + 'gs_mlp_vks.db', 'w') # KS potentia$
   pickle.dump(V_KS[0,:],f)
   f.close()
   f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(NE) + 'gs_mlp_den.db', 'w') # Density
   pickle.dump(n_x[0,:],f)
   f.close()
   f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(NE) + 'gs_mlp_elf.db' , 'w') #elf
   pickle.dump(elf[:],f)
   f.close()
   if (pm.cost == 1):
    	f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(NE) + 'gs_mlp_cost.db', 'w') # cost      
        pickle.dump(cost[:],f)
        f.close()


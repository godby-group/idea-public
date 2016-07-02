######################################################################################
# Name: Hartree-Fock                                                                 #
######################################################################################
# Author(s): Matt Hodgson                                                            #
######################################################################################
# Description:                                                                       #
# Computes ground-state density of a system using the Hartree-Fock approximation     #
#                                                                                    #
#                                                                                    #
#                                                                                    #
######################################################################################
# Notes:                                                                             #
#                                                                                    #
#                                                                                    #
#                                                                                    #
######################################################################################

# Library imports
from math import *
from numpy import *
from scipy.linalg import eig_banded, solve
from scipy import linalg as la
from scipy import special
from scipy import sparse
import scipy as scipy
from scipy.sparse import linalg as spla
import parameters as pm
import pickle
import sys
import sprint

# Import parameters
msglvl = pm.msglvl
Nx = pm.jmax
Nt = pm.imax
L = 2*pm.xmax
dx = L/(Nx-1)
sqdx = sqrt(dx)
c = pm.acon
nu = pm.nu

# Initialise matrices
T = zeros((Nx,Nx), dtype='complex')	# Kinetic energy matrix
n_x = zeros(Nx)				# Charge density
n_old = zeros(Nx)			# Charge density
n_MB = zeros(Nx)			# Many-body charge density
V = zeros(Nx)				# Matrix for the Kohn-Sham potential
F = zeros((Nx,Nx),dtype='complex')      # Fock operator
V_H = zeros(Nx)
V_ext = zeros(Nx)
V_add = zeros(Nx)
U = zeros((Nx,Nx))

# Costruct the kinetic energy matrix
for i in range(Nx):
   for j in range(Nx):
      T[i,i] = 1.0/dx**2
      if i<Nx-1:
         T[i+1,i] = -0.5/dx**2
         T[i,i+1] = -0.5/dx**2

# Function to add the hartree potential in the time domain
def hartree(U,density):
   return dot(U,density)*dx

# Function to construct coulomb matrix
def coulomb():
   for i in range(Nx):
      xi = i*dx-0.5*L
      for j in range(Nx):
         xj = j*dx-0.5*L
         U[i,j] = 1.0/(abs(xi-xj) + pm.acon)
   return U

# COnstruct fock operator
def Fock(Psi, U):
   F[:,:] = 0
   for k in range(pm.NE):
      for j in range(Nx):
         for i in range(Nx):
            F[i,j] += -(conjugate(Psi[k,i])*U[i,j]*Psi[k,j])*dx
   return F

# Compute ground-state
def Groundstate(V, F):	 						
   HGS = copy(T)	
   for i in range(Nx):
      HGS[i,i] += V[i]
   HGS[:,:] += F[:,:]
   K, U = scipy.linalg.eigh(HGS)
   Psi = zeros((pm.NE,Nx), dtype='complex')
   for i in range(pm.NE):
      Psi[i,:] = U[:,i]/sqdx 
   n_x[:]=0
   for i in range(pm.NE):
      n_x[:]+=abs(Psi[i,:])**2 
   return n_x, Psi, V, K

# Construct external potential
for i in range(Nx):
   x = i*dx-0.5*L
   V[i] = pm.well(x)
V_ext[:] = V[:] 
n_x, Psi, V, K = Groundstate(V, F)
con = 1

# Construct coulomb matrix
U = coulomb()

# Calculate ground state density
while con > pm.hf_con:
   n_old[:] = n_x[:]
   V_H = hartree(U,n_x)
   F = Fock(Psi, U)
   V_add[:] = V_ext[:] + V_H[:]
   V[:] = (1-nu)*V[:] + nu*V_add[:]
   for i in range(2):		 # Smooth the edges of the system
      V[i] = V[2]
   for i in range(Nx-2,Nx):
      V[i] = V[Nx-2]
   n_x, Psi, V, K = Groundstate(V, F)
   con = sum(abs(n_x[:]-n_old[:]))
   string = 'HF: computing ground-state density, convergence = ' + str(con)
   sprint.sprint(string,1,1,msglvl)
print

# Calculate ground state energy
E_HF = 0
for i in range(pm.NE):
   E_HF += K[i]
for i in range(Nx):
   E_HF += -0.5*(n_x[i]*V_H[i])*dx
for k in range(pm.NE):
   for i in range(Nx):
      for j in range(Nx):
         E_HF += -0.5*(conjugate(Psi[k,i])*F[i,j]*Psi[k,j])*dx
print 'HF: hartree-fock energy = %s' % E_HF.real

# Output ground state energy
output_file = open('outputs/' + str(pm.run_name) + '/data/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_hf_E.dat','w')
output_file.write(str(E_HF.real))
output_file.close()

# Output ground state density
output_file = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_hf_den.db','w')
pickle.dump(n_x,output_file)
output_file.close()

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
import sys
import math
import copy
import sprint
import pickle
import numpy as np
import scipy as sp
import parameters as pm
import scipy.linalg as spl

# Import parameters
msglvl = pm.msglvl
Nx = pm.jmax
Nt = pm.imax
L = 2*pm.xmax
dx = L/(Nx-1)
sqdx = math.sqrt(dx)
c = pm.acon
nu = pm.nu

# Initialise matrices
T = np.zeros((Nx,Nx), dtype='complex')	# Kinetic energy matrix
n_x = np.zeros(Nx)			# Charge density
n_old = np.zeros(Nx)			# Charge density
n_MB = np.zeros(Nx)			# Many-body charge density
V = np.zeros(Nx)			# Matrix for the Kohn-Sham potential
F = np.zeros((Nx,Nx),dtype='complex')   # Fock operator
V_H = np.zeros(Nx)
V_ext = np.zeros(Nx)
V_add = np.zeros(Nx)
U = np.zeros((Nx,Nx))

# Costruct the kinetic energy matrix
for i in range(Nx):
   for j in range(Nx):
      T[i,i] = 1.0/dx**2
      if i<Nx-1:
         T[i+1,i] = -0.5/dx**2
         T[i,i+1] = -0.5/dx**2

# Function to add the hartree potential in the time domain
def hartree(U,density):
   return np.dot(U,density)*dx

# Function to construct coulomb matrix
def coulomb():
   for i in range(Nx):
      xi = i*dx-0.5*L
      for j in range(Nx):
         xj = j*dx-0.5*L
         U[i,j] = 1.0/(abs(xi-xj) + pm.acon)
   return U

# Construct fock operator
def Fock(Psi, U):
   F[:,:] = 0
   for k in range(pm.NE):
      for j in range(Nx):
         for i in range(Nx):
            F[i,j] += -(np.conjugate(Psi[k,i])*U[i,j]*Psi[k,j])*dx
   return F

# Compute ground-state
def Groundstate(V, F, nu):	 						
   HGS = copy.copy(T)	
   for i in range(Nx):
      HGS[i,i] += V[i]
   if pm.fock == 1:
      HGS[:,:] += F[:,:]
   K, U = spl.eigh(HGS)
   Psi = np.zeros((pm.NE,Nx), dtype='complex')
   for i in range(pm.NE):
      Psi[i,:] = U[:,i]/sqdx 
   n_x[:] = 0
   for i in range(pm.NE):
      n_x[:]+=abs(Psi[i,:])**2 
   return n_x, Psi, V, K

# Construct external potential
for i in range(Nx):
   x = i*dx-0.5*L
   V[i] = pm.well(x)
V_ext[:] = V[:] 
n_x, Psi, V, K = Groundstate(V, F, nu)
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
   n_x, Psi, V, K = Groundstate(V, F, nu)
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
         E_HF += -0.5*(np.conjugate(Psi[k,i])*F[i,j]*Psi[k,j])*dx
print 'HF: hartree-fock energy = %s' % E_HF.real

# Output ground state energy
output_file = open('outputs/' + str(pm.run_name) + '/data/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_hf_E.dat','w')
output_file.write(str(E_HF.real))
output_file.close()

# Output ground state density
output_file = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_hf_den.db','w')
pickle.dump(n_x,output_file)
output_file.close()
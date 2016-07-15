######################################################################################
# Name: Landauer approximation                                                       #
#                                                                                    #
######################################################################################
# Authors: Matt Hodgson, Jack Wetherell                                              #
#                                                                                    #
######################################################################################
# Description:                                                                       #
# Computes time-dependent Landauer density                                           #
#                                                                                    #
######################################################################################
# Notes:                                                                             #
# This code requires an inital KS potential.                                         #
#                                                                                    #
######################################################################################

# Import librarys
import os
import sys
import copy
import time
import math
import pickle
import sprint
import numpy as np
import RE_Utilities
import parameters as pm
from scipy import sparse
from scipy import special
from scipy import linalg as la
from scipy.sparse import linalg as spla									
from scipy.linalg import eig_banded, solve

# Constants used in the code
sqdx=math.sqrt(pm.deltax)								
upper_bound = int((pm.jmax-1)/2.0)						
mu=1.0                                                 # Mixing for the ground-state KS algorithm
z=0
alpha=1                                                # Strength of noise control
frac1=1.0/3.0
frac2=1.0/24.0

# Initalise matrices
T = np.zeros((2,pm.jmax),dtype='complex')
T[0,:] = np.ones(pm.jmax,dtype='complex')/pm.deltax**2									
T[1,:] = -0.5*np.ones(pm.jmax,dtype='float')/pm.deltax**2									
J_LAN = np.zeros((pm.imax,pm.jmax),dtype='float')		
CNRHS = np.zeros(pm.jmax, dtype='complex')					
CNLHS = sparse.lil_matrix((pm.jmax,pm.jmax),dtype='complex')					
Mat = sparse.lil_matrix((pm.jmax,pm.jmax),dtype='complex')					
Matin = sparse.lil_matrix((pm.jmax,pm.jmax),dtype='complex')				
V_ext = np.zeros(pm.jmax,dtype='complex')
petrb = np.zeros(pm.jmax,dtype='complex')

# Function to read inputs
def ReadInput(approx):
   V = np.zeros((pm.imax,pm.jmax),dtype='complex')       # Only a ground-state to read in
   file_name='outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_' + str(approx) + '_vks.db'
   input_file=open(file_name,'r')
   data=pickle.load(input_file)
   V[0,:]=data
   return V

# Function to calculate the ground-state potential
def CalculateGroundstate(V,sqdx,T):
   HGS = np.copy(T)                                      # Build Hamiltonian
   HGS[0,:]+=V[0,:]
   K,U=eig_banded(HGS,True)                              # Solve KS equations
   Psi=np.zeros((pm.NE,2,pm.jmax), dtype='complex')
   for i in range(pm.NE):
       Psi[i,0,:] = U[:,i]/sqdx                          # Normalise
   n = np.zeros((pm.imax,pm.jmax),dtype='float')
   for i in range(pm.NE):
       n[0,:]+=abs(Psi[i,0,:])**2                        # Calculate the density from the single-particle wavefunctions 
   return n,Psi 

# Function to extrapolate the current density from regions of low density to the system's edges
def ExtrapolateCD(J,j,n,upper_bound):
   imaxl=0                                               # Start from the edge of the system
   nmaxl=0.0
   imaxr=0									
   nmaxr=0.0
   for l in range(upper_bound+1):
      if n[j,l]>nmaxl:                                   # Find the first peak in the density from the left
         nmaxl=n[j,l]
	 imaxl=l
      i=upper_bound+l-1
      if n[j,i]>nmaxr:                                   # Find the first peak in the density from the right
          nmaxr=n[j,i]
          imaxr=l
   U=np.zeros(pm.jmax)
   U[:]=J[:]/n[j,:]
   dUdx=np.zeros(pm.jmax)
   for i in range(imaxl+1):
      l=imaxl-i
      if n[j,l]<1e-8:
         dUdx[:]=np.gradient(U[:],pm.deltax)
         U[l]=8*U[l+1]-8*U[l+3]+U[l+4]+dUdx[l+2]*12.0*pm.deltax
   for i in range(int(0.5*(pm.jmax-1)-imaxr+1)):
      l=int(0.5*(pm.jmax-1)+imaxr+i)
      if n[j,l]<1e-8:
         dUdx[:]=np.gradient(U[:],pm.deltax)
         U[l]=8*U[l-1]-8*U[l-3]+U[l-4]-dUdx[l-2]*12.0*pm.deltax
   J[:]=n[j,:]*U[:]							
   return J

# Function to solve TDKSEs using the Crank-Nicolson method
def SolveKSE(V,Psi,j,frac1,frac2,z_change):
   Mat=sparse.lil_matrix((pm.jmax,pm.jmax),dtype='complex')					 						
   for i in range(pm.jmax):
      Mat[i,i]=1.0+0.5j*pm.deltat*(1.0/pm.deltax**2+V[j,i])
   for i in range(pm.jmax-1):
      Mat[i,i+1]=-0.25j*pm.deltat/pm.deltax**2
   for i in range(1,pm.jmax):
      Mat[i,i-1]=-0.25j*pm.deltat/pm.deltax**2
   Mat=Mat.tocsr()
   Matin=-(Mat-sparse.identity(pm.jmax,dtype='complex'))+sparse.identity(pm.jmax,dtype='complex')
   for i in range(pm.NE):
      B=Matin*Psi[i,z_change,:]
      z_change=z_change*(-1)+1                            # Only save two times at any point
      Psi[i,z_change,:]=spla.spsolve(Mat,B)	
      z_change=z_change*(-1)+1
   return Psi,z_change

# Function to calculate the current density
def CalculateCurrentDensity(n,upper_bound,j):
   J=RE_Utilities.continuity_eqn(pm.jmax,pm.deltax,pm.deltat,n[j,:],n[j-1,:])
   if pm.im==1:
      for j in range(pm.jmax):
         for k in range(j+1):
            x=k*pm.deltax-pm.xmax
            J[j]-=abs(pm.im_petrb(x))*n[j,k]*pm.deltax
   else:
      J=ExtrapolateCD(J,j,n,upper_bound)
   return J

# Main function
def main():
   z = 0
   V_lan = ReadInput(pm.lan_start)                        # Read in exact vks obtained from code
   n_LAN,Psi=CalculateGroundstate(V_lan,sqdx,T)
   if pm.TD==1:
      for i in range(pm.jmax):
         petrb[i]=pm.petrb((i*pm.deltax-pm.xmax))
      V_lan[:,:]=V_lan[0,:]+petrb[:]                      # Add the perturbing field to the external potential and the KS potential
      for j in range(1,pm.imax):                          # Propagate from the ground-state
         string = 'LAN: computing density and current density, time = ' + str(j*pm.deltat)
	 sprint.sprint(string,1,1,pm.msglvl)
         sprint.sprint(string,2,1,pm.msglvl)
         Psi,z=SolveKSE(V_lan,Psi,j,frac1,frac2,z)
         n_LAN[j,:]=0
         z=z*(-1)+1
         for i in range(pm.NE):
            n_LAN[j,:]+=abs(Psi[i,z,:])**2                 # Calculate the density from the single-particle wavefunctions
         J_LAN[j,:]=CalculateCurrentDensity(n_LAN,upper_bound,j)
      print
      file_name=open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_lan_den.db', 'w')
      pickle.dump(n_LAN[:,:].real,file_name)				
      file_name.close()
      file_name=open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_lan_cur.db', 'w')
      pickle.dump(J_LAN[:,:].real,file_name)				
      file_name.close()

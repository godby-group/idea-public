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
import pickle
import numpy as np
import RE_Utilities
import results as rs

import scipy.linalg as spla
import scipy.sparse as sps
import scipy.special as spec
import scipy.sparse.linalg as spsla

# Function to read inputs
def ReadInput(approx):
   V = np.zeros((pm.sys.imax,pm.sys.grid),dtype='complex')       # Only a ground-state to read in

   name = 'gs_{}_vks'.format(approx)
   data = rs.Results.read(name, pm)
   V[0,:]=data
   return V

# Function to calculate the ground-state potential
def CalculateGroundstate(V,sqdx,T):
   HGS = np.copy(T)                                      # Build Hamiltonian
   HGS[0,:]+=V[0,:]
   K,U=spla.eig_banded(HGS,True)                              # Solve KS equations
   Psi=np.zeros((pm.sys.NE,2,pm.sys.grid), dtype='complex')
   for i in range(pm.sys.NE):
       Psi[i,0,:] = U[:,i]/sqdx                          # Normalise
   n = np.zeros((pm.sys.imax,pm.sys.grid),dtype='float')
   for i in range(pm.sys.NE):
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
   U=np.zeros(pm.sys.grid)
   U[:]=J[:]/n[j,:]
   dUdx=np.zeros(pm.sys.grid)
   for i in range(imaxl+1):
      l=imaxl-i
      if n[j,l]<1e-8:
         dUdx[:]=np.gradient(U[:],pm.sys.deltax)
         U[l]=8*U[l+1]-8*U[l+3]+U[l+4]+dUdx[l+2]*12.0*pm.sys.deltax
   for i in range(int(0.5*(pm.sys.grid-1)-imaxr+1)):
      l=int(0.5*(pm.sys.grid-1)+imaxr+i)
      if n[j,l]<1e-8:
         dUdx[:]=np.gradient(U[:],pm.sys.deltax)
         U[l]=8*U[l-1]-8*U[l-3]+U[l-4]-dUdx[l-2]*12.0*pm.sys.deltax
   J[:]=n[j,:]*U[:]							
   return J

# Function to solve TDKSEs using the Crank-Nicolson method
def SolveKSE(V,Psi,j,frac1,frac2,z_change):
   Mat=sps.lil_matrix((pm.sys.grid,pm.sys.grid),dtype='complex')					 						
   for i in range(pm.sys.grid):
      Mat[i,i]=1.0+0.5j*pm.sys.deltat*(1.0/pm.sys.deltax**2+V[j,i])
   for i in range(pm.sys.grid-1):
      Mat[i,i+1]=-0.25j*pm.sys.deltat/pm.sys.deltax**2
   for i in range(1,pm.sys.grid):
      Mat[i,i-1]=-0.25j*pm.sys.deltat/pm.sys.deltax**2
   Mat=Mat.tocsr()
   Matin=-(Mat-sps.identity(pm.sys.grid,dtype='complex'))+sps.identity(pm.sys.grid,dtype='complex')
   for i in range(pm.sys.NE):
      B=Matin*Psi[i,z_change,:]
      z_change=z_change*(-1)+1                            # Only save two times at any point
      Psi[i,z_change,:]=spsla.spsolve(Mat,B)	
      z_change=z_change*(-1)+1
   return Psi,z_change

# Function to calculate the current density
def CalculateCurrentDensity(n,upper_bound,j):
   J=RE_Utilities.continuity_eqn(pm.sys.grid,pm.sys.deltax,pm.sys.deltat,n[j,:],n[j-1,:])
   if pm.sys.im==1:
      for j in range(pm.sys.grid):
         for k in range(j+1):
            x=k*pm.sys.deltax-pm.sys.xmax
            J[j]-=abs(pm.sys.v_pert_im(x))*n[j,k]*pm.sys.deltax
   else:
      J=ExtrapolateCD(J,j,n,upper_bound)
   return J

# Main function
def main(parameters):
   global sqdx, upper_bound, mu, z, alpha, frac1, frac2
   global T, J_LAN, CNRHS, CNLHS, Mat, Matin, V_ext, petrb

   global pm
   pm = parameters

   # Constants used in the code
   sqdx=np.sqrt(pm.sys.deltax)								
   upper_bound = int((pm.sys.grid-1)/2.0)						
   mu=1.0                                                 # Mixing for the ground-state KS algorithm
   z=0
   alpha=1                                                # Strength of noise control
   frac1=1.0/3.0
   frac2=1.0/24.0
   
   # Initalise matrices
   T = np.zeros((2,pm.sys.grid),dtype='complex')
   T[0,:] = np.ones(pm.sys.grid,dtype='complex')/pm.sys.deltax**2									
   T[1,:] = -0.5*np.ones(pm.sys.grid,dtype='float')/pm.sys.deltax**2									
   J_LAN = np.zeros((pm.sys.imax,pm.sys.grid),dtype='float')		
   CNRHS = np.zeros(pm.sys.grid, dtype='complex')					
   CNLHS = sps.lil_matrix((pm.sys.grid,pm.sys.grid),dtype='complex')					
   Mat = sps.lil_matrix((pm.sys.grid,pm.sys.grid),dtype='complex')					
   Matin = sps.lil_matrix((pm.sys.grid,pm.sys.grid),dtype='complex')				
   V_ext = np.zeros(pm.sys.grid,dtype='complex')
   petrb = np.zeros(pm.sys.grid,dtype='complex')


   z = 0
   string = "LAN: reading ks-potential from '{}'".format(pm.lan.start)
   pm.sprint(string,1)
   V_lan = ReadInput(pm.lan.start)                        # Read in exact vks obtained from code
   n_LAN,Psi=CalculateGroundstate(V_lan,sqdx,T)

   results = rs.Results()
   if pm.run.time_dependence==1:
      for i in range(pm.sys.grid):
         petrb[i]=pm.sys.v_pert((i*pm.sys.deltax-pm.sys.xmax))
      V_lan[:,:]=V_lan[0,:]+petrb[:]                      # Add the perturbing field to the external potential and the KS potential
      for j in range(1,pm.sys.imax):                          # Propagate from the ground-state
         string = 'LAN: computing density and current density, time = ' + str(j*pm.sys.deltat)
	 pm.sprint(string,1,newline=False)
         Psi,z=SolveKSE(V_lan,Psi,j,frac1,frac2,z)
         n_LAN[j,:]=0
         z=z*(-1)+1
         for i in range(pm.sys.NE):
            n_LAN[j,:]+=abs(Psi[i,z,:])**2                 # Calculate the density from the single-particle wavefunctions
         J_LAN[j,:]=CalculateCurrentDensity(n_LAN,upper_bound,j)
      print
      results.add(n_LAN.real,'td_lan_den')
      results.add(J_LAN.real,'td_lan_cur')
      if pm.run.save:
         results.save(pm)

   return results

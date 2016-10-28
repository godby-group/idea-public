######################################################################################
# Name: The mixed localisation potential (MLP)                                       #
######################################################################################
# Author(s): Matt Hodgson and Daniele Torelli                                        #
######################################################################################
# Description:                                                                       #
# Computes electron density and current using the MLP.                               #
######################################################################################
# Notes: Uses either the LDA or the external potential as a reference                #
# potential and mixes it with the SOA.                                               #
#                                                                                    #
######################################################################################

import pickle
import sprint
import numpy as np
import scipy as sp
import math as math
import copy as copy
import RE_Utilities
import LDA
import scipy.sparse as sps
import scipy.linalg as spla
import scipy.sparse.linalg as spsla
import results as rs

# Given n returns SOA potential
def SOA(den):
   v = np.zeros(pm.sys.grid,dtype='float')
   v = 0.25*(np.gradient(np.gradient(np.log(den),pm.sys.deltax),pm.sys.deltax))+0.125*np.gradient(np.log(den),pm.sys.deltax)**2 
   return v

# Given n returns SOA potential
def SOA_TD(den,cur,j,exp):
   pot = np.zeros(pm.sys.grid,dtype='float')
   vel_int = np.zeros(pm.sys.grid,dtype='float')
   vel = np.zeros(pm.sys.grid,dtype='float')
   vel0 = np.zeros(pm.sys.grid,dtype='float')
   pot[:] = 0.25*(np.gradient(np.gradient(np.log(den[j,:]),pm.sys.deltax),pm.sys.deltax))+0.125*np.gradient(np.log(den[j,:]),pm.sys.deltax)**2
   edge = int((0.01*pm.sys.xmax)/pm.sys.deltax)
   for i in range(1,edge):
      pot[-i] = pot[-edge]
   for i in range(0,edge):
      pot[i] = pot[15]
   vel[:] = cur[j,:]/den[j,:]
   vel0[:] = cur[j-1,:]/den[j-1,:]
   pot[:] -= 0.5*vel[:]**2
   pot = Filter(pot,j,exp) # Remove high frequencies from vector potential
   return pot

# Function to filter out 'noise' occuring between calculation of the exact TDSE solution and the present KS solution
def Filter(A,j,exp):
   A_Kspace = np.zeros(pm.sys.grid,dtype='complex')
   A_Kspace = momentumspace(A)
   A_Kspace[:] *= exp[:]
   A[:] = realspace(A_Kspace).real
   return A

# Solve ground-state KS equations
def groundstate(v):
   H = copy.copy(T)
   H[0,:] += v[:]
   e,eig_func = spla.eig_banded(H,True) 
   n = np.zeros(pm.sys.grid,dtype='float')
   for i in range(pm.sys.NE):
      n[:] += abs(eig_func[:,i])**2 # Calculate density
   n[:] /= pm.sys.deltax # Normalise
   return n,eig_func

# Define function for generating the Hartree potential for a given charge density
def Hartree(n,U):
   return np.dot(U,n)*pm.sys.deltax

# Coulomb matrix
def Coulomb():
   U = np.zeros((pm.sys.grid,pm.sys.grid),dtype='float')
   for i in xrange(pm.sys.grid):
      for k in xrange(pm.sys.grid):	
         U[i,k] = 1.0/(abs(i*pm.sys.deltax-k*pm.sys.deltax)+pm.sys.acon)
   return U

# Calculate elf
def Elf(den,KS,posDef=False):
   # The single particle kinetic energy density terms
   grad = np.zeros((pm.sys.NE,pm.sys.grid),dtype='float')
   for i in range(pm.sys.NE):
      grad[i,:] = np.gradient(KS[:,i],pm.sys.deltax)
   # Gradient of the density
   gradDen = np.gradient(den,pm.sys.deltax)
   # Unscaled measure
   c = np.arange(den.shape[0])
   for i in range(pm.sys.NE):
      c += np.abs(grad[i,:])**2
   c -= 0.25*((np.abs(gradDen)**2)/den)
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

def getc_h(den):
   c_h = np.arange(den.shape[0])
   c_h = (1.0/6.0)*(np.pi**2)*(den**3)
   return c_h

# Function to calculate the current density
def CalculateCurrentDensity(n,j):
   J = RE_Utilities.continuity_eqn(pm.sys.grid,pm.sys.deltax,pm.sys.deltat,n[j,:],n[j-1,:])
   if pm.sys.im == 1:
      for j in xrange(pm.sys.grid):
         for k in xrange(j+1):
            x = k*pm.sys.deltax-pm.sys.xmax
            J[j] -= abs(pm.sys.v_pert_im(x))*n[j,k]*pm.sys.deltax
   else:
      J = ExtrapolateCD(J,j,n,(int((pm.sys.grid-1)/2.0)))
   return J

# Solve the Crank Nicolson equation
def CrankNicolson(v,Psi,n,j,A): 
   Mat = LHS(v,j,A)
   Mat = Mat.tocsr()
   Matin = -(Mat-sps.identity(pm.sys.grid,dtype='complex'))+sps.identity(pm.sys.grid,dtype='complex')
   for i in range(pm.sys.NE):
      B = Matin*Psi[i,j-1,:]
      Psi[i,j,:] = spsla.spsolve(Mat,B)
      n[j,:] = 0
      for i in range(pm.sys.NE):
         n[j,:] += abs(Psi[i,j,:])**2
   return n,Psi

# Left hand side of the Crank Nicolson method
def LHS(v,j,A):
   frac1 = 1.0/3.0
   frac2 = 1.0/24.0
   CNLHS = sps.lil_matrix((pm.sys.grid,pm.sys.grid),dtype='complex') # Matrix for the left hand side of the Crank Nicholson method
   for i in range(pm.sys.grid):
      CNLHS[i,i] = 1.0+0.5j*pm.sys.deltat*(1.0/pm.sys.deltax**2+0.5*A[j,i]**2+v[j,i])
   for i in range(pm.sys.grid-1):
      CNLHS[i,i+1] = -0.5j*pm.sys.deltat*(0.5/pm.sys.deltax-(frac1)*1.0j*A[j,i+1]-(frac1)*1.0j*A[j,i])/pm.sys.deltax
   for i in range(1,pm.sys.grid):
      CNLHS[i,i-1] = -0.5j*pm.sys.deltat*(0.5/pm.sys.deltax+(frac1)*1.0j*A[j,i-1]+(frac1)*1.0j*A[j,i])/pm.sys.deltax
   for i in range(pm.sys.grid-2):	
      CNLHS[i,i+2] = -0.5j*pm.sys.deltat*(1.0j*A[j,i+2]+1.0j*A[j,i])*(frac2)/pm.sys.deltax
   for i in range(2,pm.sys.grid):
      CNLHS[i,i-2] = 0.5j*pm.sys.deltat*(1.0j*A[j,i-2]+1.0j*A[j,i])*(frac2)/pm.sys.deltax
   return CNLHS

# Function used in calculation of the Hatree potential
def realspace(vector):
   mid_k = int(0.5*(pm.sys.grid-1))
   fftin = np.zeros(pm.sys.grid-1,dtype='complex')
   fftin[0:mid_k+1] = vector[mid_k:pm.sys.grid]
   fftin[pm.sys.grid-mid_k:pm.sys.grid-1] = vector[1:mid_k]
   fftout = np.fft.ifft(fftin)
   func = np.zeros(pm.sys.grid, dtype='complex')
   func[0:pm.sys.grid-1] = fftout[0:pm.sys.grid-1]
   func[pm.sys.grid-1] = func[0]
   return func

# Function used in calculation of the Hatree potential
def momentumspace(func):
   mid_k = int(0.5*(pm.sys.grid-1))
   fftin = np.zeros(pm.sys.grid-1,dtype='complex')
   fftin[0:pm.sys.grid-1] = func[0:pm.sys.grid-1] + 0.0j
   fftout = np.fft.fft(fftin)
   vector = np.zeros(pm.sys.grid,dtype='complex')
   vector[mid_k:pm.sys.grid] = fftout[0:mid_k+1]
   vector[1:mid_k] = fftout[pm.sys.grid-mid_k:pm.sys.grid-1]
   vector[0] = vector[pm.sys.grid-1].conjugate()
   return vector

# Function to extrapolate the current density from regions of low density to the system's edges
def ExtrapolateCD(J,j,n,upper_bound):
   imaxl = 0 # Start from the edge of the system
   nmaxl = 0.0
   imaxr = 0									
   nmaxr = 0.0
   for l in range(upper_bound+1):
      if n[j,l]>nmaxl: # Find the first peak in the density from the left
         nmaxl = n[j,l]
	 imaxl = l
      i = upper_bound+l-1
      if n[j,i]>nmaxr: # Find the first peak in the density from the right
          nmaxr = n[j,i]
          imaxr = l
   U = np.zeros(pm.sys.grid)
   U[:] = J[:]/n[j,:]
   dUdx = np.zeros(pm.sys.grid)
   # Extraplorate the density for the low density regions
   for i in range(imaxl+1):
      l = imaxl-i
      if n[j,l]<1e-6:
         dUdx[:] = np.gradient(U[:],pm.sys.deltax)
         U[l] = 8*U[l+1]-8*U[l+3]+U[l+4]+dUdx[l+2]*12.0*pm.sys.deltax
   for i in range(int(0.5*(pm.sys.grid-1)-imaxr+1)):
      l = int(0.5*(pm.sys.grid-1)+imaxr+i)
      if n[j,l]<1e-6:
         dUdx[:] = np.gradient(U[:],pm.sys.deltax)
         U[l] = 8*U[l-1]-8*U[l-3]+U[l-4]-dUdx[l-2]*12.0*pm.sys.deltax
   J[:] = n[j,:]*U[:]							
   return J

# Main function
def main(parameters):
   global T, pm
   pm = parameters

   T = np.zeros((2,pm.sys.grid),dtype='float') # Kinetic Energy operator
   T[0,:] = np.ones(pm.sys.grid)/pm.sys.deltax**2 # Define kinetic energy operator							
   T[1,:] = -0.5*np.ones(pm.sys.grid)/pm.sys.deltax**2 


   v_s = np.zeros(pm.sys.grid,dtype='float')
   v_ext = np.zeros(pm.sys.grid,dtype='float')
   v_ref = np.zeros(pm.sys.grid,dtype='float')
   Psi = np.zeros((pm.sys.NE,pm.sys.imax,pm.sys.grid), dtype='complex')
   for i in xrange(pm.sys.grid):
      v_s[i] = pm.sys.v_ext((i*pm.sys.deltax-pm.sys.xmax)) # External potential
      v_ext[i] = pm.sys.v_ext((i*pm.sys.deltax-pm.sys.xmax)) # External potential
      if pm.mlp.reference_potential=='non':
         v_ref[i] = pm.sys.v_ext((i*pm.sys.deltax-pm.sys.xmax))
   n,waves = groundstate(v_s) #Inital guess
   U = Coulomb()
   n_old = np.zeros(pm.sys.grid,dtype='float')
   n_old[:] = n[:]
   convergence = 1.0
   while convergence>pm.mlp.tol: # Use MLP
      soa = SOA(n) # Calculate SOA potential
      v_s_old = copy.copy(v_s)
      edge = int(0.2*pm.sys.xmax)
      for i in range(1,edge):
         soa[-i] = soa[-edge]
      for i in range(0,edge):
         soa[i] = soa[edge]
      if pm.mlp.reference_potential=='lda':
         v_ref[:] = v_ext[:]+Hartree(n,U)+LDA.XC(n) # Up-date refernce potential (if needed)
      if str(pm.mlp.f)=='e':
         elf = Elf(n[:],waves)
	 average_elf = np.sum(n[:]*elf[:]*pm.sys.deltax)/pm.sys.NE
	 f_e = 2.2e-4*math.exp(8.5*average_elf) # Self-consistent f
         v_s[:] = f_e*soa[:]+(1-f_e)*v_ref[:] # Calculate MLP
      else:
         v_s[:] = pm.mlp.f*soa[:]+(1-pm.mlp.f)*v_ref[:] # Calculate MLP
      if pm.mlp.mix != 0: # Mix if needed
         v_s[:] = (1-pm.mlp.mix)*v_s_old[:]+pm.mlp.mix*v_s[:]
      n,waves = groundstate(v_s) # Calculate MLP density 
      convergence = np.sum(abs(n-n_old))*pm.sys.deltax
      n_old[:] = n[:]
      string = 'MLP: electron density convergence = ' + str(convergence)
      sprint.sprint(string,1,pm.run.verbosity,newline=False)

   v_xc = np.zeros(pm.sys.grid,dtype='float')
   v_xc[:]=v_s[:]-v_ext[:]-Hartree(n,U)

   results = rs.Results()

   if pm.run.time_dependence == False: # Output results
      results.add(v_s,name='gs_mlp_vks')
      results.add(v_xc,name='gs_mlp_vxc')
      results.add(n,name='gs_mlp_den')

      if str(pm.mlp.f)=='e':
         sprint.sprint('\nMLP: optimal f = %s' % f_e,1,pm.run.verbosity)
         results.add(elf,name='gs_mlp_elf')
      else:
         sprint.sprint('',1,pm.run.verbosity)

      if pm.run.save:
         results.save(pm.output_dir+'/raw')


   sprint.sprint('',1,pm.run.verbosity)
   if pm.run.time_dependence == True:
      for i in range(pm.sys.NE):
         Psi[i,0,:] = waves[:,i]/math.sqrt(pm.sys.deltax)
      v_s_t = np.zeros((pm.sys.imax,pm.sys.grid),dtype='float')
      v_xc_t = np.zeros((pm.sys.imax,pm.sys.grid),dtype='float')
      current = np.zeros((pm.sys.imax,pm.sys.grid),dtype='float')
      n_t = np.zeros((pm.sys.imax,pm.sys.grid),dtype='float')
      v_s_t[0,:] = v_s[:]
      n_t[0,:] = n[:]
      exp = np.zeros(pm.sys.grid,dtype='float')
      for i in xrange(pm.sys.grid): 
         v_s_t[1,i] = v_s[i]+pm.sys.v_pert((i*pm.sys.deltax-pm.sys.xmax))  
         v_ext[i] += pm.sys.v_pert((i*pm.sys.deltax-pm.sys.xmax))
         exp[i] = math.exp(-0.1*(i*pm.sys.deltax-pm.sys.xmax)**2)
      A = np.zeros((pm.sys.imax,pm.sys.grid),dtype='float')
      for j in range(1,pm.sys.imax): 
         string = 'MLP: evolving through real time: t = ' + str(j*pm.sys.deltat) 
         sprint.sprint(string,1,pm.run.verbosity,newline=False)
         n_t,Psi = CrankNicolson(v_s_t,Psi,n_t,j,A)
         current[j,:] = CalculateCurrentDensity(n_t,j)
         if j != pm.sys.imax-1:
            A[j+1,:] = -pm.mlp.f*current[j,:]/n_t[j,:]
            A[j+1,:] = Filter(A[j+1,:],j,exp)
            if pm.mlp.reference_potential=='lda':
               v_s_t[j+1,:] = (1-pm.mlp.f)*(v_ext[:]+Hartree(n_t[j,:],U)+LDA.XC(n_t[j,:]))+pm.mlp.f*SOA_TD(n_t,current,j,exp)
            if pm.mlp.reference_potential=='non':
               v_s_t[j+1,:] = (1-pm.mlp.f)*v_ext[:]+pm.mlp.f*SOA_TD(n_t,current,j,exp)

      for j in xrange(pm.sys.imax):
         for i in range(pm.sys.grid):
            for k in range(i+1):
               v_s_t[j,i] += (A[j,k]-A[j-1,k])*pm.sys.deltax/pm.sys.deltat # Convert vector potential into scalar potential


      # Output ground state density
      results.add(n_t,name='td_mlp_den')
      results.add(v_s_t,name='td_mlp_vks')

      if pm.run.save:
         # no need to save ground state quantities again...
         l = ['td_mlp_den', 'td_mlp_vks']
         results.save(pm.output_dir + '/raw',verbosity=pm.run.verbosity,list=l)

   sprint.sprint('',1,pm.run.verbosity)

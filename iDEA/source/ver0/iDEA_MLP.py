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
import iDEA_LDA as lda
import parameters as pm
import scipy.sparse as sps
import scipy.linalg as splg
import scipy.sparse.linalg as spla

T = np.zeros((2,pm.jmax),dtype='float') # Kinetic Energy operator
T[0,:] = np.ones(pm.jmax)/pm.deltax**2 # Define kinetic energy operator							
T[1,:] = -0.5*np.ones(pm.jmax)/pm.deltax**2 

# Given n returns SOA potential
def SOA(den):
   v = np.zeros(pm.jmax,dtype='float')
   v = 0.25*(np.gradient(np.gradient(np.log(den),pm.deltax),pm.deltax))+0.125*np.gradient(np.log(den),pm.deltax)**2 
   return v

# Given n returns SOA potential
def SOA_TD(den,cur,j,exp,exp2):
   pot = np.zeros(pm.jmax,dtype='float')
   vel_int = np.zeros(pm.jmax,dtype='float')
   vel = np.zeros(pm.jmax,dtype='float')
   vel0 = np.zeros(pm.jmax,dtype='float')
   pot[:] = 0.25*(np.gradient(np.gradient(np.log(den[j,:]),pm.deltax),pm.deltax))+0.125*np.gradient(np.log(den[j,:]),pm.deltax)**2
   edge = int(0.2*pm.xmax)
   for i in range(1,edge):
      pot[-i] = pot[-edge]
   for i in range(0,edge):
      pot[i] = pot[15]
   vel[:] = cur[j,:]/den[j,:]
   vel0[:] = cur[j-1,:]/den[j-1,:]
   for i in range(1,edge):
      vel[-i] = vel[-edge]
      vel0[-i] = vel[-edge]
   for i in range(0,edge):
      vel[i] = vel[edge]
      vel0[i] = vel[edge]
   for i in xrange(pm.jmax):
      for k in xrange(i+1):
         vel_int[i] -= (vel[k]-vel0[k])*pm.deltax/pm.deltat
   vel_int = Filter(vel_int,j,exp2) # Remove high frequencies from vector potential
   pot[:] -= 0.5*vel[:]**2
   pot = Filter(pot,j,exp) # Remove high frequencies from vector potential
   return pot

# Function to filter out 'noise' occuring between calculation of the exact TDSE solution and the present KS solution
def Filter(A,j,exp):
   A_Kspace = np.zeros(pm.jmax,dtype='complex')
   A_Kspace = momentumspace(A)
   A_Kspace[:] *= exp[:]
   A[:] = realspace(A_Kspace).real
   return A

# Solve ground-state KS equations
def groundstate(v):
   H = copy.copy(T)
   H[0,:] += v[:]
   e,eig_func = splg.eig_banded(H,True) 
   n = np.zeros(pm.jmax,dtype='float')
   for i in range(pm.NE):
      n[:] += abs(eig_func[:,i])**2 # Calculate density
   n[:] /= pm.deltax # Normalise
   return n,eig_func

# Define function for generating the Hartree potential for a given charge density
def Hartree(n,U):
   return np.dot(U,n)*pm.deltax

# Coulomb matrix
def Coulomb():
   U = np.zeros((pm.jmax,pm.jmax),dtype='float')
   for i in xrange(pm.jmax):
      for k in xrange(pm.jmax):	
         U[i,k] = 1.0/(abs(i*pm.deltax-k*pm.deltax)+pm.acon)
   return U

# Calculate elf
def Elf(den,KS,posDef=False):
   # The single particle kinetic energy density terms
   grad = np.zeros((pm.NE,pm.jmax),dtype='float')
   for i in range(pm.NE):
      grad[i,:] = np.gradient(KS[:,i],pm.deltax)
   # Gradient of the density
   gradDen = np.gradient(den,pm.deltax)
   # Unscaled measure
   c = np.arange(den.shape[0])
   for i in range(pm.NE):
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
   J = RE_Utilities.continuity_eqn(pm.jmax,pm.deltax,pm.deltat,n[j,:],n[j-1,:])
   if pm.im == 1:
      for j in xrange(pm.jmax):
         for k in xrange(j+1):
            x = k*pm.deltax-pm.xmax
            J[j] -= abs(pm.im_petrb(x))*n[j,k]*pm.deltax
   else:
      J = ExtrapolateCD(J,j,n,(int((pm.jmax-1)/2.0)))
   return J

# Solve the Crank Nicolson equation
def CrankNicolson(v,Psi,n,j): 
   Mat = LHS(v,j)
   Mat = Mat.tocsr()
   Matin = -(Mat-sps.identity(pm.jmax,dtype='complex'))+sps.identity(pm.jmax,dtype='complex')
   for i in range(pm.NE):
      B = Matin*Psi[i,j-1,:]
      Psi[i,j,:] = spla.spsolve(Mat,B)
      n[j,:] = 0
      for i in range(pm.NE):
         n[j,:] += abs(Psi[i,j,:])**2
   return n,Psi

# Left hand side of the Crank Nicolson method
def LHS(v,j):	
   CNLHS = sps.lil_matrix((pm.jmax,pm.jmax),dtype='complex') # Matrix for the left hand side of the Crank Nicholson method
   for i in xrange(pm.jmax):
      CNLHS[i,i] = 1.0+0.5j*pm.deltat*(1.0/pm.deltax**2+v[j,i])
      if i < pm.jmax-1:
         CNLHS[i,i+1] = -0.5j*pm.deltat*(0.5/pm.deltax**2)
      if i > 0:
         CNLHS[i,i-1] = -0.5j*pm.deltat*(0.5/pm.deltax**2)
   return CNLHS

# Function used in calculation of the Hatree potential
def realspace(vector):
   mid_k = int(0.5*(pm.jmax-1))
   fftin = np.zeros(pm.jmax-1,dtype='complex')
   fftin[0:mid_k+1] = vector[mid_k:pm.jmax]
   fftin[pm.jmax-mid_k:pm.jmax-1] = vector[1:mid_k]
   fftout = np.fft.ifft(fftin)
   func = np.zeros(pm.jmax, dtype='complex')
   func[0:pm.jmax-1] = fftout[0:pm.jmax-1]
   func[pm.jmax-1] = func[0]
   return func

# Function used in calculation of the Hatree potential
def momentumspace(func):
   mid_k = int(0.5*(pm.jmax-1))
   fftin = np.zeros(pm.jmax-1,dtype='complex')
   fftin[0:pm.jmax-1] = func[0:pm.jmax-1] + 0.0j
   fftout = np.fft.fft(fftin)
   vector = np.zeros(pm.jmax,dtype='complex')
   vector[mid_k:pm.jmax] = fftout[0:mid_k+1]
   vector[1:mid_k] = fftout[pm.jmax-mid_k:pm.jmax-1]
   vector[0] = vector[pm.jmax-1].conjugate()
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
   U = np.zeros(pm.jmax)
   U[:] = J[:]/n[j,:]
   dUdx = np.zeros(pm.jmax)
   # Extraplorate the density for the low density regions
   for i in range(imaxl+1):
      l = imaxl-i
      if n[j,l]<1e-2:
         dUdx[:] = np.gradient(U[:],pm.deltax)
         U[l] = 8*U[l+1]-8*U[l+3]+U[l+4]+dUdx[l+2]*12.0*pm.deltax
   for i in range(int(0.5*(pm.jmax-1)-imaxr+1)):
      l = int(0.5*(pm.jmax-1)+imaxr+i)
      if n[j,l]<1e-2:
         dUdx[:] = np.gradient(U[:],pm.deltax)
         U[l] = 8*U[l-1]-8*U[l-3]+U[l-4]-dUdx[l-2]*12.0*pm.deltax
   J[:] = n[j,:]*U[:]							
   return J

# Main function
def main():
   v_s = np.zeros(pm.jmax,dtype='float')
   v_ext = np.zeros(pm.jmax,dtype='float')
   v_ref = np.zeros(pm.jmax,dtype='float')
   Psi = np.zeros((pm.NE,pm.imax,pm.jmax), dtype='complex')
   for i in xrange(pm.jmax):
      v_s[i] = pm.well((i*pm.deltax-pm.xmax)) # External potential
      v_ext[i] = pm.well((i*pm.deltax-pm.xmax)) # External potential
      if pm.refernce_potential=='non':
         v_ref[i] = pm.well((i*pm.deltax-pm.xmax))
   n,waves = groundstate(v_s) #Inital guess
   U = Coulomb()
   n_old = np.zeros(pm.jmax,dtype='float')
   n_old[:] = n[:]
   convergence = 1.0
   while convergence>pm.MLP_tol: # Use MLP
      soa = SOA(n) # Calculate SOA potential
      v_s_old = copy.copy(v_s)
      edge = int(0.2*pm.xmax)
      for i in range(1,edge):
         soa[-i] = soa[-edge]
      for i in range(0,edge):
         soa[i] = soa[edge]
      if pm.refernce_potential=='lda':
         v_ref[:] = v_ext[:]+Hartree(n,U)+lda.XC(n) # Up-date refernce potential (if needed)
      if str(pm.f)=='e':
         elf = Elf(n[:],waves)
	 average_elf = np.sum(n[:]*elf[:]*pm.deltax)/pm.NE
	 f_e = 2.2e-4*math.exp(8.5*average_elf) # Self-consistent f
         v_s[:] = f_e*soa[:]+(1-f_e)*v_ref[:] # Calculate MLP
      else:
         v_s[:] = pm.f*soa[:]+(1-pm.f)*v_ref[:] # Calculate MLP
      if pm.MLP_mix != 0: # Mix if needed
         v_s[:] = (1-pm.MLP_mix)*v_s_old[:]+pm.MLP_mix*v_s[:]
      n,waves = groundstate(v_s) # Calculate MLP density 
      convergence = np.sum(abs(n-n_old))*pm.deltax
      n_old[:] = n[:]
      string = 'MLP: electron density convergence = ' + str(convergence)
      sprint.sprint(string,1,1,pm.msglvl)
      sprint.sprint(string,2,1,pm.msglvl)

   v_xc = np.zeros(pm.jmax,dtype='float')
   v_xc[:]=v_s[:]-v_ext[:]-Hartree(n,U)
   if pm.TD == 0: # Output results
      file1 = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_mlp_vks.db', 'w') # KS potential
      pickle.dump(v_s[:],file1)
      file1.close()
      file2 = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_mlp_vxc.db', 'w') # xc potential
      pickle.dump(v_xc[:],file2)
      file2.close()
      file3 = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_mlp_den.db', 'w') # density
      pickle.dump(n[:],file3)
      file3.close()
      if str(pm.f)=='e':
         print
         print 'MLP: optimal f = %s' % f_e
         file4 = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_mlp_elf.db' , 'w') # elf
         pickle.dump(elf[:],file4)
         file4.close()
      else:
         print
   print
   if pm.TD == 1:
      if str(pm.f)=='e':
         print "MLP: MLP does not support time-dependent ELF (do not use f='e')"
      else:
         for i in range(pm.NE):
            Psi[i,0,:] = waves[:,i]/math.sqrt(pm.deltax)
         v_s_t = np.zeros((pm.imax,pm.jmax),dtype='float')
         v_xc_t = np.zeros((pm.imax,pm.jmax),dtype='float')
         current = np.zeros((pm.imax,pm.jmax),dtype='float')
         n_t = np.zeros((pm.imax,pm.jmax),dtype='float')
         v_s_t[0,:] = v_s[:]
         n_t[0,:] = n[:]
         exp = np.zeros(pm.jmax,dtype='float')
         exp2 = np.zeros(pm.jmax,dtype='float')
         for i in xrange(pm.jmax): 
            v_s_t[1,i] = v_s[i]+pm.petrb((i*pm.deltax-pm.xmax))  
            v_ext[i] += pm.petrb((i*pm.deltax-pm.xmax))
            exp[i] = math.exp(-0.1*(i*pm.deltax-pm.xmax)**2)
            exp2[i] = math.exp(-0.5*(i*pm.deltax-pm.xmax)**2)
         for j in range(1,pm.imax): 
            string = 'MLP: evolving through real time: t = ' + str(j*pm.deltat) 
            sprint.sprint(string,1,1,pm.msglvl)
            sprint.sprint(string,2,1,pm.msglvl)
            n_t,Psi = CrankNicolson(v_s_t,Psi,n_t,j)
            current[j,:] = CalculateCurrentDensity(n_t,j)
            if j != pm.imax-1:
               if pm.refernce_potential=='lda':
                  v_s_t[j+1,:] = (1-pm.f)*(v_ext[:]+Hartree(n_t[j,:],U)+lda.XC(n_t[j,:]))+pm.f*SOA_TD(n_t,current,j,exp,exp2)
               if pm.refernce_potential=='non':
                  v_s_t[j+1,:] = (1-pm.f)*v_ext[:]+pm.f*SOA_TD(n_t,current,j,exp,exp2)
         file3 = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_mlp_den.db', 'w') # density
         pickle.dump(n_t,file3)
         file3.close()
         file3 = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_mlp_vks.db', 'w') # density
         pickle.dump(v_s_t,file3)
         file3.close()
   print

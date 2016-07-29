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
import iDEA_LDA as lda
import parameters as pm
import scipy.linalg as splg

T = np.zeros((2,pm.jmax),dtype='float') # Kinetic Energy operator
T[0,:] = np.ones(pm.jmax)/pm.deltax**2 # Define kinetic energy operator							
T[1,:] = -0.5*np.ones(pm.jmax)/pm.deltax**2 

# Given n returns SOA potential
def SOA(den):
   v = np.zeros(pm.jmax,dtype='float')
   v = 0.25*(np.gradient(np.gradient(np.log(den),pm.deltax),pm.deltax))+0.125*np.gradient(np.log(den),pm.deltax)**2 
   return v

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

# Main function
def main():
   v_s = np.zeros(pm.jmax,dtype='float')
   v_ext = np.zeros(pm.jmax,dtype='float')
   v_ref = np.zeros(pm.jmax,dtype='float')
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
      for i in range(1,15):
         soa[-i] = soa[-15]
      for i in range(0,15):
         soa[i] = soa[15]
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

######################################################################################
# Name: Local density approximation                                                  #
######################################################################################
# Author(s): Matt Hodgson and Mike Entwistle                                         #
######################################################################################
# Description:                                                                       #
# Computes approximations to VKS, VH, VXC using the LDA self consistently.           #
#                                                                                    #
######################################################################################
# Notes: Uses the [adiabatic] local density approximations ([A]LDA) to calculate the #
# [time-dependent] electron density [and current] for a system of N electrons.       #
#                                                                                    #
######################################################################################

import pickle
import sprint
import numpy as np
import scipy as sp
import math as math
import copy as copy
import RE_Utilities
import parameters as pm
import scipy.sparse as sps
import scipy.linalg as splg
import scipy.sparse.linalg as spla

T = np.zeros((2,pm.jmax),dtype='float') # Kinetic Energy operator
T[0,:] = np.ones(pm.jmax)/pm.deltax**2 # Define kinetic energy operator							
T[1,:] = -0.5*np.ones(pm.jmax)/pm.deltax**2 

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

# LDA approximation for XC potential
def XC(Den): 
   V_xc = np.zeros(pm.jmax,dtype='float')
   if (pm.NE == 1):
      V_xc[:] = ((-1.315+2.16*Den[:]-1.71*(Den[:])**2)*Den[:]**0.638) 
   elif (pm.NE == 2):
      V_xc[:] = ((-1.19+1.77*Den[:]-1.37*(Den[:])**2)*Den[:]**0.604) 
   else:
      V_xc[:] = ((-1.24+2.1*Den[:]-1.7*(Den[:])**2)*Den[:]**0.61) 
   return V_xc

# LDA approximation for XC energy 
def EXC(Den): 
   E_xc_LDA = 0.0
   if (pm.NE == 1):
      for i in xrange(pm.jmax):
         e_xc_LDA = ((-0.803+0.82*Den[i]-0.47*(Den[i])**2)*Den[i]**0.638) 
         E_xc_LDA += (Den[i])*(e_xc_LDA)*pm.deltax
   elif (pm.NE == 2):
      for i in xrange(pm.jmax):
         e_xc_LDA = ((-0.74+0.68*Den[i]-0.38*(Den[i])**2)*Den[i]**0.604) 
         E_xc_LDA += (Den[i])*(e_xc_LDA)*pm.deltax
   else:
      for i in xrange(pm.jmax):
         e_xc_LDA = ((-0.77+0.79*Den[i]-0.48*(Den[i])**2)*Den[i]**0.61)
         E_xc_LDA += (Den[i])*(e_xc_LDA)*pm.deltax
   return E_xc_LDA

# Function to calculate the current density
def CalculateCurrentDensity(n,j):
   J = RE_Utilities.continuity_eqn(pm.jmax,pm.deltax,pm.deltat,n[j,:],n[j-1,:])
   if pm.im == 1:
      for j in xrange(pm.jmax):
         for k in xrange(j+1):
            x = k*pm.deltax-pm.xmax
            J[j] -= abs(pm.im_petrb(x))*n[j,k]*pm.deltax
   return J

# Solve the Crank Nicolson equation
def CrankNicolson(v,Psi,n,j): 
   Mat = LHS(v,j)
   Mat = Mat.tocsr()
   Matin =- (Mat-sps.identity(pm.jmax,dtype='complex'))+sps.identity(pm.jmax,dtype='complex')
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

# Main function
def main():
   v_s = np.zeros(pm.jmax,dtype='float')
   v_ext = np.zeros(pm.jmax,dtype='float')
   Psi = np.zeros((pm.NE,pm.imax,pm.jmax), dtype='complex')
   for i in xrange(pm.jmax):
      v_s[i] = pm.well((i*pm.deltax-pm.xmax)) # External potential
      v_ext[i] = pm.well((i*pm.deltax-pm.xmax)) # External potential
   n,waves = groundstate(v_s) #Inital guess
   U = Coulomb()
   n_old = np.zeros(pm.jmax,dtype='float')
   n_old[:] = n[:]
   convergence = 1.0
   while convergence>pm.LDA_tol: # Use LDA
      v_s_old = copy.copy(v_s)
      if pm.LDA_mix == 0:
         v_s[:] = v_ext[:]+Hartree(n,U)+XC(n)
      else:
         v_s[:] = (1-pm.LDA_mix)*v_s_old[:]+pm.LDA_mix*(v_ext[:]+Hartree(n,U)+XC(n))
      n,waves = groundstate(v_s) # Calculate LDA density 
      convergence = np.sum(abs(n-n_old))*pm.deltax
      n_old[:] = n[:]
      string = 'LDA: electron density convergence = ' + str(convergence)
      sprint.sprint(string,1,1,pm.msglvl)
      sprint.sprint(string,2,1,pm.msglvl)

   print
   print 'LDA: ground-state xc energy: %s' % (EXC(n))
   v_h = Hartree(n,U)
   v_xc = XC(n)
   if pm.TD == 0: # Output results
      file1 = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_lda_vks.db', 'w') # KS potential
      pickle.dump(v_s[:],file1)
      file1.close()
      file2 = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_lda_vh.db', 'w') # H potential
      pickle.dump(v_h[:],file2)
      file2.close()
      file3 = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_lda_vxc.db', 'w') # xc potential
      pickle.dump(v_xc[:],file3)
      file3.close()
      file4 = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_lda_den.db', 'w') # density
      pickle.dump(n[:],file4)
      file4.close()

   if pm.TD == 1:
      for i in range(pm.NE):
         Psi[i,0,:] = waves[:,i]/math.sqrt(pm.deltax)
      v_s_t = np.zeros((pm.imax,pm.jmax),dtype='float')
      v_xc_t = np.zeros((pm.imax,pm.jmax),dtype='float')
      current = np.zeros((pm.imax,pm.jmax),dtype='float')
      n_t = np.zeros((pm.imax,pm.jmax),dtype='float')
      v_s_t[0,:] = v_s[:]
      n_t[0,:] = n[:]
      for i in xrange(pm.jmax): 
         v_s_t[1,i] = v_s[i]+pm.petrb((i*pm.deltax-pm.xmax))  
         v_ext[i] += pm.petrb((i*pm.deltax-pm.xmax)) 
      for j in range(1,pm.imax): 
         string = 'LDA: evolving through real time: t = ' + str(j*pm.deltat) 
         sprint.sprint(string,1,1,pm.msglvl)
         sprint.sprint(string,2,1,pm.msglvl)
         n_t,Psi = CrankNicolson(v_s_t,Psi,n_t,j)
         if j != pm.imax-1:
            v_s_t[j+1,:] = v_ext[:]+Hartree(n_t[j,:],U)+XC(n_t[j,:])
         current[j,:] = CalculateCurrentDensity(n_t,j)
         v_xc_t[j,:] = XC(n_t[j,:])

      # Output results
      file1 = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_lda_vks.db', 'w') 
      pickle.dump(v_s_t,file1)				
      file1.close()
      file2 = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_lda_vxc.db', 'w') 	
      pickle.dump(v_xc_t,file2)				
      file2.close()
      file3 = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_lda_den.db', 'w') 
      pickle.dump(n_t,file3)				
      file3.close()
      file4 = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_lda_cur.db', 'w') 
      pickle.dump(current,file4)				
      file4.close()
      print


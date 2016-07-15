######################################################################################
# Name: LDA approximation                                                            #
######################################################################################
# Author(s): Mike Entwistle, Matt Hodgson                                            #
######################################################################################
# Description:                                                                       #
# Computes approximations to VKS, VH, VXC using the LDA self consistently.           #
#                                                                                    #
#                                                                                    #
######################################################################################
# Notes:                                                                             #
#                                                                                    #
#                                                                                    #
#                                                                                    #
######################################################################################

# Import Libraries
import sys
import math
import copy
import sprint
import pickle
import scipy as sp
import numpy as np
import RE_Utilities
import parameters as pm
import scipy.linalg as sla
import scipy.sparse as sps
import scipy.sparse.linalg as spla

# Parameters
L = 2.0*pm.xmax
dx = pm.deltax
sqdx = math.sqrt(dx)
dt = pm.deltat
TD = pm.TD
NE = pm.LDA_NE
Mix = pm.LDA_mix
tol = pm.LDA_tol 
E_xc_Exact = 0 

# Matrices
V_h = np.zeros((pm.imax,pm.jmax), dtype ='float')         # Potentials
V_xc = np.zeros((pm.imax,pm.jmax), dtype ='float') 
V_hxc = np.zeros((pm.imax,pm.jmax), dtype ='float') 
n_x = np.zeros((pm.imax,pm.jmax), dtype ='float')         # Charge Density
n_x_old = np.zeros((pm.imax,pm.jmax), dtype='float') 
J_x = np.zeros((pm.imax,pm.jmax), dtype ='float')         # Current Density 
T = np.zeros((2,pm.jmax), dtype='complex')                # Kinetic Energy operator
T[0,:] = np.ones(pm.jmax)/dx**2 								
T[1,:] = -0.5*np.ones(pm.jmax)/dx**2 
V_KS = np.zeros((pm.imax,pm.jmax),dtype='complex')        # Kohn-Sham potential
V_KS_old = np.zeros((pm.imax,pm.jmax),dtype='complex') 
V_ext = np.zeros(pm.jmax,dtype='complex')                 # External potential
CNLHS = sps.lil_matrix((pm.jmax,pm.jmax),dtype='complex') # Matrix for the left hand side of the Crank Nicholson method
Mat = sps.lil_matrix((pm.jmax,pm.jmax),dtype='complex')   
Matin = sps.lil_matrix((pm.jmax,pm.jmax),dtype='complex') # Inverted Matrix for the right hand side of the Crank Nicholson method 
U = np.zeros((pm.jmax,pm.jmax))
																		
# Potential generator
def Potential(i,j): 
   x = -pm.xmax + i*dx 
   if (j==0): 
      V = pm.well(x)
   else: 
      V = pm.petrb(x) 
   return V

# Solve the time-independent Schrodinger equation
def TISE(V_KS,j):  					                         											
   HGS = copy.copy(T) 									
   HGS[0,:] += V_KS[j,:]																	
   K,U=sla.eig_banded(HGS,True)                         # Solve KS equations
   Psi = np.zeros((pm.NE,pm.imax,pm.jmax), dtype='complex')
   for i in range(pm.NE):
      Psi[i,j,:] = U[:,i]/sqdx                           # Normalise
   n_x[j,:]=0
   K_KS = 0.0
   for i in range(pm.NE):                                # Calculate density and cost function
      n_x[j,:]+=abs(Psi[i,j,:])**2                       # Calculate the density from the single-particle wavefunctions				   
      K_KS += K[i]
   return n_x, Psi, K_KS

# Calculate the Hartree potential 
def Hartree(density):                 
   return np.dot(coulomb(),density)*dx             
                                        
# Construct the Coulomb matrix  
def coulomb():                          
   for i in range(pm.jmax):                  
      xi = i*dx-0.5*L                   
      for j in range(pm.jmax):               
         xj = j*dx-0.5*L                
         U[i,j] = 1.0/(abs(xi-xj) + pm.acon)  
   return U                             
                                        
# Calculate current density
def Currentdensity(j, n):  											
   J = np.zeros(pm.jmax, dtype ='float')
   if j != 0:
      for i in range(pm.jmax):			
         for k in range(i+1):
            J[i] += -dx*(n[j,k]-n[j-1,k])/dt
   nmaxl = 0									
   imaxl = 0
   for i in range(int(0.5*(pm.jmax-1))+1):
      if n[j,i]>nmaxl:
         nmaxl = n[j,i]
         imaxl = i
   nmaxr = 0
   imaxr = 0
   for l in range(int(0.5*(pm.jmax-1))+1):
      i = int(0.5*(pm.jmax-1)+l)
      if n[j,i]>nmaxr:
         nmaxr = n[j,i]
         imaxr = l
   U = np.zeros(pm.jmax)
   U[:] = J[:]/n[j,:]	
   dUdx = np.zeros(pm.jmax)
   for i in range(imaxl+1):
      l = imaxl-i
      if n[j,l] < 1e-6:
         dUdx[:] = gradient(U[:], dx)
         U[l] = 8*U[l+1]-8*U[l+3]+U[l+4]+dUdx[l+2]*12.0*dx
   for i in range(int(0.5*(pm.jmax-1)-imaxr+1)):
      l = int(0.5*(pm.jmax-1)+imaxr+i)
      if n[j,l] < 1e-6:
         dUdx[:] = gradient(U[:], dx)
         U[l] = 8*U[l-1]-8*U[l-3]+U[l-4]-dUdx[l-2]*12.0*dx
   J[:] = n[j,:]*U[:]								 												
   return J[:]

# LDA approximation for XC potential
def XC(n): 
   V_xc = np.zeros(pm.jmax)
   if (NE == 1):
      V_xc[:] = ((-1.315 + 2.16*n[:] - 1.71*(n[:])**2)*n[:]**0.638) 
   elif (NE == 2):
      V_xc[:] = ((-1.19 + 1.77*n[:] - 1.37*(n[:])**2)*n[:]**0.604) 
   else:
      V_xc[:] = ((-1.24 + 2.1*n[:] - 1.7*(n[:])**2)*n[:]**0.61) 
   return V_xc[:]

# LDA approximation for XC energy 
def EXC(n): 
   E_xc_LDA = 0.0
   if(NE == 1):
      for i in range(pm.jmax-1):
         e_xc_LDA = ((-0.803 + 0.82*n[i] - 0.47*(n[i])**2)*n[i]**0.638) 
         increase = (n[i])*(e_xc_LDA)*dx
         E_xc_LDA += increase
   elif(NE == 2):
      for i in range(pm.jmax-1):
         e_xc_LDA = ((-0.74 + 0.68*n[i] - 0.38*(n[i])**2)*n[i]**0.604) 
         increase = (n[i])*(e_xc_LDA)*dx
         E_xc_LDA += increase
   else:
      for i in range(pm.jmax-1):
         e_xc_LDA = ((-0.77 + 0.79*n[i] - 0.48*(n[i])**2)*n[i]**0.61) 
         increase = (n[i])*(e_xc_LDA)*dx
         E_xc_LDA += increase
   return E_xc_LDA

# Error in the LDA approximation
def error(E_xc_LDA, E_xc_Exact): 
   if (E_xc_Exact != 0):
      print 'XC energy (Exact): E_xc =', (round(E_xc_Exact, 4))
      print 'XC energy (LDA)  : E_xc_LDA =', (round(E_xc_LDA, 4))
      if (E_xc_LDA <= E_xc_Exact): 
         LDA_error = E_xc_Exact - E_xc_LDA 
         LDA_error = abs(LDA_error)
         print 'Absolute error   : dE_xc =', (round(LDA_error, 4))
         LDA_error_percentage = LDA_error/E_xc_Exact
         LDA_error_percentage = LDA_error_percentage*100
         print 'E_xc_LDA is', (round(LDA_error_percentage, 2)),'% too low' 
      else:
         LDA_error = E_xc_LDA - E_xc_Exact
         print 'Absolute error   : dE_xc =', (round(LDA_error, 4))
         LDA_error_percentage = abs(LDA_error/E_xc_Exact)
         LDA_error_percentage = LDA_error_percentage*100
         print 'E_xc_LDA is', (round(LDA_error_percentage, 2)),'% too high' 
   else:
      print
      print 'LDA: exchange-correlation energy: ', (round(E_xc_LDA, 4))

# Print statements 
def PS(text): 
   sys.stdout.write('\033[K')
   sys.stdout.flush()
   sys.stdout.write('\r' + text)
   sys.stdout.flush()

# Solve the Crank Nicolson equation
def CrankNicolson(V_KS, Psi, n, j): 
   Mat = LHS(V_KS, j)
   Mat=Mat.tocsr()
   Matin=-(Mat-sps.identity(pm.jmax,dtype='complex'))+sps.identity(pm.jmax,dtype='complex')
   for i in range(pm.NE):
      B=Matin*Psi[i,j-1,:]
      Psi[i,j,:]=spla.spsolve(Mat,B)
      n[j,:]=0
      for i in range(pm.NE):
         n[j,:]+=abs(Psi[i,j,:])**2
   return n, Psi

# Left hand side of the Crank Nicolson method
def LHS(V_KS, j):											
   for i in range(pm.jmax):
      CNLHS[i,i] = 1.0+0.5j*dt*(1.0/dx**2+V_KS[i])
      if i < pm.jmax-1:
         CNLHS[i,i+1] = -0.5j*dt*(0.5/dx)/dx
      if i > 0:
         CNLHS[i,i-1] = -0.5j*dt*(0.5/dx)/dx
   return CNLHS

# Function to calculate the current density
def calculateCurrentDensity(total_td_density):
   current_density = []
   for i in range(0,len(total_td_density)-1):
      string = 'LDA: computing time-dependent current density t = ' + str(i*pm.deltat)
      sprint.sprint(string,1,1,pm.msglvl)
      J = np.zeros(pm.jmax)
      J = RE_Utilities.continuity_eqn(pm.jmax,dx,pm.deltat,total_td_density[i+1],total_td_density[i])
      if pm.im==1:
         for j in range(pm.jmax):
            for k in range(j+1):
               x = k*dx-pm.xmax
               J[j] -= abs(pm.im_petrb(x))*total_td_density[i][k]*dx
      current_density.append(J)
   return current_density

# Main function
def main():
   j = 0
   K_KS = 0.0
   for i in range(pm.jmax):                               # Initial guess for V_KS (External Potential)
      V_KS[j,i] = Potential(i,j) 
      V_KS_old[j,i] = Potential(i,j)
   V_ext[:] = V_KS[j,:] 
   n_x, Psi, K_KS  = TISE(V_KS,j)               # Solve Schrodinger Equation initially
   n_x_old[j,:] = n_x[j,:]
   Cost = 1
   Run = 1
   while(Cost>tol):  
      V_h[j,:] = Hartree(n_x[j,:])                        # Calculate Hartree, XC and KS potential
      V_xc[j,:] = XC(n_x[j,:])
      V_hxc[j,:] = V_h[j,:] + V_xc[j,:]
      V_KS[j,:] = V_ext[:] + V_hxc[j,:]
      V_KS[j,:] = Mix*V_KS[j,:] + (1.0-Mix)*V_KS_old[j,:] # Mix KS potential
      n_x, Psi, K_KS = TISE(V_KS,j)             # Solve Schrodinger Equation
      Cost = sum(abs(n_x[j,:]-n_x_old[j,:])*dx)
      string = 'LDA: computing ground-state Kohn-Sham potential: run = ' + str(Run) + ', convergence = ' + str(Cost)
      PS(string)
      n_x_old[j,:] = n_x[j,:]
      V_KS_old[j,:] = V_KS[j,:]
      Run = Run + 1
   V_h[j,:] = Hartree(n_x[j,:])
   V_xc[j,:] = XC(n_x[j,:])
   V_hxc[j,:] = V_h[j,:] + V_xc[j,:]
   E_xc_LDA = EXC(n_x[j,:]) 
   error(E_xc_LDA, E_xc_Exact)
   E_Total = 0.0
   E_Total += E_xc_LDA + K_KS
   for i in range(pm.jmax):
      E_Total -= n_x[0,i]*(0.50*V_h[0,i] + V_xc[0,i])*dx
   print 'Energy total =', E_Total
   f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_lda_vks.db', 'w')
   pickle.dump(V_KS[0,:],f)				
   f.close()
   f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_lda_vh.db', 'w') 	
   pickle.dump(V_h[0,:],f)				
   f.close()
   f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_lda_vxc.db', 'w') 
   pickle.dump(V_xc[0,:],f)				
   f.close()
   f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_lda_den.db', 'w') 	
   pickle.dump(n_x[0,:],f)				
   f.close()
   f = open('outputs/' + str(pm.run_name) + '/data/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_lda_E.dat', 'w')      
   f.write(str(E_Total))
   f.close()
   f = open('outputs/' + str(pm.run_name) + '/data/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_lda_Exc.dat', 'w') 
   f.write(str(E_xc_LDA))
   f.close()

   # Time dependence
   if(TD==1):
      for i in range(pm.jmax): 
         V_KS[1,i] = V_KS[0,i] + Potential(i,1)  
         V_ext[i] = V_ext[i] + Potential(i,1) 
      for j in range(1,pm.imax): 
         string = 'LDA: evolving through real time: t = ' + str(j*dt) 
         PS(string)
         n_x, Psi = CrankNicolson(V_KS[j,:], Psi, n_x, j)  
         V_h[j,:] = Hartree(n_x[j,:]) 
         V_xc[j,:] = XC(n_x[j,:])
         if(j != pm.imax-1):
            V_KS[j+1,:] = V_ext[:] + V_h[j,:] + V_xc[j,:] 

      f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_lda_vks.db', 'w') 
      pickle.dump(V_KS.real,f)				
      f.close()
      f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_lda_vh.db', 'w') 	
      pickle.dump(V_h,f)				
      f.close()
      f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_lda_vxc.db', 'w') 	
      pickle.dump(V_xc,f)				
      f.close()
      f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_lda_den.db', 'w') 
      pickle.dump(n_x,f)				
      f.close()

      # Calculate current density
      current_density = calculateCurrentDensity(n_x)
      output_file = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_lda_cur.db','w') 
      pickle.dump(current_density,output_file)
      output_file.close()
      print

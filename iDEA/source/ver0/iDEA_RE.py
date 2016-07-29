######################################################################################
# Name: Reverse Engineering                                                          #
#                                                                                    #
######################################################################################
# Authors: Matt Hodgson, James Ramsden and Matthew Smith                             #
#                                                                                    #
######################################################################################
# Description:                                                                       #
# Computes exact VKS, VH, VXC using the RE algorithm from the exact density          #
#                                                                                    #
######################################################################################
#Notes:                                                                              # #                                                                                    #
# Ground-state calculations are usually fast and stable, this may vary if the system # 
# is particularly difficult to RE, i.e. if the system has regions of very small      #
# electron density. To control the rate of convergence, and the stability of the     #
# GSRE, use the variables 'mu' and 'p'. p is used for the rate of convergence and    #
# forbringing more attention to regions of low density. mu is used for stabilising   #
# the algorithm. Time-dependent RE is much more difficult, and is dependent on the   #
# system. If the TDRE is not converging the most likely reason is that dt is too big.#
# There could also be a problem with noise. Noise should be obvious in he velocity   #
# field (current/density). If noise is dominating the system, try changing the noise #
# filtering value 'alpha'. Alpha controls how much of the high frequency terms are   #
# removed from the KS vector potential.                                              #
######################################################################################

# Import library
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
import scipy.linalg as la
import scipy.sparse as sps
import scipy.linalg as spla
import scipy.sparse.linalg as spsla

# Constants used in the code
sqdx = math.sqrt(pm.deltax)		
upper_bound = int((pm.jmax-1)/2.0)
imax = pm.imax+1
if pm.TD==0:
   imax = 1
# Initialise matrices
T = np.zeros((2,pm.jmax),dtype='complex')
T[0,:] = np.ones(pm.jmax,dtype='complex')/pm.deltax**2		
T[1,:] = -0.5*np.ones(pm.jmax,dtype='float')/pm.deltax**2
J_MB = np.zeros((imax,pm.jmax),dtype='float')
cost_n = np.zeros(imax,dtype='float')	
cost_J = np.zeros(imax,dtype='float')
exp = np.zeros(pm.jmax,dtype='float')
CNRHS = np.zeros(pm.jmax, dtype='complex')
CNLHS = sps.lil_matrix((pm.jmax,pm.jmax),dtype='complex')
Mat = sps.lil_matrix((pm.jmax,pm.jmax),dtype='complex')
Matin = sps.lil_matrix((pm.jmax,pm.jmax),dtype='complex')
V_h = np.zeros((imax,pm.jmax),dtype='float')
V_xc = np.zeros((imax,pm.jmax),dtype='complex')
V_Hxc = np.zeros((imax,pm.jmax),dtype='complex')
A_KS = np.zeros((imax,pm.jmax),dtype='complex')
A_min = np.zeros(pm.jmax,dtype='complex')
U_KS = np.zeros((imax,pm.jmax),dtype='float')
U_MB = np.zeros((imax,pm.jmax),dtype='float')
petrb = np.zeros(pm.jmax,dtype='complex')
U = np.zeros((pm.jmax,pm.jmax))

# Function to read inputs -- needs some work!
def ReadInput(approx,GS,imax):
   n = np.zeros((imax,pm.jmax),dtype='float')
   # Read in the ground-state first
   file_name = 'outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_' + str(approx) + '_den.db'
   input_file = open(file_name,'r')
   data = pickle.load(input_file)
   n[0,:] = data
   if pm.TD==1:
      Read_n = np.zeros(((imax-1),pm.jmax),dtype='float')
      # Then read im the time-dependent density
      file_name = 'outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_' + str(approx) + '_den.db'
      input_file = open(file_name,'r')
      data = pickle.load(input_file)
      Read_n[:,:] = data
      for k in range(1,imax):
         n[k,:] = Read_n[k-1,:] # Accounts for the difference in convention between MB and RE (for RE t=0 is the ground-state)
   return n

# Function to calculate the ground-state potential
def CalculateGroundstate(V,n_T,mu,sqdx,T_s,n):
   #Build Hamiltonian
   p = 0.05 # Determines the rate of convergence of the ground-state RE
   HGS = copy.copy(T_s)
   V[0,:] += mu*(n[0,:]**p-n_T[0,:]**p)
   HGS[0,:] += V[0,:]
   # Solve KS equations
   K,U = spla.eig_banded(HGS,True)
   Psi = np.zeros((pm.NE,2,pm.jmax), dtype='complex')
   for i in range(pm.NE):
      Psi[i,0,:] = U[:,i]/sqdx # Normalise
   # Calculate density and cost function
   n[0,:] = 0
   E_KS = 0.0
   for i in range(pm.NE):
      n[0,:] += abs(Psi[i,0,:])**2 # Calculate the density from the single-particle wavefunctions
      E_KS += K[i]
   cost_n_GS = sum(abs(n_T[0,:]-n[0,:]))*pm.deltax # Calculate the ground-state cost function 
   return V,n,cost_n_GS,Psi,E_KS

# Function to load or force calculation of the ground-state potential
def GroundState(n_T,mu,sqdx,T_s,n,approx):
   V_KS = np.zeros((imax,pm.jmax),dtype='complex')
   V_ext = np.zeros(pm.jmax,dtype='complex')
   print 'REV: calculating ground-state Kohn-Sham potential for the ' + str(approx) + ' density'
   for i in range(pm.jmax):
      V_KS[0,i] = pm.well((i*pm.deltax-pm.xmax)) # Initial guess for KS potential
      V_ext[i] = pm.well((i*pm.deltax-pm.xmax))
   V_KS,n,cost_n_GS,U,E_KS = CalculateGroundstate(V_KS,n_T,0,sqdx,T_s,n)
   print 'REV: initial guess electron density error = %s' % cost_n_GS
   while cost_n_GS>1e-13:
      cost_old = cost_n_GS
      string = 'REV: electron density error = ' + str(cost_old)
      sprint.sprint(string,1,1,pm.msglvl)
      sprint.sprint(string,2,1,pm.msglvl)
      V_KS,n,cost_n_GS,U,E_KS = CalculateGroundstate(V_KS,n_T,mu,sqdx,T_s,n)
      if abs(cost_n_GS-cost_old)<1e-15 or cost_n_GS>cost_old:
         mu *= 0.5
      if mu < 1e-15:
         break
   return V_KS,n,U,V_ext,E_KS

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

# Function to calculate the Hartree potential
def Hartree(density,coulomb,j):                         
   return np.dot(coulomb,density[j,:])*pm.deltax         
                                              
# Function to construct coulomb matrix        
def coulomb():            
   V_coulomb = np.zeros((pm.jmax,pm.jmax))                    
   for i in range(pm.jmax):                        
      xi = i*pm.deltax-pm.xmax                         
      for j in range(pm.jmax):                     
         xj = j*pm.deltax-pm.xmax                      
         V_coulomb[i,j] = 1.0/(abs(xi-xj) + pm.acon)  
   return V_coulomb       

# Function to calculate the exchange-correlation energy
def xcenergy(approx,n,V_h,V_xc,E_KS):
   try:
      file_name = 'outputs/' + str(pm.run_name) + '/data/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_' + str(approx) + '_E.dat'
      E_MB = np.loadtxt(file_name)
      E_xc = E_MB - E_KS
      for i in range(pm.jmax):
         E_xc += (n[0,i])*((0.50*V_h[0,i])+(V_xc[0,i]))*pm.deltax
   except:
      E_xc = 0.0
   return E_xc

# Function to extrapolate the current density from regions of low density to the system's edges
def ExtrapolateCD(J,j,n,n_T,upper_bound):
   imaxl = 0 # Start from the edge of the system
   nmaxl = 0.0
   imaxr = 0									
   nmaxr = 0.0
   for l in range(upper_bound+1):
      if n_T[j,l]>nmaxl: # Find the first peak in the density from the left
         nmaxl = n_T[j,l]
	 imaxl = l
      i = upper_bound+l-1
      if n_T[j,i]>nmaxr: # Find the first peak in the density from the right
          nmaxr = n_T[j,i]
          imaxr = l
   U = np.zeros(pm.jmax)
   U[:] = J[:]/n[j,:]
   dUdx = np.zeros(pm.jmax)
   # Extraplorate the density for the low density regions
   for i in range(imaxl+1):
      l = imaxl-i
      if n_T[j,l]<1e-8:
         dUdx[:] = np.gradient(U[:],pm.deltax)
         U[l] = 8*U[l+1]-8*U[l+3]+U[l+4]+dUdx[l+2]*12.0*pm.deltax
   for i in range(int(0.5*(pm.jmax-1)-imaxr+1)):
      l = int(0.5*(pm.jmax-1)+imaxr+i)
      if n_T[j,l]<1e-8:
         dUdx[:] = np.gradient(U[:],pm.deltax)
         U[l] = 8*U[l-1]-8*U[l-3]+U[l-4]-dUdx[l-2]*12.0*pm.deltax
   J[:] = n[j,:]*U[:]							
   return J

# Function to extrapolate the KS vector potential from regions of low density to the system's edges
def ExtrapolateVectorPotential(A,n_T,j,upper_bound):
   imaxl = 0
   nmaxl = 0.0
   imaxr = 0									
   nmaxr = 0.0
   for i in range(upper_bound+1):
      if n_T[j,i]>nmaxl:
         nmaxl = n_T[j,i]
         imaxl = i
   for l in range(upper_bound+1):
      i = upper_bound +l
      if n_T[j,i]>nmaxr:
         nmaxr = n_T[j,i]
         imaxr = l
   dAdx = np.zeros(pm.jmax,dtype='complex')
   # Extraplorate the Hxc vector potential for the low density regions
   for i in range(imaxl+1):
      l = imaxl-i
      if n_T[j,l]<1e-8:
         dAdx[:] = np.gradient(A[j,:],pm.deltax)
         A[j,l] = 8*A[j,l+1]-8*A[j,l+3]+A[j,l+4]+dAdx[l+2]*12.0*pm.deltax 
   for i in range(upper_bound +1-imaxr):
      l = (upper_bound+imaxr+i)
      if n_T[j,l]<1e-8:
         dAdx[:] = np.gradient(A[j,:],pm.deltax)
         A[j,l] = 8*A[j,l-1]-8*A[j,l-3]+A[j,l-4]-dAdx[l-2]*12.0*pm.deltax
   return A

# Function to filter out 'noise' occuring between calculation of the exact TDSE solution and the present KS solution
def Filter(A,j,exp):
   A_Kspace = np.zeros(pm.jmax,dtype='complex')
   A_Kspace = momentumspace(A[j,:])
   A_Kspace[:] *= exp[:]
   A[j,:] = realspace(A_Kspace).real
   return A

# Function to solve TDKSEs using the Crank-Nicolson method
def SolveKSE(V,A,Wavefunction,j,frac1,frac2,z):
   Mat = sps.lil_matrix((pm.jmax,pm.jmax),dtype='complex')
   for i in range(pm.jmax):
      Mat[i,i] = 1.0+0.5j*pm.deltat*(1.0/pm.deltax**2+0.5*A[j,i]**2+V[j,i])
   for i in range(pm.jmax-1):
      Mat[i,i+1] = -0.5j*pm.deltat*(0.5/pm.deltax-(frac1)*1.0j*A[j,i+1]-(frac1)*1.0j*A[j,i])/pm.deltax
   for i in range(1,pm.jmax):
      Mat[i,i-1] = -0.5j*pm.deltat*(0.5/pm.deltax+(frac1)*1.0j*A[j,i-1]+(frac1)*1.0j*A[j,i])/pm.deltax
   for i in range(pm.jmax-2):	
      Mat[i,i+2] = -0.5j*pm.deltat*(1.0j*A[j,i+2]+1.0j*A[j,i])*(frac2)/pm.deltax
   for i in range(2,pm.jmax):
      Mat[i,i-2] = 0.5j*pm.deltat*(1.0j*A[j,i-2]+1.0j*A[j,i])*(frac2)/pm.deltax
   # Solve the TDKS equations 
   Mat = Mat.tocsr()
   Matin =- (Mat-sps.identity(pm.jmax,dtype='complex'))+sps.identity(pm.jmax,dtype='complex')
   for i in range(pm.NE):
      B = Matin*Wavefunction[i,z,:]
      z = z*(-1)+1 # Only save two times at any point
      Wavefunction[i,z,:]=spsla.spsolve(Mat,B)
      z = z*(-1)+1	
   return Wavefunction,z

# Function to calculate the current density
def CalculateCurrentDensity(n,n_MB,upper_bound,j):
   J = RE_Utilities.continuity_eqn(pm.jmax,pm.deltax,pm.deltat,n[j,:],n[j-1,:])
   if pm.im == 1:
      for j in xrange(pm.jmax):
         for k in xrange(j+1):
            x = k*pm.deltax-pm.xmax
            J[j] -= abs(pm.im_petrb(x))*n[j,k]*pm.deltax
   else:
      J = ExtrapolateCD(J,j,n,n_MB,upper_bound)
   return J

# Function to calculate the KS vector (and finally scalar) potential
def CalculateKS(V_KS,A_KS,J,Psi,j,upper_bound,frac1,frac2,z,tol,n_T,J_T,cost_n,cost_J,A_min,n_KS,exp):
   # Set initial trial vector potential as previous time-step's vector potential
   Apot = np.zeros(pm.jmax,dtype='complex')
   A_KS[j,:] = A_KS[j-1,:] 
   Psi,z = SolveKSE(V_KS,A_KS,Psi,j,frac1,frac2,z)
   # Calculate KS charge density
   n_KS[j,:] = 0
   z = z*(-1)+1 # Only save two times at any point
   for i in range(pm.NE):
      n_KS[j,:] += abs(Psi[i,z,:])**2
   z = z*(-1)+1
   J[j,:] = CalculateCurrentDensity(n_KS,n_T,upper_bound,j) # Calculate KS current density
   J_T[j,:] = CalculateCurrentDensity(n_T,n_T,upper_bound,j) # Calculate MB current density
   # Evaluate cost functions corresponding to present vector potential
   cost_J[j] = sum(abs(J[j,:]-J_T[j,:]))*pm.deltax 
   cost_n[j] = sum(abs(n_KS[j,:]-n_T[j,:]))*pm.deltax
   # Set initial trial vector potential as reference vector potential
   A_min[:] = A_KS[j,:] 
   cost_min = 2.0
   count = 1
   count_max = 10 # If in accurate make larger so the algorithm can converge more
   mix = 1.0 # Mixing parameter for RE
   while (count<=count_max): # Exit condition: KS and exact current density are equal at all points
      cost_old = cost_J[j] # Keep track of convergence
      if count%10==0:
         mix *= 0.5
         if tol<1e-3: # Minimum accuracy limit
            tol*=10 # Increase allowed convergence tolerance for J_check
            A_KS[j,:] = A_KS[j-1,:] # Reset vector potential
            Psi,z = SolveKSE(V_KS,A_KS,Psi,j,frac1,frac2,z) # Solve Schrodinger equation for KS system using initial trial potential
            n_KS[j,:] = 0
            z = z*(-1)+1 # Only save two times at any point
            for i in range(pm.NE):
                n_KS[j,:] += abs(Psi[i,z,:])**2
            z = z*(-1)+1
            J[j,:] = CalculateCurrentDensity(n_KS,n_T,upper_bound,j)
            cost_J[j] = sum(abs(J[j,:]-J_T[j,:]))*pm.deltax
            cost_n[j] = sum(abs(n_KS[j,:]-n_T[j,:]))*pm.deltax
         else:
            mix = 1.0
            A_KS[j,:] = A_min[:]
            break
      A_KS[j,:] += mix*(J[j,:]-J_T[j,:])/n_T[j,:] # Update vector potential
      A_KS = ExtrapolateVectorPotential(A_KS,n_T,j,upper_bound) # Extrapolate vector potential from low density regions to edges of system
      A_KS = Filter(A_KS,j,exp) # Remove high frequencies from vector potential
      Psi,z = SolveKSE(V_KS,A_KS,Psi,j,frac1,frac2,z) # Solve KS equations using updated vector potential
      n_KS[j,:] = 0
      z = z*(-1)+1 # Only save two times at any point
      for i in range(pm.NE):
         n_KS[j,:] += abs(Psi[i,z,:])**2
      z = z*(-1)+1
      J[j,:] = CalculateCurrentDensity(n_KS,n_T,upper_bound,j) # Calculate updated KS current density
      cost_J[j] = sum(abs(J[j,:]-J_T[j,:]))*pm.deltax 
      cost_n[j] = sum(abs(n_KS[j,:]-n_T[j,:]))*pm.deltax   
      if cost_J[j]<cost_min:  # Keep present vector potential for reference if produces lower cost function evaluation
         cost_min = cost_J[j]
         A_min[:] = A_KS[j,:]
      J_check = RE_Utilities.compare(pm.jmax,J[j,:],J_T[j,:],tol) # Check if KS and exact current density are equal
      if J_check:
         A_KS[j,:] = A_min[:] # Go with the best answer
         z=z*(-1)+1 # Only save two times at any point
         break
      count += 1
      if count>=count_max:
          A_KS[j,:]=A_min[:] # Go with the best answer
          z = z*(-1)+1 # Only save two times at any point
          break
   string='REV: t = ' + str(j*pm.deltat) + ', tol = ' + str(tol) + ', current error = ' + str(cost_J[j]) + ', density error = ' + str(cost_n[j])
   sprint.sprint(string,1,1,pm.msglvl)
   Apot[:]=0 # Change guage so only have scalar potential
   for i in range(pm.jmax): # Calculate full KS scalar potential
      for k in range(i+1):
         Apot[i] += ((A_KS[j,k]-A_KS[j-1,k])/pm.deltat)*pm.deltax
   V_KS[j,:] += Apot[:]
   V_KS[j,:] += V_KS[0,(pm.jmax-1)*0.5]-V_KS[j,(pm.jmax-1)*0.5]
   return n_KS,V_KS,J,Apot,z

# Main control function
def main(approx):
   z = 0
   mu = 1.0 # Mixing for the ground-state KS algorithm
   alpha = 1 # Strength of noise control
   frac1 = 1.0/3.0
   frac2 = 1.0/24.0
   n_KS = np.zeros((imax,pm.jmax),dtype='float')
   n_MB = ReadInput(approx,0,imax) # Read in exact charge density obtained from code
   V_coulomb = coulomb()
   V_KS,n_KS,Psi,V_ext,E_KS = GroundState(n_MB,mu,sqdx,T,n_KS,approx) # Calculate (or, if already obtained, check) ground-state KS potential
   V_h[0,:] = Hartree(n_KS,V_coulomb,0) # Calculate the Hartree potential
   V_Hxc[0,:] = V_KS[0,:]-V_ext[:] # Calculate the Hartree exhange-correlation potential
   V_xc[0,:] = V_Hxc[0,:]-V_h[0,:] # Calculate the exchange-correlation potential
   E_xc = xcenergy(approx,n_KS,V_h,V_xc,E_KS) # Calculate the exchange-correlation energy
   f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_' + str(approx) + '_vks.db', 'w') # KS potential	
   pickle.dump(V_KS[0,:].real,f)				
   f.close()
   f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_' + str(approx) + '_vh.db', 'w') # H potential
   pickle.dump(V_h[0,:],f)
   f.close()
   f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_' + str(approx) + '_vxc.db', 'w') # XC potential
   pickle.dump(V_xc[0,:].real,f)
   f.close()
   f = open('outputs/' + str(pm.run_name) + '/data/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_' + str(approx) + '_Exc.dat', 'w') # XC energy
   f.write(str(E_xc.real))
   f.close()
   v_hxc = np.zeros(pm.jmax,dtype='float')
   v_hxc[:] = (V_xc[0,:]+V_h[0,:]).real
   f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_' + str(approx) + '_hxc.db', 'w') # KS potential	
   pickle.dump(v_hxc[:],f)				
   f.close()
   if(approx != 'non')
      print
   if pm.TD==1: # Time-dependence
      J_KS = np.zeros((imax,pm.jmax),dtype='float')
      n_MB = ReadInput(approx,1,imax) # Read in exact charge density obtained from code
      for i in range(pm.jmax):
         petrb[i] = pm.petrb((i*pm.deltax-pm.xmax))
         exp[i] = math.exp(-alpha*(i*pm.deltax-pm.xmax)**2)
      V_KS[:,:] = V_KS[0,:]+petrb[:] # Add the perturbing field to the external potential and the KS potential
      V_KS[0,:] -= petrb[:]
      tol = 1e-13 # Set inital convergence tolerance for exact and KS current densities
      try:
         counter = 0
         for j in range(1,imax): # Propagate from the ground-state
            n_KS,V_KS,J_KS,Apot,z = CalculateKS(V_KS,A_KS,J_KS,Psi,j,upper_bound,frac1,frac2,z,tol,n_MB,J_MB,cost_n,cost_J,A_min,n_KS,exp) # Calculate KS potential
            V_h[j,:] = Hartree(n_KS,j)
            V_Hxc[j,:] = V_KS[j,:]-(V_ext[:]+petrb[:])
            V_xc[j,:] = V_KS[j,:]-(V_ext[:]+petrb[:]+V_h[j,:])
            counter += 1
      except:
         print
         print 'REV: Stopped at timestep ' + str(counter) + '! Outputing all quantities'
      file_name=open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_' + str(approx) + '_vks.db', 'w') # KS potential
      pickle.dump(V_KS[:,:].real,file_name)
      file_name.close()
      file_name=open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_' + str(approx) + '_vh.db', 'w') # H potential
      pickle.dump(V_h[:,:],file_name)		
      file_name.close()
      file_name=open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_' + str(approx) + '_vxc.db', 'w') # xc potential
      pickle.dump(V_xc[:,:].real,file_name)	
      file_name.close()

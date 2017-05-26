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

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# Import library
import copy
import math
import pickle
import numpy as np
import scipy.sparse as sps
import scipy.linalg as spla
import scipy.sparse.linalg as spsla
from scipy.optimize import curve_fit as spocf
from . import RE_Utilities
from . import results as rs

# Function to read inputs -- needs some work!
def ReadInput(approx,GS,imax):
   n = np.zeros((imax,pm.sys.grid),dtype='float')
   # Read in the ground-state first

   name = 'gs_{}_den'.format(approx)
   data = rs.Results.read(name, pm)
   
   n[0,:] = data
   if pm.run.time_dependence == True:
      Read_n = np.zeros(((imax-1),pm.sys.grid),dtype='float')
      # Then read im the time-dependent density
      name = 'td_{}_den'.format(approx)
      data = rs.Results.read(name, pm)
      Read_n[:,:] = data
      for k in range(1,imax):
         n[k,:] = Read_n[k-1,:] # Accounts for the difference in convention between MB and RE (for RE t=0 is the ground-state)
   return n

# Function to calculate the ground-state potential
def CalculateGroundState(V,n_T,mu,sqdx,T_s,n):
   #Build Hamiltonian
   p = 0.05 # Determines the rate of convergence of the ground-state RE
   HGS = copy.copy(T_s)
   V[0,:] += mu*(n[0,:]**p-n_T[0,:]**p)
   HGS[0,:] += V[0,:]
   # Solve KS equations
   K,U = spla.eig_banded(HGS,True)
   U /= sqdx # Normalise
   Psi = np.zeros((pm.sys.NE,2,pm.sys.grid), dtype='complex')
   for i in range(pm.sys.NE):
      Psi[i,0,:] = U[:,i]
   # Calculate density and cost function
   n[0,:] = 0
   E_KS = 0.0
   for i in range(pm.sys.NE):
      n[0,:] += abs(Psi[i,0,:])**2 # Calculate the density from the single-particle wavefunctions
      E_KS += K[i]
   cost_n_GS = sum(abs(n_T[0,:]-n[0,:]))*pm.sys.deltax # Calculate the ground-state cost function 
   return V,n,cost_n_GS,Psi,E_KS,K,U

# Function to load or force calculation of the ground-state potential
def GroundState(n_T,mu,sqdx,T_s,n,approx):
        V_KS = np.zeros((imax,pm.sys.grid),dtype='complex')
        V_ext = np.zeros(pm.sys.grid,dtype='complex')
        pm.sprint('REV: calculating ground-state Kohn-Sham potential for the {} density'.format(approx),1)
        try:
                V_KS[0,:] = rs.Results.read('gs_extre_vks', pm)
                pm.sprint('REV: Found exact kohn-sham potential to start from...',1)
        except:
                pm.sprint('REV: Starting from external potential...',1)
                for i in range(pm.sys.grid):
                        V_KS[0,i] = pm.sys.v_ext((i*pm.sys.deltax-pm.sys.xmax)) # Initial guess for KS potential
        for i in range(pm.sys.grid):
                V_ext[i] = pm.sys.v_ext((i*pm.sys.deltax-pm.sys.xmax))
        V_KS,n,cost_n_GS,Psi,E_KS,K,U = CalculateGroundState(V_KS,n_T,0,sqdx,T_s,n)
        pm.sprint('REV: initial guess electron density error = %s' % cost_n_GS,1)
        while cost_n_GS>1e-13:
                cost_old = cost_n_GS
                string = 'REV: electron density error = {}'.format(cost_old)
                pm.sprint(string,1,newline=False)
                V_KS,n,cost_n_GS,Psi,E_KS,K,U = CalculateGroundState(V_KS,n_T,mu,sqdx,T_s,n)
                if abs(cost_n_GS-cost_old)<1e-15 or cost_n_GS>cost_old:
                        mu *= 0.5
                if mu < 1e-15:
                        break
        pm.sprint('',1)
        return V_KS,n,Psi,V_ext,E_KS,K,U

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

# Function to calculate the Hartree potential
def Hartree(density,coulomb,j):                         
   return np.dot(coulomb,density[j,:])*pm.sys.deltax         
                                              
# Function to construct coulomb matrix        
def coulomb():            
   V_coulomb = np.zeros((pm.sys.grid,pm.sys.grid))                    
   for i in range(pm.sys.grid):                        
      xi = i*pm.sys.deltax-pm.sys.xmax                         
      for j in range(pm.sys.grid):                     
         xj = j*pm.sys.deltax-pm.sys.xmax                      
         V_coulomb[i,j] = 1.0/(abs(xi-xj) + pm.sys.acon)
   return V_coulomb       

# Function to calculate the exchange-correlation energy
def xcenergy(approx,n,V_h,V_xc,E_KS):
   try:
      E_MB = rs.Results.read('gs_{}_E'.format(approx), pm)
      E_xc = E_MB - E_KS
      for i in range(pm.sys.grid):
         E_xc += (n[0,i])*((0.50*V_h[0,i])+(V_xc[0,i]))*pm.sys.deltax
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
   U = np.zeros(pm.sys.grid)
   U[:] = J[:]/n[j,:]
   dUdx = np.zeros(pm.sys.grid)
   # Extraplorate the density for the low density regions
   for i in range(imaxl+1):
      l = imaxl-i
      if n_T[j,l]<1e-8:
         dUdx[:] = np.gradient(U[:],pm.sys.deltax)
         U[l] = 8*U[l+1]-8*U[l+3]+U[l+4]+dUdx[l+2]*12.0*pm.sys.deltax
   for i in range(int(0.5*(pm.sys.grid-1)-imaxr+1)):
      l = int(0.5*(pm.sys.grid-1)+imaxr+i)
      if n_T[j,l]<1e-8:
         dUdx[:] = np.gradient(U[:],pm.sys.deltax)
         U[l] = 8*U[l-1]-8*U[l-3]+U[l-4]-dUdx[l-2]*12.0*pm.sys.deltax
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
   dAdx = np.zeros(pm.sys.grid,dtype='complex')
   # Extraplorate the Hxc vector potential for the low density regions
   for i in range(imaxl+1):
      l = imaxl-i
      if n_T[j,l]<1e-8:
         dAdx[:] = np.gradient(A[j,:],pm.sys.deltax)
         A[j,l] = 8*A[j,l+1]-8*A[j,l+3]+A[j,l+4]+dAdx[l+2]*12.0*pm.sys.deltax 
   for i in range(upper_bound +1-imaxr):
      l = (upper_bound+imaxr+i)
      if n_T[j,l]<1e-8:
         dAdx[:] = np.gradient(A[j,:],pm.sys.deltax)
         A[j,l] = 8*A[j,l-1]-8*A[j,l-3]+A[j,l-4]-dAdx[l-2]*12.0*pm.sys.deltax
   return A

# Function to filter out 'noise' occuring between calculation of the exact TDSE solution and the present KS solution
def Filter(A,j,exp):
   A_Kspace = np.zeros(pm.sys.grid,dtype='complex')
   A_Kspace = momentumspace(A[j,:])
   A_Kspace[:] *= exp[:]
   A[j,:] = realspace(A_Kspace).real
   return A

# Function to solve TDKSEs using the Crank-Nicolson method
def SolveKSE(V,A,Wavefunction,j,frac1,frac2,z):
   Mat = sps.lil_matrix((pm.sys.grid,pm.sys.grid),dtype='complex')
   for i in range(pm.sys.grid):
      Mat[i,i] = 1.0+0.5j*pm.sys.deltat*(1.0/pm.sys.deltax**2+0.5*A[j,i]**2+V[j,i])
   for i in range(pm.sys.grid-1):
      Mat[i,i+1] = -0.5j*pm.sys.deltat*(0.5/pm.sys.deltax-(frac1)*1.0j*A[j,i+1]-(frac1)*1.0j*A[j,i])/pm.sys.deltax
   for i in range(1,pm.sys.grid):
      Mat[i,i-1] = -0.5j*pm.sys.deltat*(0.5/pm.sys.deltax+(frac1)*1.0j*A[j,i-1]+(frac1)*1.0j*A[j,i])/pm.sys.deltax
   for i in range(pm.sys.grid-2):
      Mat[i,i+2] = -0.5j*pm.sys.deltat*(1.0j*A[j,i+2]+1.0j*A[j,i])*(frac2)/pm.sys.deltax
   for i in range(2,pm.sys.grid):
      Mat[i,i-2] = 0.5j*pm.sys.deltat*(1.0j*A[j,i-2]+1.0j*A[j,i])*(frac2)/pm.sys.deltax
   # Solve the TDKS equations 
   Mat = Mat.tocsr()
   Matin =- (Mat-sps.identity(pm.sys.grid,dtype='complex'))+sps.identity(pm.sys.grid,dtype='complex')
   for i in range(pm.sys.NE):
      B = Matin*Wavefunction[i,z,:]
      z = z*(-1)+1 # Only save two times at any point
      Wavefunction[i,z,:]=spsla.spsolve(Mat,B)
      z = z*(-1)+1
   return Wavefunction,z

# Function to calculate the current density
def CalculateCurrentDensity(n,n_MB,upper_bound,j):
   J = np.zeros(pm.sys.grid, dtype=np.float, order='F')
   J = RE_Utilities.continuity_eqn(J, n[j,:], n[j-1,:], pm.sys.deltax, pm.sys.deltat, pm.sys.grid)
   if(pm.sys.im == 0):
      J = ExtrapolateCD(J,j,n,n_MB,upper_bound)
   return J

# Function to calculate the KS vector (and finally scalar) potential
def CalculateKS(V_KS,A_KS,J,Psi,j,upper_bound,frac1,frac2,z,tol,n_T,J_T,cost_n,cost_J,A_min,n_KS,exp):
   # Set initial trial vector potential as previous time-step's vector potential
   Apot = np.zeros(pm.sys.grid,dtype='complex')
   A_KS[j,:] = A_KS[j-1,:] 
   Psi,z = SolveKSE(V_KS,A_KS,Psi,j,frac1,frac2,z)
   # Calculate KS charge density
   n_KS[j,:] = 0
   z = z*(-1)+1 # Only save two times at any point
   for i in range(pm.sys.NE):
      n_KS[j,:] += abs(Psi[i,z,:])**2
   z = z*(-1)+1
   J[j,:] = CalculateCurrentDensity(n_KS,n_T,upper_bound,j) # Calculate KS current density
   J_T[j,:] = CalculateCurrentDensity(n_T,n_T,upper_bound,j) # Calculate MB current density
   # Evaluate cost functions corresponding to present vector potential
   cost_J[j] = sum(abs(J[j,:]-J_T[j,:]))*pm.sys.deltax 
   cost_n[j] = sum(abs(n_KS[j,:]-n_T[j,:]))*pm.sys.deltax
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
            for i in range(pm.sys.NE):
                n_KS[j,:] += abs(Psi[i,z,:])**2
            z = z*(-1)+1
            J[j,:] = CalculateCurrentDensity(n_KS,n_T,upper_bound,j)
            cost_J[j] = sum(abs(J[j,:]-J_T[j,:]))*pm.sys.deltax
            cost_n[j] = sum(abs(n_KS[j,:]-n_T[j,:]))*pm.sys.deltax
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
      for i in range(pm.sys.NE):
         n_KS[j,:] += abs(Psi[i,z,:])**2
      z = z*(-1)+1
      J[j,:] = CalculateCurrentDensity(n_KS,n_T,upper_bound,j) # Calculate updated KS current density
      cost_J[j] = sum(abs(J[j,:]-J_T[j,:]))*pm.sys.deltax 
      cost_n[j] = sum(abs(n_KS[j,:]-n_T[j,:]))*pm.sys.deltax   
      if cost_J[j]<cost_min:  # Keep present vector potential for reference if produces lower cost function evaluation
         cost_min = cost_J[j]
         A_min[:] = A_KS[j,:]
      J_check = RE_Utilities.compare(J[j,:],J_T[j,:],tol,pm.sys.grid) # Check if KS and exact current density are equal
      if J_check:
         A_KS[j,:] = A_min[:] # Go with the best answer
         z=z*(-1)+1 # Only save two times at any point
         break
      count += 1
      if count>=count_max:
          A_KS[j,:]=A_min[:] # Go with the best answer
          z = z*(-1)+1 # Only save two times at any point
          break
   string='REV: t = {}, tol = {}, current error = {}, density error = {}'\
           .format(j*pm.sys.deltat, tol, cost_J[j], cost_n[j])
   pm.sprint(string,1,newline=False)
   Apot[:]=0 # Change guage so only have scalar potential
   for i in range(pm.sys.grid): # Calculate full KS scalar potential
      for k in range(i+1):
         Apot[i] += ((A_KS[j,k]-A_KS[j-1,k])/pm.sys.deltat)*pm.sys.deltax
   V_KS[j,:] += Apot[:]
   V_KS[j,:] += V_KS[0,int((pm.sys.grid-1)*0.5)]-V_KS[j,int((pm.sys.grid-1)*0.5)]
   return n_KS,V_KS,J,Apot,z

#Function to be fit for asymptotic form of V_xc
def xcfit(x, a):
   return 1/x + a

#Function to determine correction to KS potential
#Start and end is the percentage of the data from the left you want to fit
def correction(vxc, start, finish):
    xpoints = np.linspace(-pm.sys.xmax, pm.sys.xmax, pm.sys.grid)
    s = int(start*pm.sys.grid)
    f = int(finish*pm.sys.grid)
    fit = spocf(xcfit, xpoints[s:f], vxc[s:f])
    a = fit[0][0]
    aerr = np.sqrt(np.diag(fit[1]))
    #a is correction, aerr is error in fit parameter calculated by curve_fit
    return a, aerr

# Main control function
def main(parameters,approx):
   global sqdx, upper_bound, imax, T, J_MB, cost_n, cost_J, exp, CNRHS, CNLHS
   global Mat, Matin, V_h, V_xc, V_Hxc, A_KS, A_min, U_KS, U_MB, petrb
   global pm

   pm = parameters
   pm.setup_space()

   # Constants used in the code
   sqdx = math.sqrt(pm.sys.deltax)
   upper_bound = int((pm.sys.grid-1)/2.0)
   imax = pm.sys.imax+1
   if pm.run.time_dependence == False:
      imax = 1
   # Initialise matrices
   sd = pm.space.second_derivative_band
   nbnd = len(sd)
   T = np.zeros((nbnd,pm.sys.grid),dtype='complex')
   for i in range(nbnd):
      T[i,:] = -0.5 * sd[i]

   J_MB = np.zeros((imax,pm.sys.grid),dtype='float')
   cost_n = np.zeros(imax,dtype='float')
   cost_J = np.zeros(imax,dtype='float')
   exp = np.zeros(pm.sys.grid,dtype='float')
   CNRHS = np.zeros(pm.sys.grid, dtype='complex')
   CNLHS = sps.lil_matrix((pm.sys.grid,pm.sys.grid),dtype='complex')
   Mat = sps.lil_matrix((pm.sys.grid,pm.sys.grid),dtype='complex')
   Matin = sps.lil_matrix((pm.sys.grid,pm.sys.grid),dtype='complex')
   V_h = np.zeros((imax,pm.sys.grid),dtype='float')
   V_xc = np.zeros((imax,pm.sys.grid),dtype='complex')
   V_Hxc = np.zeros((imax,pm.sys.grid),dtype='complex')
   A_KS = np.zeros((imax,pm.sys.grid),dtype='complex')
   A_min = np.zeros(pm.sys.grid,dtype='complex')
   U_KS = np.zeros((imax,pm.sys.grid),dtype='float')
   U_MB = np.zeros((imax,pm.sys.grid),dtype='float')
   petrb = np.zeros(pm.sys.grid,dtype='complex')


   z = 0
   mu = 1.0 # Mixing for the ground-state KS algorithm
   alpha = 1 # Strength of noise control
   frac1 = 1.0/3.0
   frac2 = 1.0/24.0
   n_KS = np.zeros((imax,pm.sys.grid),dtype='float')
   n_MB = ReadInput(approx,0,imax) # Read in exact charge density obtained from code
   V_coulomb = coulomb()
   V_KS,n_KS,Psi,V_ext,E_KS,eigv,eigf = GroundState(n_MB,mu,sqdx,T,n_KS,approx) # Calculate (or, if already obtained, check) ground-state KS potential
   V_h[0,:] = Hartree(n_KS,V_coulomb,0) # Calculate the Hartree potential
   V_Hxc[0,:] = V_KS[0,:]-V_ext[:] # Calculate the Hartree exhange-correlation potential
   V_xc[0,:] = V_Hxc[0,:]-V_h[0,:] # Calculate the exchange-correlation potential

   #Correct V_xc and V_ks etc.
   #For some reason eigenvalues, eigv, and sum of eigenvalues, E_KS, are returned separately
   #GroundState() could just return the array of eigenvalues
   #This hasn't been changed yet to avoid breaking anything before RE.py is tidied up
   correct, correct_error = correction(np.real(V_xc[0,:]), 0.05, 0.15)
   print('Approximate error in correction to asymptotic form of V_xc = ', correct_error)

   V_KS[0,:] - V_KS[0,:]
   V_Hxc[0,:] = V_KS[0,:]-V_ext[:] # Calculate the Hartree exhange-correlation potential
   V_xc[0,:] = V_Hxc[0,:]-V_h[0,:] - correct # Calculate the exchange-correlation potential
   E_KS = E_KS - pm.sys.NE*correct # Correct sum of eigenvalues
   eigv[:] = eigv[:] - correct # Correct the actual eigenvalues
   E_xc = xcenergy(approx,n_KS,V_h,V_xc,E_KS) # Calculate the exchange-correlation energy

   # Store results
   approxre = approx + 're'
   results = rs.Results()
   results.add(n_MB[0,:],'gs_{}_den'.format(approxre))
   results.add(V_KS[0,:].real,'gs_{}_vks'.format(approxre))
   results.add(V_h[0,:], 'gs_{}_vh'.format(approxre))
   results.add(V_xc[0,:].real, 'gs_{}_vxc'.format(approxre))
   results.add(E_xc.real, 'gs_{}_Exc'.format(approxre))
   results.add(E_KS.real, 'gs_{}_Eks'.format(approxre))
   v_hxc = np.zeros(pm.sys.grid,dtype='float')
   v_hxc[:] = (V_xc[0,:]+V_h[0,:]).real
   results.add(v_hxc[:],'gs_{}_hxc'.format(approxre))

   if pm.re.save_eig:
       results.add(eigf.T,'gs_{}_eigf'.format(approxre))
       results.add(eigv,'gs_{}_eigv'.format(approxre))

   if pm.run.save:
      results.save(pm)

   #if(approx != 'non'):
   pm.sprint('',1)
   if pm.run.time_dependence==True: # Time-dependence
      J_KS = np.zeros((imax,pm.sys.grid),dtype='float')
      n_MB = ReadInput(approx,1,imax) # Read in exact charge density obtained from code
      for i in range(pm.sys.grid):
         petrb[i] = pm.sys.v_pert((i*pm.sys.deltax-pm.sys.xmax))
         exp[i] = math.exp(-alpha*(i*pm.sys.deltax-pm.sys.xmax)**2)
      V_KS[:,:] = V_KS[0,:]+petrb[:] # Add the perturbing field to the external potential and the KS potential
      V_KS[0,:] -= petrb[:]
      tol = 1e-13 # Set inital convergence tolerance for exact and KS current densities
      try:
         counter = 0
         for j in range(1,imax): # Propagate from the ground-state
            n_KS,V_KS,J_KS,Apot,z = CalculateKS(V_KS,A_KS,J_KS,Psi,j,upper_bound,frac1,frac2,z,tol,n_MB,J_MB,cost_n,cost_J,A_min,n_KS,exp) # Calculate KS potential
            V_h[j,:] = Hartree(n_KS,V_coulomb,j)
            V_Hxc[j,:] = V_KS[j,:]-(V_ext[:]+petrb[:])
            V_xc[j,:] = V_KS[j,:]-(V_ext[:]+petrb[:]+V_h[j,:])
            counter += 1
      except:
         pm.sprint('',1)
         pm.sprint('REV: Stopped at timestep {}! Outputing all quantities'.format(counter),1)

      results.add(n_MB[:,:],'td_{}_den'.format(approxre))
      results.add( V_KS[:,:].real, 'td_{}_vks'.format(approxre)) 
      results.add(V_h[:,:], 'td_{}_vh'.format(approxre)) 
      results.add( V_xc[:,:].real, 'td_{}_vxc'.format(approxre))
      if pm.run.save:
         # No need to save previous results again
         l = ['td_{}_vks'.format(approxre),'td_{}_vh'.format(approxre),'td_{}_vxc'.format(approxre)]
         results.save(pm, list=l)

   return results

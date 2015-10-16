######################################################################################
# Name: 1 electron LDA                                                               #
######################################################################################
# Author(s): Mike Entwistle                                                          #
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

# Import libraries
from math import *									
from numpy import *
from scipy.linalg import eig_banded, solve
import parameters as pm
import sys
import pickle 
from scipy import sparse
from scipy import special
from scipy.sparse import linalg as spla

# Parameters
jmax = pm.jmax
imax = pm.imax
xmax = pm.xmax 
tmax = pm.tmax
L = 2.0*xmax
dx = pm.deltax
sqdx = sqrt(dx)
dt = pm.deltat
TD = pm.TD
NE = pm.LDA_NE
Mix = 0.1
tol = 1e-12
Cost = 1 
Run = 1
E_xc_Exact = 0 

# Matrices
Psi0 = zeros((imax,jmax), dtype='complex') # Wave function for particle						
V_h = zeros((imax,jmax)) # Potentials
V_xc = zeros((imax,jmax)) 
V_hxc = zeros((imax,jmax)) 
n_x = zeros((imax,jmax), dtype ='float') # Charge Density
n_x_old = zeros((imax,jmax), dtype='float') 
J_x = zeros((imax,jmax)) # Current Density 
T = zeros((2,jmax), dtype='complex') # Kinetic Energy operator
T[0,:] = ones(jmax)/dx**2 								
T[1,:] = -0.5*ones(jmax)/dx**2 
V_KS = zeros((imax,jmax)) # Kohn-Sham potential
V_KS_old = zeros((imax,jmax)) 
V_ext = zeros(jmax) # External potential
CNLHS = sparse.lil_matrix((jmax,jmax),dtype='complex') # Matrix for the left hand side of the Crank Nicholson method
Mat = sparse.lil_matrix((jmax,jmax),dtype='complex')   
Matin = sparse.lil_matrix((jmax,jmax),dtype='complex') # Inverted Matrix for the right hand side of the Crank Nicholson method 
																		
# Potential Generator 
def Potential(i,j):
        x = -xmax + i*dx 
        if (j==0): 
            V = pm.well(x)
        else: 
            V = pm.petrb(x)
        return V

# Solve the time-independent Schrodinger equation 	
def TISE(V_KS,j): 				                         											
        HGS = copy(T) # Reset Hamiltonian									
        HGS[0,:] += V_KS[:] # Add The Kohn-Sham potential to the Hamiltonian																
        K, U = eig_banded(HGS, True) # Returns eigenvalues (K) and eigenvectors (U)					 									
        Psi0[j,:] = U[:,0]/sqdx # Normalise the wave functions 							
        n_x[j,:] = abs(Psi0[j,:])**2 # Calculate charge density				   
        return n_x[j,:], Psi0[j,:]

# Define function for Fourier transforming into real-space	
def realspace(vector): 											
	mid_k = int(0.5*(jmax-1))
	fftin = zeros(jmax-1, dtype='complex')
	fftin[0:mid_k+1] = vector[mid_k:jmax]
	fftin[jmax-mid_k:jmax-1] = vector[1:mid_k]
	fftout = fft.ifft(fftin)
	func = zeros(jmax, dtype='complex')
	func[0:jmax-1] = fftout[0:jmax-1]
	func[jmax-1] = func[0]
	return func

# Define function for Fourier transforming into k-space
def momentumspace(func): 											
	mid_k = int(0.5*(jmax-1))
	fftin = zeros(jmax-1, dtype='complex')
	fftin[0:jmax-1] = func[0:jmax-1] + 0.0j
	fftout = fft.fft(fftin)
	vector = zeros(jmax, dtype='complex')
	vector[mid_k:jmax] = fftout[0:mid_k+1]
	vector[1:mid_k] = fftout[jmax-mid_k:jmax-1]
	vector[0] = vector[jmax-1].conjugate()
	return vector

# Define function for generating the Hartree potential for a given charge density
def Hartree(n):
	n_k = momentumspace(n)*dx							 												
	X_x = zeros(jmax)
	for i in range(jmax):
		x = i*dx-0.5*L
		X_x[i] = 1.0/(abs(x)+1.0)
	X_k = momentumspace(X_x)*dx/L
	V_k = zeros(jmax, dtype='complex')
	V_k[:] = X_k[:]*n_k[:]
	fftout = realspace(V_k).real*L/dx
	V_hx = zeros(jmax)
	V_hx[0:0.5*(jmax+1)] = fftout[0.5*(jmax-1):jmax]
	V_hx[0.5*(jmax+1):jmax-1] = fftout[1:0.5*(jmax-1)]
	V_hx[jmax-1] = V_hx[0]
	return V_hx

# Calculation of the current density via the continuity equation
def Currentdensity(j, n): 											
	J = zeros(jmax, dtype ='float')
	if j != 0:
		for i in range(jmax):			
			for k in range(i+1):
				J[i] += -dx*(n[j,k]-n[j-1,k])/dt
	nmaxl = 0									
	imaxl = 0
	for i in range(int(0.5*(jmax-1))+1):
		if n[j,i]>nmaxl:
			nmaxl = n[j,i]
			imaxl = i
	nmaxr = 0
	imaxr = 0
	for l in range(int(0.5*(jmax-1))+1):
		i = int(0.5*(jmax-1)+l)
		if n[j,i]>nmaxr:
			nmaxr = n[j,i]
			imaxr = l
	U = zeros(jmax)
	U[:] = J[:]/n[j,:]	
	dUdx = zeros(jmax)
	for i in range(imaxl+1):
		l = imaxl-i
		if n[j,l] < 1e-6:
			dUdx[:] = gradient(U[:], dx)
			U[l] = 8*U[l+1]-8*U[l+3]+U[l+4]+dUdx[l+2]*12.0*dx
	for i in range(int(0.5*(jmax-1)-imaxr+1)):
		l = int(0.5*(jmax-1)+imaxr+i)
		if n[j,l] < 1e-6:
			dUdx[:] = gradient(U[:], dx)
			U[l] = 8*U[l-1]-8*U[l-3]+U[l-4]-dUdx[l-2]*12.0*dx
	J[:] = n[j,:]*U[:]								 												
	return J[:]

# LDA approximation for XC potential
def XC(n):
        V_xc = zeros(jmax)
        if (NE == 1):
          V_xc[:] = ((-1.389 + 2.44*n[:] - 2.05*(n[:])**2)*n[:]**0.653) 
        elif (NE == 2):
          V_xc[:] = ((-1.19 + 1.77*n[:] - 1.37*(n[:])**2)*n[:]**0.604) 
        else:
          V_xc[:] = ((-1.24 + 2.1*n[:] - 1.7*(n[:])**2)*n[:]**0.61) 
        return V_xc[:]

# LDA approximation for XC energy
def EXC(n):
        E_xc_LDA = 0.0
        if (NE == 1):
          for i in range(jmax-1):
              e_xc_LDA = ((-0.84 + 0.92*n[i] - 0.56*(n[i])**2)*n[i]**0.653) 
              Increase = (n[i])*(e_xc_LDA)*dx
              E_xc_LDA += Increase
        elif (NE == 2):
          for i in range(jmax-1):
              e_xc_LDA = ((-0.74 + 0.68*n[i] - 0.38*(n[i])**2)*n[i]**0.604) 
              Increase = (n[i])*(e_xc_LDA)*dx
              E_xc_LDA += Increase
        else:
          for i in range(jmax-1):
              e_xc_LDA = ((-0.77 + 0.79*n[i] - 0.48*(n[i])**2)*n[i]**0.61) 
              Increase = (n[i])*(e_xc_LDA)*dx
              E_xc_LDA += Increase
        return E_xc_LDA

# Error in the LDA approximation
def error(E_xc_LDA, E_xc_Exact): 
        if (E_xc_Exact != 0):
            #print 'XC energy (Exact): E_xc =', (round(E_xc_Exact, 4))
            #print 'XC energy (LDA)  : E_xc_LDA =', (round(E_xc_LDA, 4))
            if (E_xc_LDA <= E_xc_Exact): 
                LDA_error = E_xc_Exact - E_xc_LDA 
                LDA_error = abs(LDA_error)
                #print 'Absolute error   : dE_xc =', (round(LDA_error, 4))
                LDA_error_percentage = LDA_error/E_xc_Exact
                LDA_error_percentage = LDA_error_percentage*100
                #print 'E_xc_LDA is', (round(LDA_error_percentage, 2)),'% too low' 
            else:
                LDA_error = E_xc_LDA - E_xc_Exact
                #print 'Absolute error   : dE_xc =', (round(LDA_error, 4))
                LDA_error_percentage = abs(LDA_error/E_xc_Exact)
                LDA_error_percentage = LDA_error_percentage*100
                #print 'E_xc_LDA is', (round(LDA_error_percentage, 2)),'% too high' 
        else:
            print 'LDA: exchange-correlation energy: ', (round(E_xc_LDA, 4))

# Print statements 
def PS(text): 
        sys.stdout.write('\033[K')
	sys.stdout.flush()
	sys.stdout.write('\r' + text)
	sys.stdout.flush()

# Solve the Crank Nicolson equation
def CrankNicolson(V_KS, Psi0, n, j): 
	Mat = LHS(V_KS, j) # The Hamiltonian here is using the Kohn-Sham potential. 												
	Mat = Mat.tocsr()
	Matin = -(Mat-sparse.identity(jmax, dtype=cfloat)) + sparse.identity(jmax, dtype=cfloat)
	B0 = Matin*Psi0[j-1,:] # Solve the Crank Nicolson equation to get the wave-function at dt later.
	Psi0[j,:] = spla.spsolve(Mat, B0) 														
	n[j,:] = abs(Psi0[j,:])**2
	return n, Psi0

# Left hand side of the Crank Nicolson method
def LHS(V_KS, j): 												
	for i in range(jmax):
	    CNLHS[i,i] = 1.0+0.5j*dt*(1.0/dx**2+V_KS[i])
	    if i < jmax-1:
		CNLHS[i,i+1] = -0.5j*dt*(0.5/dx)/dx
	    if i > 0:
		CNLHS[i,i-1] = -0.5j*dt*(0.5/dx)/dx
	return CNLHS

# Find groundstate values
j = 0
for i in range(jmax): # Initial guess for V_KS (External Potential)
    V_KS[j,i] = Potential(i,j) 
    V_KS_old[j,i] = Potential(i,j)
V_ext[:] = V_KS[j,:] 
n_x[j,:], Psi0[j,:] = TISE(V_KS[j,:],j) # Solve Schrodinger Equation initially
n_x_old[j,:] = n_x[j,:]
while(Cost>tol):  
    V_h[j,:] = Hartree(n_x[j,:]) # Calculate Hartree, XC and KS potential
    V_xc[j,:] = XC(n_x[j,:])
    V_hxc[j,:] = V_h[j,:] + V_xc[j,:]
    V_KS[j,:] = V_ext[:] + V_hxc[j,:]
    V_KS[j,:] = Mix*V_KS[j,:] + (1.0-Mix)*V_KS_old[j,:] # Mix KS potential
    n_x[j,:], Psi0[j,:] = TISE(V_KS[j,:],j) # Solve Schrodinger Equation
    Cost = sum(abs(n_x[j,:]-n_x_old[j,:])*dx)
    string = 'LDA: ground-state KS potential: run = ' + str(Run) + ', charge density cost = ' + str(Cost)
    PS(string)
    n_x_old[j,:] = n_x[j,:]
    V_KS_old[j,:] = V_KS[j,:]
    Run = Run + 1
V_h[j,:] = Hartree(n_x[j,:]) # Calculate Hartree and XC potential with correct density 
V_xc[j,:] = XC(n_x[j,:])
V_hxc[j,:] = V_h[j,:] + V_xc[j,:]
E_xc_LDA = EXC(n_x[j,:]) # XC energy 
error(E_xc_LDA, E_xc_Exact)    
if (TD == 0): 
   f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_1gs_lda_vks.db', 'w') # KS potential	
   pickle.dump(V_KS[0,:],f)				
   f.close()
   f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_1gs_lda_vh.db', 'w') # H potential	
   pickle.dump(V_h[0,:],f)				
   f.close()
   f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_1gs_lda_vxc.db', 'w') # XC potential	
   pickle.dump(V_xc[0,:],f)				
   f.close()
   f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_1gs_lda_den.db', 'w') # Density	
   pickle.dump(n_x[0,:],f)				
   f.close()


# Find realtime values
if (TD ==1):
   for i in range(jmax): # Perturbed V_KS
       V_KS[1,i] = V_KS[0,i] + Potential(i,1)  
   V_ext[:] = V_ext[:] + Potential(i,1) # Perturbed external potential 
   for j in range(1,imax): # Evolve TDSE using Crank-Nicolson scheme
       string = 'LDA: evolving through real time: t = ' + str(j*dt) 
       PS(string)
       n_x, Psi0 = CrankNicolson(V_KS[j,:], Psi0, n_x, j)  
       J_x[j,:] = Currentdensity(j,n_x)
       V_h[j,:] = Hartree(n_x[j,:]) 
       V_xc[j,:] = XC(n_x[j,:])
       if(j != imax-1):
           V_KS[j+1,:] = V_ext[:] + V_h[j,:] + V_xc[j,:] # Update KS potential
   for i in range(jmax): # Ground-state external potential
      V_ext[i] = Potential(i,0) 
   f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_1td_lda_vks.db', 'w') # KS potential	
   pickle.dump(V_KS,f)				
   f.close()
   f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_1td_lda_vh.db', 'w') # H potential	
   pickle.dump(V_h,f)				
   f.close()
   f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_1td_lda_vxc.db', 'w') # XC potential	
   pickle.dump(V_xc,f)				
   f.close()
   f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_1td_lda_den.db', 'w') # Density	
   pickle.dump(n_x,f)				
   f.close()
   print


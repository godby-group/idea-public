# coding=utf-8
######################################################################################
# Name: MLP 2electrons                                                               #
######################################################################################
# Author(s):                                                                         #
######################################################################################
# Description:                                                                       #
# Computes MLP approximations if pm.MLP=1 ground-state and td                        #
# MLP = pm.f*SOA + (1-pm.f)*LDA                                                      #
# elf (savin et al), cost function                                                   #
######################################################################################
# Notes:  if f = 'e' in parameters file, the MLP is optimazed (it calculates the     #
#         best f in relation with the elf both in gs and td (use cautiously)         #
#         if the current is very noise at the edge, uncomment last lines in function #
#         DensityCurrent to apply a mask that removes current noise at the edge, or  #
#         change alpha (noise filter in the KS potential)                            #
#                                                                                    #
#                                                                                    #
#                                                                                    #
#                                                                                    #
#                                                                                    #
######################################################################################


# Library imports
from math import *									
from numpy import *
import numpy as np
import scipy
from scipy.linalg import eig_banded, solve
import parameters as pm
import sys
import pickle 
from scipy import sparse
from scipy import special
from scipy.sparse import linalg as spla
from scipy import linalg as la
import sprint
import os
import os.path

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
NE = pm.NE
Mix = 0.1   
tol = 1e-11 
Cost = 1 
Run = 1
x_cut = 45 # both for flatten vks and defined soa(infinity)
alpha= 0.1 # Strength of noise control

# Matrices
Psi0 = zeros((imax,jmax), dtype='complex') # Wave function for each particle						
Psi1 = zeros((imax,jmax), dtype='complex')
V_h = zeros((imax,jmax)) # Potentials
V_xc = zeros((imax,jmax)) 
V_hxc = zeros((imax,jmax)) 
n_x = zeros((imax,jmax), dtype ='float') # Charge Density
n_x_old = zeros((imax,jmax), dtype='float') 
J_x = zeros((imax,jmax)) # Current Density 
T = zeros((2,jmax), dtype='complex') # Kinetic Energy operator
T[0,:] = ones(jmax)/dx**2 								
T[1,:] = -0.5*ones(jmax)/dx**2 
V_KS = zeros((imax,pm.jmax)) # Kohn-Sham potential
V_KS_old = zeros((imax,jmax)) 
V_ext = zeros(jmax) # External potential
CNLHS = sparse.lil_matrix((jmax,jmax),dtype='complex') # Matrix for the left hand side of the Crank Nicholson method
Mat = sparse.lil_matrix((jmax,jmax),dtype='complex')   
Matin = sparse.lil_matrix((jmax,jmax),dtype='complex') # Inverted Matrix for the right hand side of the Crank Nicholson method 
K=[Psi0[0,:],Psi1[0,:]]

						
ff = zeros((imax,jmax)) # f weight
if (type(pm.f)==float):
        for i in range(imax):
                for j in range(jmax):
                        ff[i,j]=pm.f

elif (str(pm.f)=='e'):
	print 'f optimized, starting with f=0.0'


else:
        print('error in f parameter')
        sys.exit(0)
			
def func_lin(x,a,b):
	return a+b*x
def func_Lin_inv(x,a,b):
	return a-b*x

def func(x,a,b):
	return a+b/x
def func_inv(x,a,b):
	return a-b/x


def extrap(mb_vks,x_a,x_b):

    xx = np.arange(-xmax,xmax+dx,dx)
    x = xx[x_a:x_b]
    v = mb_vks[x_a:x_b]
    vks_extrap = mb_vks

    #for m in range (1,x_a):
	#u_e[x_a-m-1]= u[x_a-m]+(u[x_a-m]-u[x_a-m+1])/dx
	#V_KS[j,jmax-x_cut+m]=V_KS[j,jmax-x_cut+m-1]+(V_KS[j,jmax-x_cut]-V_KS[j,jmax-x_cut-1])/dx
    popt, pcov = curve_fit(func, x, v)
    for l in range(x_a):	
    	vks_extrap[l] = func(-xmax+l*dx,*popt)
	vks_extrap[jmax-l-1] = func_inv(xmax-l*dx, *popt)
    return vks_extrap, popt[0]

def extrap_lin(mb_vks,x_a,x_b):

    xx = np.arange(-xmax,xmax+dx,dx)
    x = xx[x_a:x_b]
    v = mb_vks[x_a:x_b]
    vks_extrap = mb_vks

    #for m in range (1,x_a):
	#u_e[x_a-m-1]= u[x_a-m]+(u[x_a-m]-u[x_a-m+1])/dx
	#V_KS[j,jmax-x_cut+m]=V_KS[j,jmax-x_cut+m-1]+(V_KS[j,jmax-x_cut]-V_KS[j,jmax-x_cut-1])/dx
    popt, pcov = curve_fit(func_lin, x, v)
    for l in range(x_a):	
    	vks_extrap[l] = func_lin(-xmax+l*dx,*popt)
	vks_extrap[jmax-l-1] = func_lin_inv(xmax-l*dx, *popt)
    return vks_extrap, popt[0]

												
# Potential Generator
def Potential(i,j=0): 
        x = -xmax + i*dx 
        if (j==0): 
            V = pm.well(x)
        else: 
            V = pm.petrb(x)
        return V

# Solve TISE
def TISE(V_KS,j=0):  					                         											
        HGS = copy(T) # Reset Hamiltonian									
        HGS[0,:] += V_KS[:]								
        K, U = eig_banded(HGS, True) # Returns eigenvalues (K) and eigenvectors (U)					 									
        Psi0[j,:] = U[:,0]/sqdx # Normalise the wave functions 							
        Psi1[j,:] = U[:,1]/sqdx
        n_x[j,:] = abs(Psi0[j,:])**2+abs(Psi1[j,:])**2 # Calculate charge density				   
        return n_x[j,:], Psi0[j,:], Psi1[j,:]

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
	u = np.zeros((jmax,jmax))
	for i in range(jmax):
		for k in range(jmax):	
			u[i,k] = 1.0/(abs(i*dx-k*dx)+1)
	return np.dot(u,n)*dx

# LDA Exchange-Correlation
def XC(n,j=0):
        V_xc = zeros((imax,jmax))
        if (pm.LDA_NE == 1):
          V_xc[j,:] = ((-1.389 + 2.44*n[:] - 2.05*(n[:])**2)*n[:]**0.653)
        elif (pm.LDA_NE == 2):
          V_xc[j,:] = ((-1.19 + 1.77*n[:] - 1.37*(n[:])**2)*n[:]**0.604)
        elif (pm.LDA_NE == 3):
          V_xc[j,:] = ((-1.24 + 2.1*n[:] - 1.7*(n[:])**2)*n[:]**0.61)
        return V_xc[j,:]

def XC(n):
        V_xc = zeros(jmax)
        V_xc[:] = ((-1.389 + 2.44*n[:] - 2.05*(n[:])**2)*n[:]**0.653)
        return V_xc

# Print statements 
def PS(text): 
        sys.stdout.write('\033[K')
	sys.stdout.flush()
	sys.stdout.write('\r' + text)
	sys.stdout.flush()

# Solve the Crank Nicolson equation
def CrankNicolson(V_KS, Psi0, Psi1, n, j): 
	Mat = LHS(V_KS, j) # The Hamiltonian here is using the Kohn-Sham potential. 												
	Mat = Mat.tocsr()
	Matin = -(Mat-sparse.identity(jmax, dtype=cfloat)) + sparse.identity(jmax, dtype=cfloat)
	B0 = Matin*Psi0[j-1,:] # Solve the Crank Nicolson equation to get the wave-function at dt later.
	Psi0[j,:] = spla.spsolve(Mat, B0) 		
	B1 = Matin*Psi1[j-1,:]
	Psi1[j,:] = spla.spsolve(Mat, B1)						 										
	n[j,:] = abs(Psi0[j,:])**2+abs(Psi1[j,:])**2
	return n[j,:], Psi0, Psi1

# Left hand side of the Crank Nicolson method
def LHS(V_KS, j): 												
	for i in range(jmax):
	    CNLHS[i,i] = 1.0+0.5j*dt*(1.0/dx**2+V_KS[i])
	    if i < jmax-1:
		CNLHS[i,i+1] = -0.5j*dt*(0.5/dx)/dx
	    if i > 0:
		CNLHS[i,i-1] = -0.5j*dt*(0.5/dx)/dx
	return CNLHS

# Given n, return potential V_SOA
def SOA(n):	
	V_SOA =  (1.0/4)*(np.gradient( np.gradient(np.log(n), dx), dx)) + (1.0/8)*np.gradient(np.log(n),dx)**2
	return V_SOA

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
	
	#for s in range(0,pm.jmax): 
	#      x = -pm.xmax + s*dx
        #      J[s]=J[s]*np.exp(-4*(x/pm.xmax)**(12))
								 												
	return J[:]

# Calculate elf
def getElf(den, KS, j=None, posDef=False):

    # The single particle kinetic energy density terms
    grad1 = np.gradient(KS[0], pm.deltax)
    grad2 = np.gradient(KS[1], pm.deltax)

    # Gradient of the density
    gradDen = np.gradient(den, pm.deltax)

    # Unscaled measure
    c = np.arange(den.shape[0])
    if j == None:
        c = (np.abs(grad1)**2 + np.abs(grad2)**2)   \
            - (1./4.)* ((np.abs(gradDen)**2)/den)
    elif (j.shape == den.shape):
        c = (np.abs(grad1)**2 + np.abs(grad2)**2)   \
            - (1./4.)* ((np.abs(gradDen)**2)/den) - (j**2)/(den)

    else:
        print "Error: Invalid Current Density given to ELF"
        print "       Either j wasn't a numpy array or it "
        print "       was of the wrong dimensions         "
        return None

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
    c_h = (1./6.)*(np.pi**2)*(den**3)

    return c_h

j = 0

# Find groundstate values
for i in range(jmax): # Initial guess for V_KS (External Potential)
    V_KS[j,i] = Potential(i,j) 
    V_KS_old[j,i] = Potential(i,j)
    
V_ext[:] = V_KS[j,:] 
n_x[j,:], Psi0[j,:] , Psi1[j,:] = TISE(V_KS[j,:],j) # Solve Schrodinger Equation initially
n_x_old[j,:] = n_x[j,:]

V_SOA = zeros((imax,jmax))
V_xc_LDA = zeros((imax,jmax))
V_LDA = zeros((imax,jmax))

V_SOA[j,:]=SOA(n_x[j,:])
V_xc_LDA[j,:]=XC(n_x[j,:])
V_h[j,:] = Hartree(n_x[j,:])
V_LDA[j,:]=V_xc_LDA[j,:]+V_ext[:]+V_h[j,:] 

V_KS[j,:] = ff[j,:]*V_SOA[j,:]+(1-ff[j,:])*V_LDA[j,:] # Initial V_MLP ks
 
while(Cost>tol):
    n_x[j,:], Psi0[j,:] , Psi1[j,:] = TISE(V_KS[j,:],j) # Solve Schrodinger Equation
    V_SOA[j,:]= SOA(n_x[j,:])
    
    V_xc_LDA[j,:]=XC(n_x[j,:])
    V_h[j,:] = Hartree(n_x[j,:])
    V_LDA[j,:]=V_xc_LDA[j,:]+V_ext[:]+V_h[j,:]
    if (str(pm.f)=='e'):
    	K=[Psi0[0,:],Psi1[0,:]]
        elf = getElf(n_x[j,:],K)
	elf_whole = np.sum(n_x[j,:]*elf[:]*dx)/NE
	for i in range(jmax):
		ff[0,i]= 0.00022*np.exp(8.5*elf_whole) # if you use getElf
    V_KS[j,:] = ff[j,:]*V_SOA[j,:]+(1-ff[j,:])*V_LDA[j,:] 
    V_KS[j,:] = Mix*V_KS[j,:] + (1.0-Mix)*V_KS_old[j,:] # Mix KS potential# 

    
    Cost = sum(abs(n_x[j,:]-n_x_old[j,:])*dx)
    string = 'MLP: ground-state KS potential: run = ' + str(Run) + ', charge density cost (convergence)= ' + str(Cost) + ', f = ' +str(ff[0,5])
    PS(string)
    n_x_old[j,:] = n_x[j,:]
    V_KS_old[j,:] = V_KS[j,:]
    

    Run = Run + 1
print "\n"



# Calculate cost
if (pm.cost == 1):
	# open mb data
	file_name = 'outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_2gs_ext_den.db'
	file_obj = open(file_name,'r')
	mb_den = pickle.load(file_obj)
	n_x_e = zeros(jmax) # define n_x and K so that getElf is ok with them
	n_x_e[:] = n_x[j,:]
	cost=zeros(jmax)
	cost[:]=np.abs(mb_den[:]-n_x_e[:])
        cost_whole = np.sum(mb_den[:]*cost[:]*dx)
        cost_wholeld = np.sum(1/(mb_den[:])*cost[:]*dx)
        cost_wholenw = np.sum(cost[:]*dx)
        print "\ncost_whole in high density: ", cost_whole
	print "cost_whole not weighted: ", cost_wholenw
#	print "\ncost_whole in low density: ", cost_wholeld

# elf as a whole
K=[Psi0[0,:],Psi1[0,:]]
elf = getElf(n_x[j,:],K)
elf_whole = np.sum(n_x[j,:]*elf[:]*dx)/NE
print "elf_whole", elf_whole
print "\n"


# Output results
if (TD == 0):
   f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(NE) + 'gs_mlp_f'+str(pm.f)+'_vks.db', 'w') # KS potentia$
   pickle.dump(V_KS[0,:],f)
   f.close()
   f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(NE) + 'gs_mlp_f'+str(pm.f)+'_den.db', 'w') # Density
   pickle.dump(n_x[0,:],f)
   f.close()
   f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(NE) + 'gs_mlp_f'+str(pm.f)+'_elf.db' , 'w') #elf
   pickle.dump(elf[:],f)
   f.close()
   f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(NE) + 'gs_ext_vxt.db' , 'w') # v_ext
   pickle.dump(V_ext,f)
   f.close()

   if (pm.cost == 1):
    	f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(NE) + 'gs_mlp_f'+str(pm.f)+'_cost.db', 'w') # cost      
        pickle.dump(cost[:],f)
        f.close()



def SOA_td(n,jj, du):
	soa = SOA(n)
	soatd = np.zeros(jmax) 
	for ii in range(0,jmax):
		soatd[ii] =  soa[ii]-0.5*(jj[ii]*jj[ii])/(n[ii]*n[ii])-np.sum(du[:(ii+1)]*dx)
	return soatd
	
def Filter(V,t,exp):
    A_Kspace=np.zeros(pm.jmax,dtype='complex')
    A_Kspace=momentumspace(V)
    A_Kspace[:]*=exp[:]
    V=realspace(A_Kspace).real
    return V

##############################################
# Find realtime values
if(TD==1):
  
   exp=np.zeros(pm.jmax,dtype='float')	
   j_x_old = np.zeros(jmax)
   nx_old = n_x[0,:]
   f = pm.f
   du = np.zeros(jmax)
   V_KS_nofilter = zeros((imax,jmax))
   V_KS_nofilter[0,:] = V_KS[0,:]
  
   for i in range(jmax): # Perturbed V_KS
       V_KS[1,i] = V_KS[0,i] + Potential(i,1)  
       V_KS_nofilter[1,i] = V_KS[0,i] + Potential(i,1)
       V_ext[i] = V_ext[i] + Potential(i,1) # Perturbed external potential 
       exp[i]=math.exp(-alpha*(i*pm.deltax-pm.xmax)**2)
   for t in range(1,imax): # Evolve TDSE using Crank-Nicolson scheme
       string = 'MLP: evolving through real time: t = ' + str(t*dt) + ', elfw = ' + str(elf_whole) + ', f= ' + str(ff[t-1,3]) 
       PS(string)
       n_x[t,:], Psi0, Psi1 = CrankNicolson(V_KS[t,:], Psi0, Psi1, n_x, t) 
       J_x[t,:] = Currentdensity(t,n_x)  
       du[:] = (J_x[t,:]/n_x[t,:]-j_x_old[:]/nx_old[:])/dt
       V_h[t,:] = Hartree(n_x[t,:]) 
       V_xc[t,:] = XC(n_x[t,:])


       V_SOA[t,:] = SOA_td(n_x[t,:], J_x[t,:] , du)
      
       V_LDA[t,:] = V_ext[:] + V_h[t,:] + V_xc[t,:]
       
       if(t != imax-1):
	     
           if (str(pm.f)=='e'):
    	   	K=[Psi0[t,:],Psi1[t,:]]
        	elf = getElf(n_x[t,:],K)
		elf_whole = np.sum(n_x[t,:]*elf[:]*dx)/NE
		for i in range(jmax):
			ff[t,i]= ff[t,i]= 0.00022*np.exp(8.5*elf_whole)   
                                         

           V_KS_nofilter[t+1,:] = ff[t,:]*V_SOA[t,:]+(1-ff[t,:])*V_LDA[t,:] # Update KS potential
           V_KS[t+1,:] = Filter(V_KS_nofilter[t+1,:],t+1,exp) # Remove high frequencies from ks
         
       j_x_old[:] = J_x[t,:]
       nx_old[:] = n_x[t,:]
		


   f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_mlp_f'+str(pm.f)+'_vks.db', 'w') # KS potential
   pickle.dump(V_KS,f)				
   f.close()

   f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_mlp_f'+str(pm.f)+'_den.db', 'w') # den
   pickle.dump(n_x,f)				
   f.close()
   output_file = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_mlp_f'+str(pm.f)+'_cur.db','w') # Current density 
   pickle.dump(J_x,output_file)
   output_file.close()
   
   # Calculate cost
   if (pm.cost == 1):
	# open mb data
	file_name = 'outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_2td_ext_den.db'
	file_obj = open(file_name,'r')
	mb_den_td1 = pickle.load(file_obj)
	mb_den_td= np.array(mb_den_td1)
	cost_td = zeros((imax,jmax)) 
	cost_whole_td = zeros(imax)
	for j in range (imax):
		cost_td[j,:]=np.abs(n_x[j,:]-mb_den_td[j,:])
		cost_whole_td[j]= np.sum(mb_den_td[j,:]*cost_td[j,:]*dx)
          	
        output_file = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_2td_mlp_f'+str(pm.f)+'_costwhole.db','w')  
        pickle.dump(cost_whole_td,output_file)
        output_file.close()


   print







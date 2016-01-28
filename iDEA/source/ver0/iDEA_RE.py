######################################################################################
# Name: Reverse Engineering                                                          #
######################################################################################
# Author(s): Matt Hodgson, James Ramsden, Matthew Smith                              # 
######################################################################################
# Description:                                                                       #
# Computes exact VKS, VH, VXC using the RE algorithm from the exact density          #
#                                                                                    #
#                                                                                    #
######################################################################################
# Notes:                                                                             #
# Ground-state calculations are usually fast and stable, this may vary if the system #
# is particularly difficult to RE, i.e. if the system has regions of very small      #
# electron density. To control the rate of convergence, and the stability of the     #
# GSRE, use the variables 'mu' and 'p'. p is used for the rate of convergence        #
# and for bringing more attention to regions of low density. mu is used for          #
# stabiliesing the algorithm. Time-dependent RE is much more difficult, and is       #
# dependent on the system. If the TDRE is not converging the most likely reason      #
# is that dt is too big. There could also be a problem with noise. Noise should be   #
# obvious in he velocity field (current/density). If noise is dominating the system, #
# try changing the noise filtering value 'alpha'. Alpha controls how much of the     #
# high frequency terms are removed from the KS vector potential.                     #
#                                                                                    #
######################################################################################

# Import librarys.
from math import *									
from numpy import *
from scipy.linalg import eig_banded, solve
from scipy import linalg as la
from scipy import special
from scipy import sparse
from scipy.sparse import linalg as spla
import parameters as pm
import pickle
import sys
import os
import RE_Utilities
import time
import sprint

# Variable initialisation
global upper_bound, sqdx, frac1, frac2
global T, n_MB, n_KS, J_KS, J_MB, V_h, V_xc, cost_J, CNRHS, CNLHS, Mat, Matin, V_ext, A_KS, A_min, Apot, V_Hxc, cost_n_GS, cost_n, cost_J, U_KS, U_MB, V_KS, mu
sqdx=sqrt(pm.deltax)							 												
upper_bound = int((pm.jmax-1)/2.0)								
L = 2*pm.xmax
imax = pm.imax
if pm.TD==0:
    imax = 1
mu = 1.0				
frac1 = 1.0/3.0
frac2 = 1.0/24.0

# Initalise matrices
T = zeros((2,pm.jmax),dtype='complex')
T[0,:] = ones(pm.jmax)/pm.deltax**2									
T[1,:] = -0.5*ones(pm.jmax)/pm.deltax**2
Psi0 = zeros((imax,pm.jmax), dtype='complex')
Psi1 = zeros((imax,pm.jmax), dtype='complex')
Psi2 = zeros((imax,pm.jmax), dtype='complex')									
n_MB = zeros((imax,pm.jmax),dtype='float_')									
n_KS = zeros((imax,pm.jmax),dtype='float_')
J_KS = zeros((imax,pm.jmax),dtype='float_')									
J_MB = zeros((imax,pm.jmax),dtype='float_')
cost_n = zeros(imax)								
cost_J = zeros(imax)								
CNRHS = zeros(pm.jmax, dtype='complex')					
CNLHS = sparse.lil_matrix((pm.jmax,pm.jmax),dtype='complex')					
Mat = sparse.lil_matrix((pm.jmax,pm.jmax),dtype='complex')					
Matin = sparse.lil_matrix((pm.jmax,pm.jmax),dtype='complex')				
V_ext = zeros((imax,pm.jmax))
V_KS = zeros((imax,pm.jmax))
V_h = zeros((imax,pm.jmax))									
V_xc = zeros((imax,pm.jmax))
V_Hxc = zeros((imax,pm.jmax))								
A_KS = zeros((imax,pm.jmax))									
A_min = zeros(pm.jmax)									
Apot = zeros(pm.jmax)
U_KS = zeros((imax,pm.jmax))
U_MB = zeros((imax,pm.jmax))
U_t = zeros((2,pm.jmax,pm.NE),dtype='complex')
B = zeros(pm.NE,dtype='complex')

# Function to read inputs
def ReadInput(approx):
    global n_MB
    if(pm.TD):
        file_name = 'outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_' + str(approx) + '_den.db'
        input_file = open(file_name,'r')
        data = pickle.load(input_file)
        n_MB[:,:] = data
    else:
        file_name = 'outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_' + str(approx) + '_den.db'
        input_file = open(file_name,'r')
        data = pickle.load(input_file)
        n_MB[0,:] = data
    return

# Function to calculate the ground-state potential
def CalculateGroundstate(V_KS,n_MB,mu):

    #Build Hamiltonian
    p=0.05 #Determines the rate of convergence of the ground-state RE
    HGS=copy(T)
    V_KS[0,:]+=mu*(n_KS[0,:]**p-n_MB[0,:]**p)
    HGS[0,:]+=V_KS[0,:]

    #Solve KS equations
    K,U=eig_banded(HGS,True)
    U[:,:]/=sqdx #Normalise

    #Calculate density and cost function
    n_KS[0,:]=0
    for i in range(pm.NE):
        n_KS[0,:]+=abs(U[:,i])**2
    cost_n_GS=sum(abs(n_MB[0,:]-n_KS[0,:]))*pm.deltax
    return V_KS,n_KS,cost_n_GS,U

# Function to load or force calculation of the ground-state potential
def GroundState(V_KS,n_MB,mu):
    print 'REV: calculating ground-state Kohn-Sham potential'
    for i in range(pm.jmax):
        x = i*pm.deltax-pm.xmax
        V_KS[0,i] = pm.well(x) #Initial guess for KS potential
        V_ext[0,i] = pm.well(x)
    V_KS,n_KS,cost_n_GS,U=CalculateGroundstate(V_KS,n_MB,0)
    print 'REV: initial guess cost = %s' % cost_n_GS
    iterations = 0
    max_iterations = 10000
    while cost_n_GS > 1e-13:
        cost_old = cost_n_GS
        string = 'REV: charge density cost = ' + str(cost_old)
	sprint.sprint(string,1,1,pm.msglvl)
	sprint.sprint(string,2,1,pm.msglvl)
        V_KS,n_KS,cost_n_GS,U=CalculateGroundstate(V_KS,n_MB,mu)
	if abs(cost_n_GS-cost_old) < 1e-15 or cost_n_GS > cost_old:
	    mu = mu*0.5
        if mu < 1e-15:
            break
        iterations += 1
    V_h[0,:] = Hartree(n_KS[0,:])
    V_Hxc[0,:] = V_KS[0,:]-V_ext[0,:]
    V_xc[0,:] = V_Hxc[0,:]-V_h[0,:]
    z=0
    for k in range(pm.NE):
        U_t[z,:,k]=U[:,k]
    return

# Function used in calculation of the Hatree potential
def realspace(vector):
    mid_k=int(0.5*(pm.jmax-1))
    fftin=zeros(pm.jmax-1, dtype='complex')
    fftin[0:mid_k+1]=vector[mid_k:pm.jmax]
    fftin[pm.jmax-mid_k:pm.jmax-1]=vector[1:mid_k]
    fftout=fft.ifft(fftin)
    func=zeros(pm.jmax, dtype='complex')
    func[0:pm.jmax-1]=fftout[0:pm.jmax-1]
    func[pm.jmax-1]=func[0]
    return func

# Function used in calculation of the Hatree potential
def momentumspace(func):
    mid_k=int(0.5*(pm.jmax-1))
    fftin=zeros(pm.jmax-1, dtype='complex')
    fftin[0:pm.jmax-1] = func[0:pm.jmax-1] + 0.0j
    fftout=fft.fft(fftin)
    vector=zeros(pm.jmax, dtype='complex')
    vector[mid_k:pm.jmax]=fftout[0:mid_k+1]
    vector[1:mid_k]=fftout[pm.jmax-mid_k:pm.jmax-1]
    vector[0]=vector[pm.jmax-1].conjugate()
    return vector

# Function used to calculate the Hatree potential
def Hartree(n):
    n_k=momentumspace(n)*pm.deltax
    X_x=zeros(pm.jmax)
    for i in range(pm.jmax):
        x=i*pm.deltax-pm.xmax
        X_x[i]=1.0/(abs(x)+pm.acon)
    X_k=momentumspace(X_x)*pm.deltax/L
    V_k=zeros(pm.jmax, dtype='complex')
    V_k[:]=X_k[:]*n_k[:]
    fftout=realspace(V_k).real*2*pm.xmax/pm.deltax
    V_hx=zeros(pm.jmax)
    V_hx[0:0.5*(pm.jmax+1)]=fftout[0.5*(pm.jmax-1):pm.jmax]
    V_hx[0.5*(pm.jmax+1):pm.jmax-1]=fftout[1:0.5*(pm.jmax-1)]
    V_hx[pm.jmax-1]=V_hx[0]
    return V_hx

# Function to extrapolate the current density from regions of low density to the system's edges
def ExtrapolateCD(J,j,n,n_MB):
    imaxl=0
    nmaxl=0.0
    imaxr=0									
    nmaxr=0.0
    for l in range(upper_bound+1):
        if n_MB[j,l]>nmaxl:
            nmaxl=n_MB[j,l]
	    imaxl=l
        i=upper_bound+l-1
        if n_MB[j,i]>nmaxr:
             nmaxr=n_MB[j,i]
             imaxr=l
    U=zeros(pm.jmax)
    U[:]=J[:]/n[j,:]
    dUdx=zeros(pm.jmax)
    for i in range(imaxl+1):
        l=imaxl-i
        if n_MB[j,l]<1e-6:
            dUdx[:]=gradient(U[:], pm.deltax)
            U[l]=8*U[l+1]-8*U[l+3]+U[l+4]+dUdx[l+2]*12.0*pm.deltax
    for i in range(int(0.5*(pm.jmax-1)-imaxr+1)):
        l=int(0.5*(pm.jmax-1)+imaxr+i)
        if n_MB[j,l]<1e-6:
            dUdx[:]=gradient(U[:], pm.deltax)
            U[l]=8*U[l-1]-8*U[l-3]+U[l-4]-dUdx[l-2]*12.0*pm.deltax
    J[:]=n[j,:]*U[:]								
    return J

# Function to extrapolate the KS vector potential from regions of low density to the system's edges
def ExtrapolateVectorPotential(A_KS,n_MB):
    imaxl=0
    nmaxl=0.0
    imaxr=0									
    nmaxr=0.0
    for i in range(upper_bound+1):
        if n_MB[j,i]>nmaxl:
            nmaxl=n_MB[j,i]
            imaxl=i
    for l in range(upper_bound+1):
        i = upper_bound +l
        if n_MB[j,i]>nmaxr:
            nmaxr=n_MB[j,i]
            imaxr=l
    dAdx=zeros(pm.jmax)
    for i in range(imaxl+1):
        l=imaxl-i
        if n_MB[j,l]<1e-6:
            dAdx[:]=gradient(A_KS[j,:], pm.deltax)
            A_KS[j,l]=8*A_KS[j,l+1]-8*A_KS[j,l+3]+A_KS[j,l+4]+dAdx[l+2]*12.0*pm.deltax 
    for i in range(upper_bound +1-imaxr):
        l=(upper_bound+imaxr+i)
        if n_MB[j,l]<1e-6:
            dAdx[:]=gradient(A_KS[j,:], pm.deltax)
            A_KS[j,l]=8*A_KS[j,l-1]-8*A_KS[j,l-3]+A_KS[j,l-4]-dAdx[l-2]*12.0*pm.deltax
    return A_KS

# Function to filter out 'noise' occuring between calculation of the exact TDSE solution and the present KS solution
def Filter(A_KS,j):
    A_Kspace = zeros(pm.jmax, dtype='complex')
    A_Kspace = momentumspace(A_KS[j,:])
    alpha = 1
    for i in range(pm.jmax):
        k=i*pm.deltax-pm.xmax
        A_Kspace[i]=A_Kspace[i]*exp(-alpha*k**2)
    A_KS[j,:]=realspace(A_Kspace).real
    return A_KS

# Function to solve TDKSEs, using the Crank-Nicolson method
def SolveSE(V_KS,A_KS,U_t,j):
    Mat=sparse.lil_matrix((pm.jmax,pm.jmax),dtype='complex')					 						
    for i in range(pm.jmax):
        Mat[i,i]=1.0+0.5j*pm.deltat*(1.0/pm.deltax**2+0.5*A_KS[j,i]**2+V_KS[j,i])
    for i in range(pm.jmax-1):
        Mat[i,i+1]=-0.5j*pm.deltat*(0.5/pm.deltax-(frac1)*1.0j*A_KS[j,i+1]-(frac1)*1.0j*A_KS[j,i])/pm.deltax
    for i in range(1,pm.jmax):
        Mat[i,i-1]=-0.5j*pm.deltat*(0.5/pm.deltax+(frac1)*1.0j*A_KS[j,i-1]+(frac1)*1.0j*A_KS[j,i])/pm.deltax
    for i in range(pm.jmax-2):	
        Mat[i,i+2]=-0.5j*pm.deltat*(1.0j*A_KS[j,i+2]+1.0j*A_KS[j,i])*(frac2)/pm.deltax
    for i in range(2,pm.jmax):
        Mat[i,i-2]=0.5j*pm.deltat*(1.0j*A_KS[j,i-2]+1.0j*A_KS[j,i])*(frac2)/pm.deltax
    Mat=Mat.tocsr()
    Matin=-(Mat-sparse.identity(pm.jmax,dtype=cfloat))+sparse.identity(pm.jmax,dtype=cfloat)
    for i in range(pm.NE):
        B[i]=Matin*U_t[z,:,i]
        z=1+(-1)**z
        U_t[z,:,i]=spla.spsolve(Mat,B[i])
        z=1+(-1)**z
    z=1+(-1)**z									
    V_KS,A_KS,n_KS,cost_n,U_t

# Function to calculate the current density
def CalculateCurrentDensity(n,n_MB):			
    J = zeros(pm.jmax)
    RE_Utilities.continuity_eqn(j+1,pm.jmax,pm.deltax,pm.deltat,n,J)
    CD = ExtrapolateCD(J,j,n,n_MB)
    return CD

# Function to calculate the KS vector (and finally scalar) potential
def CalculateKohnSham(V_KS,A_KS,J_KS,U_t):

    # Set initial trial vector potential as previous time-step's vector potential
    global tol
    A_KS[j,:]=A_KS[j-1,:] 
    SolveSE(V_KS,A_KS,U_t,j) 

    # Calculate KS charge density
    n_KS[j,:]=0
    for i in range(pm.NE):
        n_KS[j,:]+=abs(U_t[j,:,i])**2

    J_KS[j,:]=CalculateCurrentDensity(n_KS,n_MB) # Calculate KS current density
    J_MB[j,:]=CalculateCurrentDensity(n_MB,n_MB) # Calculate KS current density
    
    # Evaluate cost functions corresponding to present vector potential
    cost_J[j]=sum(abs(J_KS[j,:]-J_MB[j,:]))*pm.deltax 
    cost_n[j]=sum(abs(n_KS[j,:]-n_MB[j,:]))*pm.deltax

    # Set initial trial vector potential as reference vector potential
    A_min[:]=A_KS[j,:] 
    cost_min = 2.0
    count = 1
    countmin = 1 
    while (count<=400): #Exit condition: KS and exact current density are equal at all points
        if count%100==0:
            #print "t, count, tol, cost_J, cost_n ... ", j*pm.deltat, count, tol, cost_J[j], cost_n[j]
            if tol<1e-3:
                tol=tol*10 # Increase allowed convergence tolerance for J_check
                A_KS[j,:]=A_KS[j-1,:] # Reset vector potential
                SolveSE(V_KS,A_KS,U_t,j) # Solve Schrodinger equation for KS system using initial trial potential
                n_KS[j,:]=0
                for i in range(pm.NE):
                    n_KS[j,:]+=abs(U_t[j,:,i])**2
                J_KS[j,:]=CalculateCurrentDensity(n_KS,n_MB)
                cost_J[j]=sum(abs(J_KS[j,:]-J_MB[j,:]))*pm.deltax
                cost_n[j]=sum(abs(n_KS[j,:]-n_MB[j,:]))*pm.deltax
            else:
                A_KS[j,:]=A_min[:]
                break
        A_KS[j,:]+=(J_KS[j,:]-J_MB[j,:])/n_MB[j,:] # Update vector potential
        A_KS=ExtrapolateVectorPotential(A_KS,n_MB) # Extrapolate vector potential from low density regions to edges of system
        A_KS=Filter(A_KS,j) # Remove high frequencies from vector potential
        SolveSE(V_KS,A_KS,U_t,j) # Solve Schrodinger equation using updated vector potential
        n_KS[j,:]=0
        for i in range(pm.NE):
            n_KS[j,:]+=abs(U_t[j,:,i])**2
        J_KS[j,:]=CalculateCurrentDensity(n_KS,n_MB) # Calculate updated KS current density
        cost_J[j] = sum(abs(J_KS[j,:]-J_MB[j,:]))*pm.deltax 
        cost_n[j] = sum(abs(n_KS[j,:]-n_MB[j,:]))*pm.deltax   
        if cost_J[j]<cost_min:  # Keep present vector potential for reference if produces lower cost function evaluation
            cost_min=cost_J[j] 
            A_min[:]=A_KS[j,:]
            countmin=count 
        J_check=RE_Utilities.compare(J_KS,J_MB,tol) # Check if KS and exact current density are equal 
        if J_check:
            break
        count += 1
    string = 'REV: t = ' + str(j*pm.deltat) + ', tol = ' + str(tol) + ', cost_J = ' + str(cost_J[j]) + ', cost_Q = ' + str(cost_n[j])
    sprint.sprint(string,1,1,pm.msglvl)
    Apot[:]=0 # Change guage so only have scalar potential.
    for i in range(pm.jmax):
        for k in range(i+1):
            Apot[i]+=((A_KS[j,k]-A_KS[j-1,k])/pm.deltat)*pm.deltax
    V_KS[j,:]+=Apot[:]
    V_KS[j,:]+=V_KS[0,(pm.jmax-1)*0.5]-V_KS[j,(pm.jmax-1)*0.5] # Calculate full KS scalar potential								
    return 

# Main control function
def main(approx):
    global j, tol, z, U_t
    ReadInput(approx) # Read in exact charge density obtained from code
    GroundState(V_KS,n_MB,mu) # Calculate (or, if already obtained, check) ground-state KS potentia
    f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_' + str(approx) + '_vks.db', 'w') # KS potential	
    pickle.dump(V_KS[0,:],f)				
    f.close()
    f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_' + str(approx) + '_vh.db', 'w') # H potential	
    pickle.dump(V_h[0,:],f)				
    f.close()
    f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'gs_' + str(approx) + '_vxc.db', 'w') # XC potential	
    pickle.dump(V_xc[0,:],f)				
    f.close()
    tol = 1e-12 # Set inital convergence tolerance for exact and KS current densities
    for j in range(1,imax): # Propagate from the ground-state
        CalculateKohnSham(V_KS,A_KS,J_KS,U_t) # Calculate KS potential
        U_KS[j,:] = J_KS[j,:]/n_KS[j,:] # Calculate KS velocity field
        V_h[j,:] = Hartree(n_KS[j,:])
        V_Hxc[j,:] = V_KS[j,:]-V_ext[j,:]
        V_xc[j,:] = V_KS[j,:]-V_ext[j,:]-V_h[j,:]
    print
    if pm.TD==1:
         f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_' + str(approx) + '_vks.db', 'w') # KS potential	
         pickle.dump(V_KS[:,:],f)				
         f.close()
         f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_' + str(approx) + '_vh.db', 'w') # H potential	
         pickle.dump(V_h[:,:],f)				
         f.close()
         f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_' + str(approx) + '_vxc.db', 'w') # XC potential	
         pickle.dump(V_xc[:,:],f)				
         f.close()
         #f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_ext_jks.db', 'w') # current potential	
         #pickle.dump(J_KS[:,:],f)				
         #f.close()
         #f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_ext_gso.db', 'w') # ground-state orbital potential	
         #pickle.dump(Psi0[:,:].real,f)				
         #f.close()
         #f = open('outputs/' + str(pm.run_name) + '/raw/' + str(pm.run_name) + '_' + str(pm.NE) + 'td_ext_1eo.db', 'w') # 1st excited state orbital potential	
         #pickle.dump(Psi1[:,:].real,f)				
         #f.close()
    return

